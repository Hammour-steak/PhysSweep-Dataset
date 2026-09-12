"""Prepare the pinned runtime and assets used by the public generators."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import venv

from tools.assets.sketchfab_download import request_json
from tools.assets.sketchfab_policy import noai_declared, require_glb_download

BLENDER = 'blender-3.4.0-linux-x64'
BLENDER_SHA256 = 'f9aaf69339e4aad3b7927a4cf7ba372453e393df662c92bc1fedf94e0c6b5382'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def within(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in str(relative):
        raise ValueError('Resource paths must be relative and remain inside the destination')
    result = (root/path).resolve()
    result.relative_to(root.resolve())
    return result


def download(url, output, expected):
    """Publish only complete checksum-matching downloads; preserve existing assets."""
    if output.exists():
        if digest(output) != expected:
            raise ValueError(f'Existing resource has a different checksum: {output}')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        temporary = None
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'PhysSweep/1.0'})
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as target:
                temporary = Path(target.name)
                import certifi
                context = ssl.create_default_context(cafile=os.environ.get('SSL_CERT_FILE') or certifi.where())
                with urllib.request.urlopen(request, timeout=60, context=context) as response:
                    shutil.copyfileobj(response, target)
            if digest(temporary) != expected:
                raise ValueError(f'Download checksum mismatch: {output.name}')
            temporary.replace(output)
            return
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def extract_checked(archive, destination):
    """Extract into a fresh directory, rejecting traversal and external links."""
    with tarfile.open(archive) as package:
        members = package.getmembers()
        names = set()
        for member in members:
            path = within(destination, member.name)
            if member.name in names:
                raise ValueError('Duplicate archive member')
            names.add(member.name)
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError('Unsupported archive entry')
            if member.issym() or member.islnk():
                target = Path(member.linkname)
                if target.is_absolute():
                    raise ValueError('Absolute archive link')
                target = (path.parent if member.issym() else destination)/target
                target.resolve().relative_to(destination.resolve())
        # Files come before links, so archive paths cannot write through an archive-created link.
        for member in members:
            path = within(destination, member.name)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with package.extractfile(member) as source, path.open('xb') as target:
                    shutil.copyfileobj(source, target)
                path.chmod(member.mode & 0o777)
        for member in members:
            if member.issym() or member.islnk():
                path = within(destination, member.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                if member.issym():
                    path.symlink_to(member.linkname)
                else:
                    os.link(within(destination, member.linkname), path)


def setup_environment(root):
    if platform.system() != 'Linux' or platform.machine() not in {'x86_64', 'AMD64'}:
        raise RuntimeError('The packaged renderer requires Linux x86_64 and an NVIDIA GPU')
    if sys.version_info < (3, 10):
        raise RuntimeError('Python 3.10 or newer is required')
    if not shutil.which('gcc') or not shutil.which('nvidia-smi'):
        raise RuntimeError('Install GCC and the NVIDIA driver before running setup')
    environment = root/'.venv'
    executable = environment/'bin/python'
    if not executable.exists():
        venv.EnvBuilder(with_pip=True, symlinks=True).create(environment)
    requirements = root/'requirements.txt'
    marker = environment/'physweep_requirements.sha256'
    expected = digest(requirements)
    if not marker.exists() or marker.read_text().strip() != expected:
        subprocess.run([str(executable), '-m', 'pip', 'install', '-r', str(requirements)], check=True)
        marker.write_text(expected+'\n')
    return executable


def install_blender(root):
    executable = root/'runtime'/BLENDER/'blender'
    if executable.exists():
        version = subprocess.check_output([str(executable), '--version'], text=True)
        if not version.startswith('Blender 3.4.0'):
            raise ValueError('The existing Blender runtime is not version 3.4.0')
        return
    cache = root/'cache/downloads'
    archive = cache/(BLENDER+'.tar.xz')
    download('https://download.blender.org/release/Blender3.4/'+archive.name, archive, BLENDER_SHA256)
    runtime = root/'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime) as directory:
        staging = Path(directory)
        extract_checked(archive, staging)
        (staging/BLENDER).rename(runtime/BLENDER)


def resource_url(record):
    if 'url' in record:
        return record['url']
    token = os.environ.get('SKETCHFAB_API_TOKEN', '')
    if not token:
        raise RuntimeError('Set SKETCHFAB_API_TOKEN to download the reviewed Sketchfab models')
    uid = record['sketchfab_uid']
    model = request_json('https://api.sketchfab.com/v3/models/'+uid, token)
    if model.get('license', {}).get('slug') not in {'by', 'cc0'} or noai_declared(model):
        raise ValueError(f'The current model policy does not allow this download: {uid}')
    result = request_json('https://api.sketchfab.com/v3/models/'+uid+'/download', token)
    return require_glb_download(result)['url']


def install_resources(root):
    descriptor = json.loads((root/'assets/manifests/runtime_resources.json').read_text())
    inventory = json.loads((root/'assets/manifests/runtime_resources_inventory.json').read_text())['files']
    missing = []
    for record in inventory:
        path = within(root, record['path'])
        if path.exists():
            if digest(path) != record['sha256']:
                raise ValueError(f'Existing resource changed: {record["path"]}')
        else:
            missing.append(record)
    if not missing:
        return
    archive = root/'cache/downloads'/descriptor['filename']
    download(descriptor['url'], archive, descriptor['sha256'])
    with tempfile.TemporaryDirectory(dir=archive.parent) as directory:
        staging = Path(directory)
        extract_checked(archive, staging)
        expected_paths = {record['path'] for record in inventory}
        actual_paths = {p.relative_to(staging).as_posix() for p in staging.rglob('*') if p.is_file()}
        if actual_paths != expected_paths:
            raise ValueError('Resource archive does not match its inventory')
        for record in inventory:
            if digest(within(staging, record['path'])) != record['sha256']:
                raise ValueError('Resource archive file checksum mismatch')
        for record in missing:
            target = within(root, record['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            within(staging, record['path']).rename(target)


def prepare_assets(root):
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    install_blender(root)
    install_resources(root)
    records = json.loads((root/'assets/manifests/runtime_downloads.json').read_text())['files']
    def fetch(record):
        target = within(root, record['path'])
        if target.exists():
            if digest(target) != record['sha256']:
                raise ValueError(f'Existing asset changed: {record["path"]}')
        else:
            download(resource_url(record), target, record['sha256'])
    with ThreadPoolExecutor(max_workers=8) as executor:
        for index, _ in enumerate(executor.map(fetch, records), 1):
            if index % 50 == 0 or index == len(records):
                print(f'Assets verified: {index}/{len(records)}', flush=True)
    if not shutil.which('ffmpeg'):
        import imageio_ffmpeg
        target = root/'.venv/bin/ffmpeg'
        if not target.exists():
            target.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ['PATH'] = str(root/'.venv/bin')+os.pathsep+os.environ.get('PATH', '')
    print('Runtime and assets are ready.', flush=True)
