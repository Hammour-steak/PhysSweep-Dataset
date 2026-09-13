"""Strict release/lineage reading shared by object-count source selectors.

This module verifies provenance and localizes hash-verified collision resources.
Shape and host eligibility belong to the consumer's scene rules; an old
admission is never a new scene's admission.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json
from tools.core.paths import safe_scene_id

RELEASE_SCHEMA_PAIRS = {
    'physweep_base_release_view_v14': ('physweep_base_pipeline_view_v12', 'physweep_base_sample_v11'),
    'physweep_base_release_view_v15': ('physweep_base_pipeline_view_v13', 'physweep_base_sample_v12'),
}


def declared_within(root: Path, value: Path) -> Path:
    root = root.resolve()
    candidate = value if value.is_absolute() else root / value
    absolute = Path(os.path.abspath(candidate))
    absolute.relative_to(root)
    # Frozen generation trees may contain relocated, hash-bound dataset links.
    # Preserve their declared namespace; every consumed file is hash-verified.
    return absolute


def release_metadata_records(*, root: Path, released_path: Path,
                             released: dict[str, Any], families: tuple[str, ...]) -> dict:
    try:
        pipeline_schema, sample_schema = RELEASE_SCHEMA_PAIRS[released['schema_version']]
    except KeyError as error:
        raise ValueError('unsupported released base schema') from error
    base_root = released_path.parent.resolve()
    base_root.relative_to(root.resolve())
    result = {}
    for family in families:
        binding = released.get('pipelines', {}).get(family)
        if not isinstance(binding, dict) or set(binding) != {'manifest', 'manifest_sha256'}:
            raise ValueError(f'released 1obj base lacks the {family} pipeline')
        path = declared_within(base_root, Path(str(binding['manifest'])))
        if sha256(path) != binding['manifest_sha256']:
            raise ValueError(f'released {family} pipeline manifest hash mismatch')
        manifest = read_json(path)
        records = manifest.get('records')
        if (manifest.get('schema_version') != pipeline_schema or manifest.get('pipeline') != family
                or not isinstance(records, list) or manifest.get('sample_count') != len(records)):
            raise ValueError(f'released {family} pipeline manifest is invalid')
        for record in records:
            scene_id = safe_scene_id(str(record.get('scene_id', '')))
            metadata_path = declared_within(base_root, path.parent / scene_id / 'metadata.json')
            expected_hash = str(record.get('metadata_sha256', ''))
            if scene_id in result or not expected_hash or sha256(metadata_path) != expected_hash:
                raise ValueError(f'released {family} sample is invalid: {scene_id}')
            metadata = read_json(metadata_path)
            if metadata.get('schema_version') != sample_schema or metadata.get('scene_id') != scene_id:
                raise ValueError(f'released {family} metadata identity is invalid')
            result[scene_id] = (family, metadata_path, metadata, expected_hash)
    return result


def source_reference(*, root: Path, source_root: Path, generation_record: dict,
                     generation_metadata: dict, release_record: tuple) -> dict:
    family, release_path, _, release_hash = release_record
    generation_path = declared_within(source_root, Path(str(generation_record['path'])))
    return {'scene_id': str(generation_metadata['scene_id']), 'source_family': family,
            'release_metadata': {'path': release_path.relative_to(root).as_posix(), 'sha256': release_hash},
            'generation_metadata': {'path': generation_path.relative_to(source_root).as_posix(),
                                    'sha256': str(generation_record['metadata_sha256'])}}


def verified_generation_records(*, root: Path, released_base_manifest_path: Path,
                                source_root: Path, source_manifest_path: Path,
                                source_contract: dict, family_schemas: dict) -> list[dict]:
    """Verify the whole selected lineage before returning reusable raw inputs."""
    root, source_root = root.resolve(), source_root.resolve()
    released_path = declared_within(root, released_base_manifest_path)
    generation_path = declared_within(source_root, source_manifest_path)
    released, generation = read_json(released_path), read_json(generation_path)
    if released.get('schema_version') != source_contract['released_base_manifest_schema_version']:
        raise ValueError('released 1obj base manifest has the wrong schema')
    schema = source_contract['generation_manifest_schema_version']
    provenance = released.get('provenance', {}).get('source_generation_release_metadata', {})
    if provenance.get('schema_version') != schema or provenance.get('manifest_sha256') != sha256(generation_path):
        raise ValueError('released 1obj base does not name this generation manifest')
    records = generation.get('records')
    if (generation.get('schema_version') != schema or generation.get('dataset_id') != released.get('dataset_id')
            or not isinstance(records, list) or generation.get('sample_count') != len(records)
            or generation.get('group_count') != released.get('sample_count')):
        raise ValueError('generation manifest contradicts the released 1obj base')
    bases = [r for r in records if r.get('kind') == source_contract['sample_kind']]
    ids = [str(r.get('scene_id', '')) for r in bases]
    if len(bases) != generation['group_count'] or '' in ids or len(set(ids)) != len(ids):
        raise ValueError('generation manifest has invalid canonical base records')
    released_records = release_metadata_records(root=root, released_path=released_path,
        released=released, families=tuple(family_schemas.values()))
    selected = [r for r in bases if r.get('source_schema_version') in family_schemas]
    if {r['scene_id'] for r in selected} != set(released_records):
        raise ValueError('released samples differ from generation metadata')
    result = []
    for record in selected:
        scene_id, schema = record['scene_id'], record['source_schema_version']
        released_record = released_records[scene_id]
        path = declared_within(source_root, Path(record['path']))
        if sha256(path) != record['metadata_sha256']:
            raise ValueError(f'source metadata changed after release: {scene_id}')
        metadata = read_json(path)
        if (released_record[0] != family_schemas[schema] or metadata.get('schema_version') != schema
                or metadata.get('scene_id') != scene_id
                or released_record[2].get('lineage', {}).get('source_generation_metadata_sha256') != record['metadata_sha256']):
            raise ValueError(f'source metadata identity is invalid: {scene_id}')
        result.append({'record': record, 'metadata': metadata, 'release_metadata': released_record[2],
            'source': source_reference(root=root, source_root=source_root, generation_record=record,
                                      generation_metadata=metadata, release_record=released_record)})
    return result


def localize_marble_source_rows(rows,source_root,data_root,collision_directory):
    """Copy verified collision contents into the new work; retain original bindings."""
    collision_directory=Path(collision_directory)
    collision_directory.resolve().relative_to(data_root.resolve())
    result=copy.deepcopy(rows)
    for row in result:
        bindings=[]
        for c in row['metadata']['physics']['fixture']['mesh_components']:
            original=copy.deepcopy(c['collision']);source=source_root/original['path']
            if sha256(source)!=original['sha256']:raise ValueError('original marble collision hash mismatch')
            target=collision_directory/(original['sha256']+'.obj')
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha256(target)!=original['sha256']:raise ValueError('localized marble collision changed')
            else:
                with target.open('xb') as output:output.write(source.read_bytes())
                if sha256(target)!=original['sha256']:raise ValueError('marble collision copy failed')
            c['collision']['path']=target.relative_to(data_root).as_posix()
            bindings.append({'component_id':c['id'],'original':{'path':str(source),'sha256':original['sha256']},'localized':copy.deepcopy(c['collision'])})
        row['fixture_resource_localization']=bindings
    return result
