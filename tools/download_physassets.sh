#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PHYSWEEP_ROOT:-$(cd -- "${script_dir}/.." && pwd)}"
dataset_root="${project_root}/external/physassets"
archive_dir="${dataset_root}/archives"
proxy="${PHYSASSETS_SOCKS_PROXY:-socks5h://127.0.0.1:7891}"
base_url="https://huggingface.co/datasets/yaya234/PhysAssets/resolve/main"
expected_sha256="bce802e3b7cbfdb1d9824b2d164d0ad911b428251ed2694a6bc2ad4971425328"

parts=(
  dataset.tar.gz.part_00
  dataset.tar.gz.part_01
  dataset.tar.gz.part_02
  dataset.tar.gz.part_03
  dataset.tar.gz.part_04
  dataset.tar.gz.part_05
  dataset.tar.gz.part_06
)

mkdir -p "${archive_dir}" "${dataset_root}/index" "${dataset_root}/previews" \
  "${dataset_root}/candidates"

download() {
  local name="$1"
  curl \
    --location \
    --proxy "${proxy}" \
    --continue-at - \
    --retry 100 \
    --retry-delay 5 \
    --retry-all-errors \
    --speed-limit 1024 \
    --speed-time 120 \
    --output "${archive_dir}/${name}" \
    "${base_url}/${name}"
}

download README.md
for part in "${parts[@]}"; do
  download "${part}"
done

actual_sha256="$(
  cat "${parts[@]/#/${archive_dir}\/}" | sha256sum | awk '{print $1}'
)"
printf '%s  dataset.tar.gz\n' "${actual_sha256}" \
  > "${archive_dir}/combined.sha256"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
  echo "PhysAssets checksum mismatch" >&2
  exit 2
fi

date --iso-8601=seconds > "${dataset_root}/DOWNLOAD_COMPLETE"
echo "PhysAssets download and checksum verification complete."
