#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PHYSWEEP_ROOT:-$(cd -- "${script_dir}/.." && pwd)}"
dataset_root="${project_root}/external/physassets"
archive_dir="${dataset_root}/archives"
extract_dir="${dataset_root}/extracted"
index_dir="${dataset_root}/index"

parts=(
  dataset.tar.gz.part_00
  dataset.tar.gz.part_01
  dataset.tar.gz.part_02
  dataset.tar.gz.part_03
  dataset.tar.gz.part_04
  dataset.tar.gz.part_05
  dataset.tar.gz.part_06
)

if [[ ! -f "${dataset_root}/DOWNLOAD_COMPLETE" ]]; then
  echo "PhysAssets download is not verified." >&2
  exit 2
fi
for part in "${parts[@]}"; do
  if [[ ! -s "${archive_dir}/${part}" ]]; then
    echo "Missing archive part: ${part}" >&2
    exit 2
  fi
done
if [[ -f "${dataset_root}/EXTRACT_COMPLETE" ]]; then
  echo "PhysAssets is already extracted."
  exit 0
fi

mkdir -p "${extract_dir}" "${index_dir}"
cat "${parts[@]/#/${archive_dir}\/}" \
  | gzip -dc \
  | tar -xf - \
      --directory="${extract_dir}" \
      --strip-components=9 \
      --no-same-owner \
      --no-same-permissions \
      --checkpoint=10000 \
      --checkpoint-action='echo=extracted_files=%u'

sample_count="$(find "${extract_dir}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
file_count="$(find "${extract_dir}" -type f | wc -l)"
size_bytes="$(du -sb "${extract_dir}" | awk '{print $1}')"
png_file_count="$(find "${extract_dir}" -type f -iname '*.png' | wc -l)"
json_file_count="$(find "${extract_dir}" -type f -iname '*.json' | wc -l)"
mesh_file_count="$(
  find "${extract_dir}" -type f \
    \( -iname '*.glb' -o -iname '*.gltf' -o -iname '*.obj' \
       -o -iname '*.fbx' -o -iname '*.ply' -o -iname '*.stl' \
       -o -iname '*.usd' -o -iname '*.usda' -o -iname '*.usdc' \) \
    | wc -l
)"
cat > "${index_dir}/extraction_summary.json" <<EOF
{
  "sample_directory_count": ${sample_count},
  "file_count": ${file_count},
  "size_bytes": ${size_bytes},
  "png_file_count": ${png_file_count},
  "json_file_count": ${json_file_count},
  "mesh_file_count": ${mesh_file_count},
  "stripped_archive_components": 9
}
EOF
date --iso-8601=seconds > "${dataset_root}/EXTRACT_COMPLETE"
echo "PhysAssets extraction complete: samples=${sample_count} files=${file_count}"
