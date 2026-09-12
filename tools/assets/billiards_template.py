"""Refresh billiards catalog provenance only if the physical fixture is unchanged."""
from pathlib import Path
import copy
from tools.assets.physical_proxy_catalog import load_catalog, records_by_id
from tools.assets.static_support_proxy import compile_static_support_binding, validate_static_support_binding_files
from tools.core.hashing import relative_file_binding


def refresh_billiards_fixture(root: Path, template: dict) -> dict:
    billiards = copy.deepcopy(template)
    catalog_path = root / billiards["physical_proxy_catalog"]["path"]
    catalog, records = load_catalog(root, catalog_path, require_runtime_validation=True)
    old = billiards["physics"]["static_support_binding"]
    record = records_by_id(records)[old["asset_id"]]
    frame = old["target_support_frame"]
    current = compile_static_support_binding(
        record, usage_id=old["usage_id"],
        target_size_xy_m=frame["size_xy_m"],
        target_center_xy_m=frame["center_xy_m"],
        target_support_plane_z_m=frame["plane_z_m"],
        maximum_axis_scale_ratio=old["usage_contract"]["maximum_axis_scale_ratio"],
    )
    provenance_keys = {"catalog_record_sha256", "binding_sha256"}
    if ({key: value for key, value in old.items() if key not in provenance_keys}
            != {key: value for key, value in current.items() if key not in provenance_keys}):
        raise ValueError("current billiards catalog changes the frozen fixture binding")
    validate_static_support_binding_files(root, current)
    billiards["physics"]["static_support_binding"] = current
    billiards["physical_proxy_catalog"] = {
        **relative_file_binding(root, catalog_path), "records_sha256": catalog["records_sha256"],
    }
    return billiards
