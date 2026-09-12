"""Verified released source pools and their actual three-object capacity."""
from collections import Counter
from pathlib import Path

import numpy as np

from tools.sampling.released_asset_sources import asset_object_template
from tools.sampling.released_asset_hosts import asset_host_template
from tools.sampling.released_object_sources import verified_generation_records
from tools.scene_rules.three_object import validate_host


def _source_family_schemas(source: dict) -> dict[str, str]:
    schemas = source.get("generation_metadata_schemas")
    if schemas is None:
        schemas = {"generic": source["generation_metadata_schema"]}
    if (
        not isinstance(schemas, dict)
        or set(schemas) != set(source["families"])
        or any(not isinstance(value, str) or not value for value in schemas.values())
        or len(set(schemas.values())) != len(schemas)
    ):
        raise ValueError("three-object source family schemas are invalid")
    return {schema: family for family, schema in schemas.items()}


def load_sources(
    *,
    root: Path,
    source_root: Path,
    source_manifest: Path,
    released_manifest: Path,
    rules: dict,
) -> dict:
    source = rules["matrix"]["source"]
    rows = verified_generation_records(
        root=root,
        released_base_manifest_path=released_manifest,
        source_root=source_root,
        source_manifest_path=source_manifest,
        source_contract={
            "released_base_manifest_schema_version": source["release_schema"],
            "generation_manifest_schema_version": source["generation_manifest_schema"],
            "sample_kind": "base",
        },
        family_schemas=_source_family_schemas(source),
    )
    objects = []
    hosts = []
    rejections = Counter()
    mixed = rules["matrix"].get("geometry_scope") in {"mixed_flat_R_v1","motion_diversity_v1","multi_mesh_pair_roles_v1"}
    compound_enabled = rules["matrix"].get("coverage_mode") in {"compound_asset_proxy_v1","multi_mesh_pair_roles_v1"}
    mesh_hosts_enabled = rules["matrix"].get("coverage_mode") == "exact_mesh_host_v1"
    registry_cache = {}
    visual_hash_cache = {}
    family_counts = Counter()
    asset_host_sources = {}
    for original in sorted(rows, key=lambda row: row["source"]["scene_id"]):
        family = original["source"]["source_family"]
        family_counts[family] += 1
        row = original
        if family == "asset":
            support_asset_id = str(original["metadata"]["assets"]["support_asset_id"])
            asset_host_sources.setdefault(support_asset_id, original)
            template, reason = asset_object_template(
                source_root=source_root,
                runtime_root=root,
                generation_metadata=original["metadata"],
                release_metadata=original["release_metadata"],
                eligibility=source["asset_eligibility"],
                registry_cache=registry_cache,
                visual_hash_cache=visual_hash_cache,
                include_compound_inertia_reference=compound_enabled,
            )
            if template is None:
                rejections["asset normalization: " + reason] += 1
                continue
            row = {**original, "metadata": template}
        elif family != "generic":
            raise ValueError("unsupported three-object source family")

        obj = row["metadata"]["simulation"]["objects"][0]
        geometry = obj["geometry"]
        size = np.asarray(geometry["size_m"])
        if (
            geometry["type"] == "sphere"
            and np.isfinite(size).all()
            and (size > 0).all()
            and np.allclose(size, size[0], rtol=0, atol=1e-9)
        ):
            objects.append(row)
        elif mixed and geometry["type"] in {"cuboid", "cylinder"}:
            from tools.core.primitive_support import initial_extents

            if obj["collision_profile"]["type"] == "compound" and not compound_enabled:
                rejections[
                    "object: support extent calculation requires a matching primitive proxy"
                ] += 1
                continue
            try:
                initial_extents(obj)
            except ValueError as error:
                rejections["object: " + str(error)] += 1
            else:
                objects.append(row)
        if family == "generic":
            try:
                validate_host(row["metadata"], rules["scene"])
            except ValueError as error:
                rejections[str(error)] += 1
            else:
                hosts.append(row)
    if mesh_hosts_enabled:
        required = list(rules["matrix"]["required_support_asset_ids"])
        excluded = list(rules["matrix"].get("excluded_support_asset_ids", []))
        if set(asset_host_sources) != set(required) | set(excluded):
            raise ValueError("released exact-mesh host ids differ from the frozen scope")
        donors_by_environment = {}
        for row in hosts:
            category = row["metadata"]["appearance"]["scene_visual"]["environment_category"]
            donors_by_environment.setdefault(category, []).append(row)
        for values in donors_by_environment.values():
            values.sort(key=lambda value: value["source"]["scene_id"])
        categories = sorted(donors_by_environment)
        if not categories or sum(map(len, donors_by_environment.values())) < len(required):
            raise ValueError("insufficient released room donors for exact-mesh hosts")
        donor_offsets = Counter()
        overrides = rules["matrix"].get("effective_safe_surface_overrides", {})
        mesh_hosts = []
        for index, asset_id in enumerate(required):
            category = categories[index % len(categories)]
            offset = donor_offsets[category]
            if offset >= len(donors_by_environment[category]):
                raise ValueError("exact-mesh host room donor budget is exhausted")
            donor = donors_by_environment[category][offset]
            donor_offsets[category] += 1
            original = asset_host_sources[asset_id]
            template = asset_host_template(
                source_root=source_root,
                runtime_root=root,
                generation_metadata=original["metadata"],
                release_metadata=original["release_metadata"],
                environment_host_template=donor["metadata"],
                effective_surface_override=overrides.get(asset_id),
                registry_cache=registry_cache,
            )
            validate_host(template, rules["scene"], allow_exact_mesh=True)
            mesh_hosts.append({
                "metadata": template,
                "source": original["source"],
                "environment_source": donor["source"],
            })
        hosts.extend(mesh_hosts)
    if len(objects) < 3 or not hosts:
        raise ValueError("insufficient verified three-object source capacity")
    object_families = Counter(row["source"]["source_family"] for row in objects)
    primitive_asset_ids = {
        row["metadata"]["simulation"]["objects"][0]["visual_profile"]["id"]
        for row in objects
        if row["source"]["source_family"] == "asset"
        and row["metadata"]["simulation"]["objects"][0]["collision_profile"]["type"]
        == row["metadata"]["simulation"]["objects"][0]["geometry"]["type"]
    }
    compound_asset_ids = {
        row["metadata"]["simulation"]["objects"][0]["visual_profile"]["id"]
        for row in objects
        if row["source"]["source_family"] == "asset"
        and row["metadata"]["simulation"]["objects"][0]["collision_profile"]["type"]
        == "compound"
    }
    if source["families"] == ["generic"]:
        summary = {
            "generic_lineage_records": len(rows),
            "sphere_source_records": sum(
                row["metadata"]["simulation"]["objects"][0]["geometry"]["type"]
                == "sphere"
                for row in objects
            ),
            "host_source_records": len(hosts),
            **(
                {
                    "exact_mesh_host_asset_ids": sorted(
                        row["metadata"]["simulation"]["support"]["asset_id"]
                        for row in hosts
                        if row["metadata"]["simulation"]["support"].get("exact_static_binding")
                        and "asset_id" in row["metadata"]["simulation"]["support"]
                    )
                }
                if mesh_hosts_enabled
                else {}
            ),
            **(
                {
                    "object_sources_by_shape": dict(
                        Counter(
                            row["metadata"]["simulation"]["objects"][0]["geometry"]["type"]
                            for row in objects
                        )
                    )
                }
                if mixed
                else {}
            ),
            "host_classes": dict(
                Counter(
                    row["metadata"]["simulation"]["support"]["scene_class"]
                    for row in hosts
                )
            ),
            "object_visual_assets": len(
                {
                    row["metadata"]["simulation"]["objects"][0]["visual_profile"]["id"]
                    for row in objects
                }
            ),
            "host_rejection_reasons": dict(rejections),
        }
        return {"objects": objects, "hosts": hosts, "summary": summary}
    return {
        "objects": objects,
        "hosts": hosts,
        "summary": {
            "verified_lineage_records_by_family": dict(family_counts),
            "eligible_object_records_by_family": dict(object_families),
            "sphere_source_records": sum(
                row["metadata"]["simulation"]["objects"][0]["geometry"]["type"]
                == "sphere"
                for row in objects
            ),
            "host_source_records": len(hosts),
            **(
                {
                    "exact_mesh_host_asset_ids": sorted(
                        row["metadata"]["simulation"]["support"]["asset_id"]
                        for row in hosts
                        if row["metadata"]["simulation"]["support"].get("exact_static_binding")
                        and "asset_id" in row["metadata"]["simulation"]["support"]
                    )
                }
                if mesh_hosts_enabled
                else {}
            ),
            **(
                {
                    "object_sources_by_shape": dict(
                        Counter(
                            row["metadata"]["simulation"]["objects"][0]["geometry"]["type"]
                            for row in objects
                        )
                    )
                }
                if mixed
                else {}
            ),
            "host_classes": dict(
                Counter(
                    row["metadata"]["simulation"]["support"]["scene_class"]
                    for row in hosts
                )
            ),
            "object_visual_assets": len(
                {
                    row["metadata"]["simulation"]["objects"][0]["visual_profile"]["id"]
                    for row in objects
                }
            ),
            "primitive_asset_ids": sorted(primitive_asset_ids),
            **(
                {"compound_asset_ids": sorted(compound_asset_ids)}
                if compound_enabled
                else {}
            ),
            "host_and_object_rejection_reasons": dict(rejections),
        },
    }
