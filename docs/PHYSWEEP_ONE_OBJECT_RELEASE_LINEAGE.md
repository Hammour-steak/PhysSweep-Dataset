# One-object release lineage

The released one-object datasets are immutable, hash-linked layers. A newer
release replaces complete 13-record groups and keeps every retained source
group byte-equivalent to its source release.

| Release | Added specialized family | Group distribution | Source boundary |
| --- | --- | --- | --- |
| v3 | `billiards` | 2240 generic, 928 asset proxy, 32 billiards | frozen v3 project |
| v4 | `passive_pinball` | 2208 generic, 928 asset proxy, 32 billiards, 32 passive pinball | frozen v3 plus the v4 worktree |
| v5 | `marble_run` | 2176 generic, 928 asset proxy, 32 billiards, 32 passive pinball, 32 marble run | frozen v4 plus the v5 worktree |

Every family in v5 uses the same one-object group contract: one canonical base,
four `mass_kg` values, four `contact_friction` values, and four
`contact_restitution` values. A derived record targets object index 0 and may
change only the named material field.

## Source-root rule

Paths and implementation hashes in retained metadata must be resolved against
the `source_project_root` recorded by the release manifest. In particular, the
v4 passive-pinball metadata binds the v4 specialized registry, while new v5
marble-run metadata binds the v5 registry. Resolving both against whichever
worktree happens to be current is invalid.

The version-specific v4 preparation and publication entry points are retained
only in the frozen source history at
`feature/passive-pinball-v4@29aa9c238542c03a9ddbeb34db16852fa7f39514`.
They are deliberately absent from the forward development checkout, so an old
publisher cannot be mistaken for the active extension path.

New extensions use the declarative path:

- `configs/<extension>_release_extension.json`
- `tools/prepare_specialized_release_replacements.py`
- `tools/publish_specialized_release_extension.py`

The compatibility entry points must not be copied to create another family.
Reproduce v4 in its frozen source root; use the declarative tools for every new
extension.

## Render order and provenance

Render canonical bases first. Derived sweeps are a separate explicit stage and
must not start until the complete base set passes file, numeric, camera, and
visual audits. Generic render commands are bound to the release metadata
manifest hash and to `physweep_pybullet_rigid_metadata_v1`; stale bound scenes
with matching names but different release provenance are rejected.

Specialized render inputs are produced by
`tools/prepare_sweep_render_manifests.py`, which verifies the release metadata,
physics manifests, source metadata hashes, complete 13-record groups, and the
registered source schema before writing a branch-specific render plan.

Never overwrite a published release directory. Publish a replacement extension
to a new directory, compare hashes and record sets, then promote it explicitly.
Audit historical code bindings against their frozen owner rather than the
current checkout:

```bash
python tools/audit_release_provenance.py \
  --release-manifest datasets/one_object_v5/release/manifest.json \
  --release-project-root /path/to/frozen-v5-worktree
```
