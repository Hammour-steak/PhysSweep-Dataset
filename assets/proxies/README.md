# Generated Collision Proxies

Collision meshes are generated artifacts and are not stored in Git. The compact
catalog and object-record index remain versioned so every expected proxy keeps a
stable ID, source reference, hash, and validation contract.

Install the frozen, validated proxies with `python3 generate.py --setup-only`
from the repository root. They are included in the runtime resource archive;
users do not need to regenerate them before sampling. See the
[generation guide](../../docs/GENERATION.md#setup-and-resources).
Source visual assets and licenses are recorded in
[THIRD_PARTY_ASSETS.json](../THIRD_PARTY_ASSETS.json).
