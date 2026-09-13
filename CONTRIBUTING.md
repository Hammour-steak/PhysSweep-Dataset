# Contributing

1. Keep generated datasets, renders, checkpoints, caches, environments, and
   downloaded assets out of Git.
2. Preserve the dataset-only boundary documented in `README.md`.
3. With prepared assets, run `.venv/bin/python -m unittest discover -s tests`.
   Without assets, install `requirements.txt` and run `python tests/run_ci.py`;
   this skips the explicitly listed external-asset test modules, as GitHub CI does.
4. Update provenance and attribution when adding an asset or data source.
5. Never commit credentials, private keys, subscription URLs, or machine-local
   absolute paths.

Bug reports should include the command, configuration, relevant manifest IDs,
and the smallest reproducible example. Do not attach restricted source assets.
