#!/usr/bin/env python3
"""Publish or verify the canonical three-object dataset using shared contracts."""
from pathlib import Path
from tools.cli.build_object_dataset import load_config as load_object_config
from tools.cli.build_object_dataset import publish_dataset as publish_object_dataset
from tools.cli.build_object_dataset import verify_dataset as verify_object_dataset,run_cli

PROJECT_ROOT=Path(__file__).resolve().parents[2]
EXPECTED_OBJECT_COUNT=3


def load_config(path):
    return load_object_config(path,expected_object_count=EXPECTED_OBJECT_COUNT)


def publish_dataset(**kwargs):
    return publish_object_dataset(**kwargs,expected_object_count=EXPECTED_OBJECT_COUNT)


def verify_dataset(release_root):
    return verify_object_dataset(release_root,expected_object_count=EXPECTED_OBJECT_COUNT)


def main():
    run_cli(description=__doc__,default_config=Path('configs/datasets/three_object.json'),project_root=PROJECT_ROOT,
            load_config_fn=load_config,publish_dataset_fn=publish_dataset,verify_dataset_fn=verify_dataset)


if __name__=='__main__':main()
