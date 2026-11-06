#!/usr/bin/env python3

from pathlib import Path
import yaml

def load_config():
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

    with open(config_path, "r") as file:
        return yaml.safe_load(file)
