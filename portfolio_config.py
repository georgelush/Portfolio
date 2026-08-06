"""Shared public portfolio configuration."""

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "portfolio.json"
with _CONFIG_PATH.open(encoding="utf-8") as config_file:
    _CONFIG = json.load(config_file)

CONTACT_EMAIL = _CONFIG["contactEmail"]
ASSISTANT_NAME = _CONFIG["assistantName"]
