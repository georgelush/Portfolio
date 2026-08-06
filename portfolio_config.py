"""Shared public portfolio configuration."""

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "portfolio.json"
with _CONFIG_PATH.open(encoding="utf-8") as config_file:
    _CONFIG = json.load(config_file)

CONTACT_EMAIL = _CONFIG["contactEmail"]
# Reserved for post-launch; do not surface until mailbox is active.
FUTURE_CONTACT_EMAIL = _CONFIG.get("futureContactEmail", "")
ASSISTANT_NAME = _CONFIG["assistantName"]
ASSISTANT_DESCRIPTION = _CONFIG.get(
    "assistantDescription",
    "Ask about my experience, technical projects, and availability.",
)
FLOWENTIC_SITE_ENABLED = bool(_CONFIG.get("flowenticSiteEnabled", False))
FLOWENTIC_SITE_URL = _CONFIG.get("flowenticSiteUrl", "https://flowentic.com")
