from __future__ import annotations

# Shared utilities used across the stock_prediction package.
# Import from here instead of redeclaring in every module.

import os
import sys

# ---------------------------------------------------------------------------
# .env loading — walks up from this file's location to find the project root
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    # File is at: backend/stock_prediction/utils.py
    # Walk up: stock_prediction/ → backend/ → project root
    _here = os.path.dirname(os.path.abspath(__file__))
    _candidates = [_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))]
    for _candidate in _candidates:
        _env_path = os.path.join(_candidate, ".env")
        if os.path.exists(_env_path):
            load_dotenv(_env_path)
            break
except ImportError:
    pass


# ---------------------------------------------------------------------------
# sys.path — ensures project root is importable regardless of working directory
# ---------------------------------------------------------------------------

def ensure_project_on_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


# ---------------------------------------------------------------------------
# ANSI terminal colours
# ---------------------------------------------------------------------------

GREEN  = "\033[92m"
BLUE   = "\033[94m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Output directory — all charts and results land here
# ---------------------------------------------------------------------------

OUTPUT_DIR = "results"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
