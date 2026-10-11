#!/usr/bin/env python3
"""Validate the spin example using the shared TLC runner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validate import main

if __name__ == "__main__":
    sys.exit(
        main(
            Path(__file__).resolve().parent,
            "TV_AcquireLock",
            Path(__file__).resolve().parent / "windows_AcquireLock.ndjson",
        )
    )
