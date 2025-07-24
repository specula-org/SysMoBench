#!/usr/bin/env python3
"""
Convenience script to test model connections.

Usage:
    python scripts/test_models.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import and run the test
from tests.test_models.test_model_connection import main

if __name__ == "__main__":
    main()