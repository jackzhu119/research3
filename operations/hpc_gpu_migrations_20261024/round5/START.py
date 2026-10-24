#!/usr/bin/env python3
"""Prepare Round 5, optionally start its detached rolling controller. Stdlib only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper3r5.submit import main

if __name__ == "__main__":
    raise SystemExit(main())
