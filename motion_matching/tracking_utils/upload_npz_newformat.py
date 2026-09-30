"""Compatibility entry point for the packaged PHP dataset uploader."""

import sys
from pathlib import Path

# Use this checkout when invoked directly from tracking_utils/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_matching.upload_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())
