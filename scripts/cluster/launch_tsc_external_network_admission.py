#!/usr/bin/env python3
"""Launch the version-selected LA/Jinan full-horizon admission matrix."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cluster.launch_tsc_external_network_admission_v6 import main


if __name__ == "__main__":
    raise SystemExit(main())
