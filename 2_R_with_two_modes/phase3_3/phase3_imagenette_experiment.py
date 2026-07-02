#!/usr/bin/env python3
"""Entry point for Phase 3.3 ImageNette-64 mixture-schedule experiment.

Run:
    python3 phase3_3/phase3_imagenette_experiment.py [--preset smoke|fast|full]

Or as a module from 2_R_with_two_modes/:
    python3 -m phase3_3.run [args]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase3_3.run import main  # noqa: E402

if __name__ == "__main__":
    main()
