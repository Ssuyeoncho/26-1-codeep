#!/usr/bin/env python3
"""Backward-compatible entry point for Phase 3.3 ImageNette-64.

Run:
    python3 phase3_3/phase3_cifar_experiment.py [--preset smoke|fast|full]

Or as a module from 2_R_with_two_modes/:
    python3 -m phase3_3.run [args]
"""
import sys
from pathlib import Path

# Allow running as a plain script (relative imports need the package on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase3_3.run import main  # noqa: E402

if __name__ == "__main__":
    main()
