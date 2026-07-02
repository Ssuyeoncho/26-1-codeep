#!/usr/bin/env python3
"""Entry point — keeps the original run command working.

Run:
    python3 phase2_2/phase2_mnist_experiment.py [args]

Or as a module from 2_R_with_two_modes/:
    python3 -m phase2_2.run [args]
"""
import sys
from pathlib import Path

# Allow running as a plain script (relative imports need the package on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2_2.run import main  # noqa: E402

if __name__ == "__main__":
    main()
