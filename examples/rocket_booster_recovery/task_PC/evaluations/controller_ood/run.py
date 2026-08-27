#!/usr/bin/env python3
"""Public task-owned evaluator entrypoint.

Accelerator binding is resolved before JAX or project code is imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[2]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from assets.harness.accelerator_binding import apply_scheduler_assignment  # noqa: E402


apply_scheduler_assignment(os.environ)

from evaluations.controller_ood.evaluator import main  # noqa: E402


if __name__ == "__main__":
    main()
