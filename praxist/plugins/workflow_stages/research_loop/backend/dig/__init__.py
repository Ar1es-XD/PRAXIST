"""Deep Innovation Gate (DIG-Lite) support.

DIG-Lite is a pre-code design gate for research-loop peers. It is deliberately
generic: task projects provide domain semantics through their normal prompt and
task-spec context, while this package only enforces the planning contract.
"""

from .config import DIGLiteConfig, QualityDiversityConfig
from .runner import DIGLiteResult, run_dig_lite
from .schema import SelectedContract

__all__ = [
    "DIGLiteConfig",
    "DIGLiteResult",
    "QualityDiversityConfig",
    "SelectedContract",
    "run_dig_lite",
]
