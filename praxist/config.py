"""
Environment-level defaults for Praxist compatibility paths.

Canonical task behavior comes from plugin descriptors plus explicit
CLI/environment overrides captured during startup. This module remains
a shared place for infrastructure knobs and legacy/direct caller fallbacks
— but is now a thin compatibility shim. Every constant exposed here is
re-exported from :mod:`praxist.core.run_config`, where the real
defaults live.

#75 batch 10 (A-class config.py collapse): the previous
RunPod / Docker / output-root constants in this file (``RUNPOD_API_KEY``,
``DOCKER_LOCAL_MODE``, ``DEFAULT_OUTPUT_ROOT``, …) were import-time
``os.getenv`` reads with no Python consumer — every real reader
(``infrastructure/runpod.py``, runpod deployment scripts) called
``os.environ.get(...)`` itself at the boundary. The dead constants
have been deleted; nothing in-tree imports them. External callers
that previously used them can either:

* read the env directly via ``os.environ.get(...)`` at their boundary
  (matches what every in-tree reader already does), or
* read the canonical default from
  ``praxist.core.run_config.DEFAULT_*``.
"""

# Re-export shims for legacy callers and the operational-surface contract
# test that patches ``config.LOGS_DIR`` / ``config.S3_BUCKET`` / etc.
# The real defaults live in ``praxist.core.run_config`` (#75
# batches 7b / 7c / 8a / 8b); these aliases keep
# ``praxist.config`` import-time free of ``os.getenv`` for the
# named constants while preserving the legacy import surface.
from praxist.core.run_config import (
    DEFAULT_AWS_ACCESS_KEY_ID as AWS_ACCESS_KEY_ID,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_AWS_SECRET_ACCESS_KEY as AWS_SECRET_ACCESS_KEY,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_COHORT_SIZE as COHORT_SIZE,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_FINDINGS_POLL_INTERVAL_SECONDS as FINDINGS_POLL_INTERVAL,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_FRONTIER_STRATEGY as FRONTIER_STRATEGY,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_FULL_AUTO_MAX_RUNTIME_SECONDS as FULL_AUTO_MAX_RUNTIME_SECONDS,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_LOCAL_FINDINGS_DIR as LOCAL_FINDINGS_DIR,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_LOGS_DIR as LOGS_DIR,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_MAX_GENERATIONS as MAX_GENERATIONS,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_PER_GENERATION_HOURS as PER_GENERATION_HOURS,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_PROMOTE_TOP_K as PROMOTE_TOP_K,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_S3_BUCKET as S3_BUCKET,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_S3_ENDPOINT_URL as S3_ENDPOINT_URL,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_S3_FRONTIER_PREFIX as S3_FRONTIER_PREFIX,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_S3_IDEAS_PREFIX as S3_IDEAS_PREFIX,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_S3_REGION as S3_REGION,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_S3_RESULTS_PREFIX as S3_RESULTS_PREFIX,  # noqa: F401 — re-export shim
)
from praxist.core.run_config import (
    DEFAULT_WORKSPACE_ROOT as WORKSPACE_DIR,  # noqa: F401 — re-export shim
)

# AGENT_MODEL moved to ``praxist.core.run_config.DEFAULT_AGENT_MODEL`` in
# #75 batch 7a. The env value is read by ``RunConfig.from_environ`` at the CLI
# boundary; downstream callers consume ``RunConfig.model`` instead. External
# callers that still need the raw env can read it directly via
# ``os.environ.get("AGENT_MODEL", DEFAULT_AGENT_MODEL)``.
