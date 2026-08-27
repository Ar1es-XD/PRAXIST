"""PI role definitions."""

from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles._base_pi import (
    BasePI,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles.builder_pi import (
    BuilderPI,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles.external_validity_pi import (
    ExternalValidityPI,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles.portfolio_pi import (
    PortfolioPI,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles.skeptic_pi import (
    SkepticPI,
)

ROLE_REGISTRY = {
    "builder": BuilderPI,
    "skeptic": SkepticPI,
    "portfolio": PortfolioPI,
    "external_validity": ExternalValidityPI,
}

__all__ = [
    "BasePI",
    "BuilderPI",
    "SkepticPI",
    "PortfolioPI",
    "ExternalValidityPI",
    "ROLE_REGISTRY",
]
