"""Portfolio PI — guards research diversity, prevents lock-in, audits bridges."""

from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles._base_pi import (
    BasePI,
)


class PortfolioPI(BasePI):
    """PI role focused on portfolio balance, resource allocation, and exploration coverage."""

    role_name = "portfolio"
    role_ref = "task_role:portfolio_pi"
    prompt_template_name = "base.jinja2"
    private_kb_dir_name = "portfolio"

    def fixed_questions(self) -> list[str]:
        return self._fixed_questions_or(
            [
                "How much research budget is the current mainline absorbing?",
                "Which mechanisms were declared obsolete prematurely?",
                "Is anti-mainline forbidden_list growing too narrow?",
                "Which bridge target has already been covered? Continuing would be a no-op.",
                "Which previously failed lineage is worth reviving in current context?",
                "Are we keeping at least one non-mainline peer?",
            ]
        )
