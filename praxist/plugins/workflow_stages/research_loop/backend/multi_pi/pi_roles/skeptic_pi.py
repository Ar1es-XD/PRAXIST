"""Skeptic PI — attacks the mainline narrative; ensures fair baselines and controls."""

from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles._base_pi import (
    BasePI,
)


class SkepticPI(BasePI):
    """PI role focused on falsification, weak evidence, and unsupported conclusions."""

    role_name = "skeptic"
    role_ref = "task_role:skeptic_pi"
    prompt_template_name = "base.jinja2"
    private_kb_dir_name = "skeptic"

    def fixed_questions(self) -> list[str]:
        return self._fixed_questions_or(
            [
                "Is this result merely a baseline/configuration advantage in disguise?",
                "Is the claim conflating parameter or search-space discovery with mechanism novelty?",
                "Is there an implementation / data / protocol / simulator artifact at play?",
                "What is the smallest experiment that would kill the current claim?",
                "Which words must be downgraded? (universal / dominance / obsolete / breakthrough)",
            ]
        )
