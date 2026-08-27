"""External Validity PI — high-stakes robustness and publication-level checks."""

from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles._base_pi import (
    BasePI,
)


class ExternalValidityPI(BasePI):
    """PI role focused on robustness, external validity, and benchmark transfer."""

    role_name = "external_validity"
    role_ref = "task_role:external_validity_pi"
    prompt_template_name = "base.jinja2"
    private_kb_dir_name = "external_validity"

    def fixed_questions(self) -> list[str]:
        return self._fixed_questions_or(
            [
                "Does the claim survive an independent implementation or harness family?",
                "Does it survive cross-scenario, cross-data-regime, or cross-environment variation?",
                "Does it survive more budget, longer horizon, or repeated trials?",
                "Which reviewer attack patterns apply to this claim?",
                "What language must the claim be bounded with until independent validation?",
            ]
        )
