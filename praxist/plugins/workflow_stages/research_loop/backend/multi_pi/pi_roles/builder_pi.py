"""Builder PI — constructs the strongest coherent research mainline."""

from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles._base_pi import (
    BasePI,
)


class BuilderPI(BasePI):
    """PI role focused on constructive next-step hypotheses and experiment design."""

    role_name = "builder"
    role_ref = "task_role:builder_pi"
    # All PI roles share base.jinja2; differentiation comes from
    # fixed_questions, private_kb_dir, and the private_pack contents
    # (which are filtered by role in evidence_pack_builder).
    prompt_template_name = "base.jinja2"
    private_kb_dir_name = "builder"

    def fixed_questions(self) -> list[str]:
        return self._fixed_questions_or(
            [
                "What is the strongest mechanism explanation right now?",
                "Which mechanism pattern is most worth continuing to amplify or stress-test?",
                "Which Pareto arm has the most remaining headroom?",
                "Which findings can be synthesized into a cross-peer hypothesis?",
                "Which peer next gen should advance the mainline?",
            ]
        )
