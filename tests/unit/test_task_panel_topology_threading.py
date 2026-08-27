"""Regression tests for ``TaskSpec.panel_topology_ref`` threading.

A custom task project declares its panel topology under
``praxist_plugins.panel.topology`` in the task descriptor (e.g.
the rocket_8e_panel project's four-PI topology with its interpreter
role). Before the fix, ``legacy_two_round_executor`` hardcoded the
legacy ``panel_topology:legacy_multi_pi_two_round`` ref at five call
sites, so panels silently fell back to the three-PI shape and the
declared interpreter role never ran.

These tests pin the threading from the descriptor through each layer:

* ``load_task_spec`` reads ``praxist_plugins.panel.topology``.
* ``_select_pi_roles`` / ``_select_pi_role_specs`` / ``_has_high_stakes_signal``
  honor the ``topology_ref`` kwarg.
* ``run_panel`` forwards ``topology_ref`` into those helpers.
* ``PIAgent`` carries the ref into ``run_panel``.
* ``GenerationLoop`` pulls the ref off ``TaskSpec`` and passes it to
  ``PIAgent``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.core.panel_topology import PanelRoleSpec, PanelTopologySpec
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
    legacy_two_round_executor as executor_mod,
)
from praxist.task_spec import load_task_spec

_LEGACY_REF = "panel_topology:legacy_multi_pi_two_round"
_CUSTOM_REF = "panel_topology:rocket_8e_panel"

_FOUR_PI_ROLES = ["skeptic", "adversary", "interpreter", "reviewer"]


def _build_four_pi_spec() -> PanelTopologySpec:
    """A fake four-PI topology spec mirroring the rocket_8e_panel shape."""
    role_specs = {
        role_id: PanelRoleSpec(
            role_id=role_id,
            legacy_role_id=role_id,
            role_ref=f"task_role:{role_id}",
            role_kind="pi",
            private_pack_key=role_id,
        )
        for role_id in _FOUR_PI_ROLES
    }
    return PanelTopologySpec(
        topology_ref=_CUSTOM_REF,
        modes={"full": list(_FOUR_PI_ROLES)},
        rounds=[],
        role_specs=role_specs,
    )


def _build_custom_chair_spec() -> PanelTopologySpec:
    base = _build_four_pi_spec()
    return PanelTopologySpec(
        topology_ref=base.topology_ref,
        modes=base.modes,
        rounds=base.rounds,
        role_specs=base.role_specs,
        chair_role_ref="task_role:custom_chair",
    )


class TaskSpecPanelTopologyRefTest(unittest.TestCase):
    """``load_task_spec`` exposes ``praxist_plugins.panel.topology``."""

    def _write_task(self, tmp: Path, praxist_plugins_block: str) -> Path:
        (tmp / "description.md").write_text("d", encoding="utf-8")
        (tmp / "prompt_task.jinja2").write_text("p", encoding="utf-8")
        spec_path = tmp / "task.yaml"
        spec_path.write_text(
            f"""task_id: t
task_name: T
description_file: description.md
research_direction: r
{praxist_plugins_block}
""",
            encoding="utf-8",
        )
        return spec_path

    def test_descriptor_topology_propagates_to_task_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = self._write_task(
                Path(tmp),
                praxist_plugins_block=(
                    f"praxist_plugins:\n  panel:\n    topology: {_CUSTOM_REF}\n"
                ),
            )
            spec = load_task_spec(str(spec_path))
            self.assertEqual(spec.panel_topology_ref, _CUSTOM_REF)

    def test_missing_descriptor_falls_back_to_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = self._write_task(Path(tmp), praxist_plugins_block="")
            spec = load_task_spec(str(spec_path))
            self.assertEqual(spec.panel_topology_ref, _LEGACY_REF)

    def test_malformed_topology_value_falls_back_to_legacy(self) -> None:
        """Non-string topology values are ignored — better safe than mute."""
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = self._write_task(
                Path(tmp),
                praxist_plugins_block=(
                    "praxist_plugins:\n  panel:\n    topology: [not_a_string]\n"
                ),
            )
            spec = load_task_spec(str(spec_path))
            self.assertEqual(spec.panel_topology_ref, _LEGACY_REF)


class SelectHelpersHonorTopologyRefTest(unittest.TestCase):
    """``_select_pi_*`` / ``_has_high_stakes_signal`` route via the ref."""

    def test_select_pi_roles_routes_to_provided_topology(self) -> None:
        spec = _build_four_pi_spec()
        with patch.object(
            executor_mod,
            "panel_topology_for_ref",
            return_value=spec,
        ) as resolver:
            roles = executor_mod._select_pi_roles("full", topology_ref=_CUSTOM_REF)
        resolver.assert_called_once_with(_CUSTOM_REF, registry=None)
        self.assertEqual(roles, _FOUR_PI_ROLES)

    def test_select_pi_role_specs_routes_to_provided_topology(self) -> None:
        spec = _build_four_pi_spec()
        with patch.object(
            executor_mod,
            "panel_topology_for_ref",
            return_value=spec,
        ) as resolver:
            specs = executor_mod._select_pi_role_specs("full", topology_ref=_CUSTOM_REF)
        resolver.assert_called_once_with(_CUSTOM_REF, registry=None)
        self.assertEqual([s.role_id for s in specs], _FOUR_PI_ROLES)

    def test_select_chair_role_ref_routes_to_provided_topology(self) -> None:
        spec = _build_custom_chair_spec()
        with patch.object(
            executor_mod,
            "panel_topology_for_ref",
            return_value=spec,
        ) as resolver:
            role_ref = executor_mod._select_chair_role_ref(topology_ref=_CUSTOM_REF)
        resolver.assert_called_once_with(_CUSTOM_REF, registry=None)
        self.assertEqual(role_ref, "task_role:custom_chair")

    def test_has_high_stakes_signal_routes_to_provided_topology(self) -> None:
        spec = _build_four_pi_spec()
        with patch.object(
            executor_mod,
            "panel_topology_for_ref",
            return_value=spec,
        ) as resolver:
            executor_mod._has_high_stakes_signal({}, topology_ref=_CUSTOM_REF)
        resolver.assert_called_once_with(_CUSTOM_REF, registry=None)

    def test_default_kwarg_preserves_legacy_resolution(self) -> None:
        """No-arg helpers still resolve the legacy topology — the v1 contract
        single-task tests rely on."""
        with patch.object(
            executor_mod,
            "panel_topology_for_ref",
            return_value=_build_four_pi_spec(),
        ) as resolver:
            executor_mod._select_pi_roles("full")
        resolver.assert_called_once_with(_LEGACY_REF, registry=None)

    def test_select_pi_roles_forwards_registry_to_resolver(self) -> None:
        """#151: when ``registry`` is supplied, ``_select_pi_roles`` must
        thread it into ``panel_topology_for_ref`` so the resolver sees
        the same registry the orchestrator built (with task-project
        plugin roots). Without this, a task-level custom topology is
        invisible and the panel silently falls back to single-PI.
        """
        sentinel_registry = object()
        with patch.object(
            executor_mod,
            "panel_topology_for_ref",
            return_value=_build_four_pi_spec(),
        ) as resolver:
            executor_mod._select_pi_roles(
                "full",
                topology_ref=_CUSTOM_REF,
                registry=sentinel_registry,
            )
        resolver.assert_called_once_with(_CUSTOM_REF, registry=sentinel_registry)

    def test_select_pi_role_specs_forwards_registry_to_resolver(self) -> None:
        sentinel_registry = object()
        with patch.object(
            executor_mod,
            "panel_topology_for_ref",
            return_value=_build_four_pi_spec(),
        ) as resolver:
            executor_mod._select_pi_role_specs(
                "full",
                topology_ref=_CUSTOM_REF,
                registry=sentinel_registry,
            )
        resolver.assert_called_once_with(_CUSTOM_REF, registry=sentinel_registry)

    def test_has_high_stakes_signal_forwards_registry_to_resolver(self) -> None:
        sentinel_registry = object()
        with patch.object(
            executor_mod,
            "panel_topology_for_ref",
            return_value=_build_four_pi_spec(),
        ) as resolver:
            executor_mod._has_high_stakes_signal(
                {}, topology_ref=_CUSTOM_REF, registry=sentinel_registry
            )
        resolver.assert_called_once_with(_CUSTOM_REF, registry=sentinel_registry)


class PIAgentForwardsTopologyRefTest(unittest.TestCase):
    """``PIAgent.synthesize`` forwards ``panel_topology_ref`` to ``run_panel``."""

    def _make_agent(self, tmp: Path, *, topology_ref: str | None):
        from praxist.plugins.workflow_stages.research_loop.backend import (
            pi_agent as pi_mod,
        )
        from praxist.task_spec import MultiPIConfig

        run_dir = tmp / "run"
        run_dir.mkdir()
        workspace = tmp / "ws"
        workspace.mkdir()
        kwargs: dict = dict(
            run_dir=run_dir,
            workspace=workspace,
            cohort_size=3,
            model="fake-model",
            reasoning_effort="high",
            use_multi_pi_panel=True,
            multi_pi_config=MultiPIConfig(enabled=True, panel_mode_default="full", n_rounds=1),
        )
        if topology_ref is not None:
            kwargs["panel_topology_ref"] = topology_ref
        return pi_mod, pi_mod.PIAgent(**kwargs), run_dir

    def test_pi_agent_passes_topology_ref_into_run_panel(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock

        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.legacy_two_round_executor import (
            PanelResult,
        )

        with tempfile.TemporaryDirectory() as tmp:
            pi_mod, agent, run_dir = self._make_agent(Path(tmp), topology_ref=_CUSTOM_REF)
            self.assertEqual(agent.panel_topology_ref, _CUSTOM_REF)

            fake_run_panel = AsyncMock(
                return_value=PanelResult(
                    success=True,
                    panel_mode="full",
                    agenda={"k": "v"},
                ),
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.multi_pi.run_panel",
                fake_run_panel,
            ):
                asyncio.run(
                    agent._run_multi_pi_panel(
                        completed_gen_id=0,
                        out_path=run_dir / "agenda.yaml",
                    )
                )
            fake_run_panel.assert_awaited_once()
            kwargs = fake_run_panel.await_args.kwargs
            self.assertEqual(kwargs.get("topology_ref"), _CUSTOM_REF)
            self.assertEqual(kwargs.get("reasoning_effort"), "high")

    def test_pi_agent_forwards_plugin_registry_into_run_panel(self) -> None:
        """#151: when the orchestrator hands ``PIAgent`` a plugin registry
        (with task-project roots already attached), the agent must
        forward it into ``run_panel`` so the topology resolver finds a
        task-level ``praxist_plugins.panel.topology`` plugin instead
        of silently falling back to single-PI.
        """
        import asyncio
        from unittest.mock import AsyncMock

        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.legacy_two_round_executor import (  # noqa: E501
            PanelResult,
        )

        with tempfile.TemporaryDirectory() as tmp:
            pi_mod, agent, run_dir = self._make_agent(Path(tmp), topology_ref=_CUSTOM_REF)
            sentinel_registry = object()
            agent._plugin_registry = sentinel_registry

            fake_run_panel = AsyncMock(
                return_value=PanelResult(
                    success=True,
                    panel_mode="full",
                    agenda={"k": "v"},
                ),
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.multi_pi.run_panel",
                fake_run_panel,
            ):
                asyncio.run(
                    agent._run_multi_pi_panel(
                        completed_gen_id=0,
                        out_path=run_dir / "agenda.yaml",
                    )
                )
            kwargs = fake_run_panel.await_args.kwargs
            self.assertIs(kwargs.get("registry"), sentinel_registry)

    def test_pi_agent_omits_registry_when_not_provided(self) -> None:
        """No registry → don't pass ``registry`` so ``run_panel`` keeps
        its own default (None) and legacy contract stays intact.
        """
        import asyncio
        from unittest.mock import AsyncMock

        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.legacy_two_round_executor import (  # noqa: E501
            PanelResult,
        )

        with tempfile.TemporaryDirectory() as tmp:
            pi_mod, agent, run_dir = self._make_agent(Path(tmp), topology_ref=_CUSTOM_REF)
            self.assertIsNone(agent._plugin_registry)

            fake_run_panel = AsyncMock(
                return_value=PanelResult(
                    success=True,
                    panel_mode="full",
                    agenda={"k": "v"},
                ),
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.multi_pi.run_panel",
                fake_run_panel,
            ):
                asyncio.run(
                    agent._run_multi_pi_panel(
                        completed_gen_id=0,
                        out_path=run_dir / "agenda.yaml",
                    )
                )
            kwargs = fake_run_panel.await_args.kwargs
            self.assertNotIn("registry", kwargs)

    def test_pi_agent_omits_topology_ref_when_not_set(self) -> None:
        """No custom topology → don't pass ``topology_ref`` so ``run_panel``
        applies its own legacy default and existing pinning tests stay green.
        """
        import asyncio
        from unittest.mock import AsyncMock

        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.legacy_two_round_executor import (
            PanelResult,
        )

        with tempfile.TemporaryDirectory() as tmp:
            pi_mod, agent, run_dir = self._make_agent(Path(tmp), topology_ref=None)
            self.assertIsNone(agent.panel_topology_ref)

            fake_run_panel = AsyncMock(
                return_value=PanelResult(
                    success=True,
                    panel_mode="full",
                    agenda={"k": "v"},
                ),
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.multi_pi.run_panel",
                fake_run_panel,
            ):
                asyncio.run(
                    agent._run_multi_pi_panel(
                        completed_gen_id=0,
                        out_path=run_dir / "agenda.yaml",
                    )
                )
            kwargs = fake_run_panel.await_args.kwargs
            self.assertNotIn("topology_ref", kwargs)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
