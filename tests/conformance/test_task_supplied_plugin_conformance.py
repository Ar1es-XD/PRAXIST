"""Conformance for task-supplied plugin discovery + resolution.

These tests close the second gap in
``tests/conformance/test_runtime_provider_replay_conformance.py`` — it
only loads bundled plugins (``templates/tasks/toy_math`` lives in-repo
and references bundled / test-fixture panel topologies). That leaves
the task-supplied plugin path — task projects shipping their own
plugins under ``<task>/.praxist/plugins/`` — uncovered, which is
exactly where #87 / #79 fired in production.

Each test materialises a *repo-external* task project in a temp dir,
ships a task-supplied panel_topology there, points the task descriptor
at it, and runs ``prepare_research_loop_plugin_run`` end-to-end. The
``plugin_resolution.json`` manifest is then asserted to have selected
the topology from ``source="task_project"`` — proving:

* ``PluginRoots.defaults(task_path=...)`` materialises the task root
  in the ``task_project`` list.
* ``PluginLoader.resolve(enforce_bundled_execution=True)`` accepts
  ``task_project`` alongside ``bundled`` (i.e. ``task_project`` is in
  ``TRUSTED_EXECUTION_SOURCES``).
* ``assert_bundled_execution_manifest`` does not reject the manifest.
* ``finalize_research_loop_plugin_run`` + ``verify_run`` round-trip
  cleanly when the panel topology came from the task path.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from praxist.core.replay import verify_run
from praxist.plugins.workflow_stages.research_loop.startup import (
    finalize_research_loop_plugin_run,
    prepare_research_loop_plugin_run,
)

_BUNDLED_TOY_MATH = Path("templates/tasks/toy_math")
_FAKE_TWO_ROUND_FIXTURE = Path("tests/fixtures/plugins/panel_topologies/fake_two_round")
_TASK_LOCAL_TOPOLOGY_NAME = "task_local_topology"


def _copy_toy_math(target: Path) -> None:
    """Copy the bundled toy_math task to ``target``.

    The destination must not yet exist; ``shutil.copytree`` populates it
    fully. Symlinks are dereferenced so the copy is hermetic and not
    sensitive to repo-relative paths after the move.
    """
    shutil.copytree(_BUNDLED_TOY_MATH, target, symlinks=False)


def _install_task_local_panel_topology(task_root: Path, *, name: str) -> Path:
    """Write a task-supplied panel_topology under ``task_root``.

    The topology is a near-copy of the ``fake_two_round`` fixture with
    ``name`` and ``topology_ref`` rewritten so the bundled
    ``fake_two_round`` does not satisfy the task's reference. This
    guarantees discovery must find the task-supplied entry for
    resolution to succeed.

    Returns the absolute path to the written ``plugin.yaml``.
    """
    plugin_dir = task_root / ".praxist" / "plugins" / "panel_topologies" / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    plugin_yaml = plugin_dir / "plugin.yaml"
    source_yaml = _FAKE_TWO_ROUND_FIXTURE / "plugin.yaml"
    spec = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    spec["name"] = name
    spec["description"] = f"Task-supplied topology (renamed copy of fake_two_round) for {name}."
    spec["topology"]["topology_ref"] = f"panel_topology:{name}"
    plugin_yaml.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return plugin_yaml


def _retarget_task_descriptor_to_topology(task_root: Path, *, topology_name: str) -> None:
    """Rewrite ``task.yaml`` to require ``panel_topology:<topology_name>``."""
    task_yaml = task_root / "task.yaml"
    spec = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
    spec.setdefault("praxist_plugins", {}).setdefault("panel", {})["topology"] = (
        f"panel_topology:{topology_name}"
    )
    task_yaml.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def _prepare_task_local(
    *,
    task_root: Path,
    workspace: Path,
    run_dir: Path,
):
    """Drive ``prepare_research_loop_plugin_run`` for a task-supplied panel."""
    return prepare_research_loop_plugin_run(
        task_project_path=task_root,
        workspace=workspace,
        run_dir=run_dir,
        runtime_ref="agent_runtime:fake_runtime",
        model_provider_ref="model_provider:fake_provider",
        budget_policy_ref="budget_policy:fake_tiered",
        model="fake-deterministic",
        local_mode=True,
        frontier_strategy="auto",
        credential_profile="fake_multi_key",
        command="cell-b task-supplied plugin conformance",
    )


class TaskSuppliedPluginConformanceTests(unittest.TestCase):
    """End-to-end conformance for the task-supplied plugin path."""

    def test_task_supplied_panel_topology_resolved_from_task_project_source(self) -> None:
        """The task-supplied panel_topology is selected with ``source='task_project'``.

        This is the cell that #79 (registry lost task-supplied roots) and
        #87 (task-supplied plugin discovery) lived in. Together they
        cover:

        * ``PluginRoots.defaults(task_path=...)`` surfaces the task root.
        * ``PluginLoader.resolve(enforce_bundled_execution=True)``
          accepts ``task_project`` candidates.
        * ``assert_bundled_execution_manifest`` accepts the resolved set.
        * ``finalize`` + ``verify_run`` round-trip cleanly.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            task_root = root / "task_project_external"
            _copy_toy_math(task_root)
            _install_task_local_panel_topology(task_root, name=_TASK_LOCAL_TOPOLOGY_NAME)
            _retarget_task_descriptor_to_topology(
                task_root, topology_name=_TASK_LOCAL_TOPOLOGY_NAME
            )

            run_dir = root / "run_task_supplied_panel"
            prepared = _prepare_task_local(
                task_root=task_root, workspace=workspace, run_dir=run_dir
            )
            finalize_research_loop_plugin_run(
                prepared,
                success=True,
                result={
                    "generations_completed": 0,
                    "run_dir": str(run_dir),
                    "exit_condition": "resolve_only",
                },
            )

            resolution = json.loads(
                (run_dir / "plugin_resolution.json").read_text(encoding="utf-8")
            )
            selected_by_key = {
                (item["metadata"]["kind"], item["metadata"]["name"]): item
                for item in resolution["selected"]
            }
            self.assertIn(
                ("panel_topology", _TASK_LOCAL_TOPOLOGY_NAME),
                selected_by_key,
                sorted(selected_by_key.keys()),
            )
            self.assertEqual(
                selected_by_key[("panel_topology", _TASK_LOCAL_TOPOLOGY_NAME)]["source"],
                "task_project",
            )
            self.assertEqual(resolution.get("execution_source_policy"), "bundled_only")
            self.assertTrue((run_dir / "task_project_manifest.json").exists())

            report = verify_run(run_dir)
            self.assertTrue(report["success"], report)

    def test_task_descriptor_referencing_task_supplied_topology_without_task_path_fails(
        self,
    ) -> None:
        """Negative cell: without the task-supplied plugin folder, resolution must fail.

        If somebody refactors ``PluginRoots.defaults`` to drop the
        ``task_project`` roots, or refactors ``PluginLoader.resolve`` to
        silently fall back to a bundled near-match, this test catches it
        — running the same task descriptor against a task path that has
        no ``.praxist/plugins/`` raises a resolution error.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            task_root = root / "task_project_external"
            _copy_toy_math(task_root)
            _retarget_task_descriptor_to_topology(
                task_root, topology_name=_TASK_LOCAL_TOPOLOGY_NAME
            )
            # Intentionally do NOT install the task-supplied plugin.

            run_dir = root / "run_missing_task_supplied"
            with self.assertRaises(Exception) as ctx:
                _prepare_task_local(task_root=task_root, workspace=workspace, run_dir=run_dir)
            self.assertIn(_TASK_LOCAL_TOPOLOGY_NAME, str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
