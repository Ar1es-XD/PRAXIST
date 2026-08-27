"""Unit tests for :class:`praxist.core.run_config.RunConfig`.

``RunConfig`` is the only sanctioned bridge between ``os.environ`` and
the rest of the codebase under the configuration-discipline contract
documented in ``docs/concepts/config_discipline.md``.  These tests pin
the env-name → field mapping, the override semantics, the path/int
coercions, and the frozen-immutability contract.
"""

from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

from praxist.core.run_config import RunConfig


class RunConfigFromEnvironTest(unittest.TestCase):
    """Mapping from env vars to fields."""

    def test_empty_env_yields_documented_defaults(self) -> None:
        """An empty env produces the zero-value config (no exception)."""
        cfg = RunConfig.from_environ({})
        self.assertEqual(cfg.run_id, "")
        self.assertIsNone(cfg.run_dir)
        self.assertEqual(cfg.stage_id, "")
        self.assertIsNone(cfg.role_ref)
        self.assertEqual(cfg.agent_runtime_ref, "")
        self.assertEqual(cfg.agent_system, "")
        self.assertEqual(cfg.model_provider_ref, "")
        self.assertEqual(cfg.model_profile_ref, "")
        self.assertEqual(cfg.model, "")
        self.assertIsNone(cfg.workspace_root)
        self.assertIsNone(cfg.task_project_path)
        self.assertEqual(cfg.budget_grant_id, "")
        self.assertEqual(cfg.budget_request_id, "")
        # Explicit Codex override sentinel is the only non-empty runtime default.
        self.assertEqual(cfg.codex_bin, "codex")
        self.assertEqual(dict(cfg.subprocess_env), {})

    def test_run_identity_keys_propagate(self) -> None:
        """``PRAXIST_RUN_ID`` / ``PRAXIST_RUN_DIR`` / ``PRAXIST_STAGE_ID`` round-trip."""
        cfg = RunConfig.from_environ(
            {
                "PRAXIST_RUN_ID": "run-42",
                "PRAXIST_RUN_DIR": "/var/runs/42",
                "PRAXIST_STAGE_ID": "research_loop",
            }
        )
        self.assertEqual(cfg.run_id, "run-42")
        self.assertEqual(cfg.run_dir, Path("/var/runs/42"))
        self.assertEqual(cfg.stage_id, "research_loop")

    def test_role_ref_empty_string_normalizes_to_none(self) -> None:
        """``PRAXIST_ROLE_REF=`` (empty) means "unset" → None, not ""."""
        cfg = RunConfig.from_environ({"PRAXIST_ROLE_REF": ""})
        self.assertIsNone(cfg.role_ref)

    def test_role_ref_value_passes_through(self) -> None:
        cfg = RunConfig.from_environ({"PRAXIST_ROLE_REF": "role:peer"})
        self.assertEqual(cfg.role_ref, "role:peer")

    def test_agent_runtime_and_agent_system_are_independent(self) -> None:
        """Resolved ref and user-facing label are stored separately."""
        cfg = RunConfig.from_environ(
            {
                "PRAXIST_AGENT_RUNTIME_REF": "agent_runtime:codex_sdk",
                "PRAXIST_AGENT_SYSTEM": "codex_sdk",
            }
        )
        self.assertEqual(cfg.agent_runtime_ref, "agent_runtime:codex_sdk")
        self.assertEqual(cfg.agent_system, "codex_sdk")

    def test_model_provider_and_profile_propagate(self) -> None:
        cfg = RunConfig.from_environ(
            {
                "PRAXIST_MODEL_PROVIDER_REF": "model_provider:deepseek_alias",
                "PRAXIST_MODEL_PROFILE_REF": "strong_reasoner",
            }
        )
        self.assertEqual(cfg.model_provider_ref, "model_provider:deepseek_alias")
        self.assertEqual(cfg.model_profile_ref, "strong_reasoner")

    def test_praxist_model_overrides_legacy_agent_model(self) -> None:
        """``PRAXIST_MODEL`` is the V0.2 flag name and wins over ``AGENT_MODEL``."""
        cfg = RunConfig.from_environ(
            {"PRAXIST_MODEL": "deepseek-v4-pro", "AGENT_MODEL": "claude-opus-4-7"}
        )
        self.assertEqual(cfg.model, "deepseek-v4-pro")

    def test_legacy_agent_model_used_when_praxist_model_absent(self) -> None:
        """``AGENT_MODEL`` is honored for backward compatibility."""
        cfg = RunConfig.from_environ({"AGENT_MODEL": "claude-opus-4-7"})
        self.assertEqual(cfg.model, "claude-opus-4-7")

    def test_workspace_and_task_paths_expand_user(self) -> None:
        """Paths run through ``Path.expanduser`` so ``~/foo`` works."""
        cfg = RunConfig.from_environ(
            {"PRAXIST_WORKSPACE_ROOT": "~/ws", "PRAXIST_TASK_PROJECT_PATH": "~/ws/task"}
        )
        assert cfg.workspace_root is not None
        assert cfg.task_project_path is not None
        self.assertFalse(str(cfg.workspace_root).startswith("~"))
        self.assertFalse(str(cfg.task_project_path).startswith("~"))

    def test_codex_bin_round_trip(self) -> None:
        cfg = RunConfig.from_environ({"PRAXIST_CODEX_BIN": "/usr/local/bin/codex"})
        self.assertEqual(cfg.codex_bin, "/usr/local/bin/codex")


class RunConfigOverridesTest(unittest.TestCase):
    """``overrides=`` mirrors how argparse flags beat env vars."""

    def test_overrides_win_over_env(self) -> None:
        cfg = RunConfig.from_environ(
            {"PRAXIST_MODEL": "deepseek-v4-pro"},
            overrides={"model": "claude-opus-4-7"},
        )
        self.assertEqual(cfg.model, "claude-opus-4-7")

    def test_overrides_unknown_key_raises(self) -> None:
        """Misspelled override key fails loud at CLI assembly."""
        with self.assertRaises(TypeError):
            RunConfig.from_environ({}, overrides={"not_a_real_field": "x"})

    def test_overrides_can_set_path_fields(self) -> None:
        cfg = RunConfig.from_environ({}, overrides={"workspace_root": Path("/srv/ws")})
        self.assertEqual(cfg.workspace_root, Path("/srv/ws"))


class RunConfigImmutabilityTest(unittest.TestCase):
    """Frozen dataclass invariant — no in-place mutation."""

    def test_run_config_is_frozen(self) -> None:
        cfg = RunConfig.from_environ({"PRAXIST_RUN_ID": "r"})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.run_id = "other"  # type: ignore[misc]

    def test_with_subprocess_env_returns_new_instance(self) -> None:
        """``with_subprocess_env`` is the only sanctioned mutator (copy-on-write)."""
        cfg = RunConfig.from_environ({})
        copied = cfg.with_subprocess_env({"OPENAI_API_KEY": "redacted"})
        self.assertEqual(dict(copied.subprocess_env), {"OPENAI_API_KEY": "redacted"})
        # Original unchanged.
        self.assertEqual(dict(cfg.subprocess_env), {})
        self.assertIsNot(copied, cfg)


class S3DefaultsTest(unittest.TestCase):
    """#75 batch 8a — S3 / AWS literal defaults live in run_config."""

    def test_s3_default_constants_exported(self) -> None:
        from praxist.core import run_config

        self.assertEqual(run_config.DEFAULT_S3_BUCKET, "")
        self.assertEqual(run_config.DEFAULT_S3_ENDPOINT_URL, "")
        self.assertEqual(run_config.DEFAULT_S3_REGION, "us-east-1")
        self.assertEqual(run_config.DEFAULT_S3_IDEAS_PREFIX, "ideas/")
        self.assertEqual(run_config.DEFAULT_S3_RESULTS_PREFIX, "results/")
        self.assertEqual(run_config.DEFAULT_S3_FRONTIER_PREFIX, "frontier/")
        self.assertEqual(run_config.DEFAULT_AWS_ACCESS_KEY_ID, "")
        self.assertEqual(run_config.DEFAULT_AWS_SECRET_ACCESS_KEY, "")

    def test_config_shim_re_exports_match_run_config_defaults(self) -> None:
        """``praxist.config.X`` re-exports must equal the canonical
        ``run_config.DEFAULT_X`` value, so consumer code that reads
        ``config.X`` as a fallback sees the same default the run-config
        side defines.
        """
        from praxist import config
        from praxist.core import run_config

        self.assertEqual(config.S3_BUCKET, run_config.DEFAULT_S3_BUCKET)
        self.assertEqual(config.S3_ENDPOINT_URL, run_config.DEFAULT_S3_ENDPOINT_URL)
        self.assertEqual(config.S3_REGION, run_config.DEFAULT_S3_REGION)
        self.assertEqual(config.S3_IDEAS_PREFIX, run_config.DEFAULT_S3_IDEAS_PREFIX)
        self.assertEqual(config.S3_RESULTS_PREFIX, run_config.DEFAULT_S3_RESULTS_PREFIX)
        self.assertEqual(config.S3_FRONTIER_PREFIX, run_config.DEFAULT_S3_FRONTIER_PREFIX)
        self.assertEqual(config.AWS_ACCESS_KEY_ID, run_config.DEFAULT_AWS_ACCESS_KEY_ID)
        self.assertEqual(config.AWS_SECRET_ACCESS_KEY, run_config.DEFAULT_AWS_SECRET_ACCESS_KEY)

    def test_config_module_has_no_s3_env_reads_at_import(self) -> None:
        """Static check: ``praxist.config`` must not call os.getenv
        for any of the migrated S3 / AWS names. The point of the
        migration is to remove import-time env reads for these names.
        """
        import pathlib

        config_src = pathlib.Path(__file__).resolve().parents[2] / "praxist" / "config.py"
        text = config_src.read_text(encoding="utf-8")
        for name in (
            "S3_BUCKET",
            "S3_ENDPOINT_URL",
            "S3_IDEAS_PREFIX",
            "S3_RESULTS_PREFIX",
            "S3_FRONTIER_PREFIX",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ):
            with self.subTest(name=name):
                self.assertNotIn(f'os.getenv("{name}"', text)
        # ``AWS_REGION`` is the env name behind ``S3_REGION`` — same check.
        self.assertNotIn('os.getenv("AWS_REGION"', text)


class CohortDefaultsTest(unittest.TestCase):
    """#75 batch 8b — cohort / frontier literal defaults live in run_config."""

    def test_cohort_default_constants_exported(self) -> None:
        from praxist.core import run_config

        self.assertEqual(run_config.DEFAULT_MAX_GENERATIONS, 1)
        self.assertEqual(run_config.DEFAULT_COHORT_SIZE, 5)
        self.assertEqual(run_config.DEFAULT_PER_GENERATION_HOURS, 5)
        self.assertEqual(run_config.DEFAULT_PROMOTE_TOP_K, 2)
        self.assertEqual(run_config.DEFAULT_FRONTIER_STRATEGY, "auto")

    def test_config_shim_re_exports_cohort_defaults(self) -> None:
        from praxist import config
        from praxist.core import run_config

        self.assertEqual(config.MAX_GENERATIONS, run_config.DEFAULT_MAX_GENERATIONS)
        self.assertEqual(config.COHORT_SIZE, run_config.DEFAULT_COHORT_SIZE)
        self.assertEqual(config.PER_GENERATION_HOURS, run_config.DEFAULT_PER_GENERATION_HOURS)
        self.assertEqual(config.PROMOTE_TOP_K, run_config.DEFAULT_PROMOTE_TOP_K)
        self.assertEqual(config.FRONTIER_STRATEGY, run_config.DEFAULT_FRONTIER_STRATEGY)

    def test_config_module_has_no_cohort_env_reads_at_import(self) -> None:
        """``praxist.config`` must not ``os.getenv`` these names at import."""
        import pathlib

        config_src = pathlib.Path(__file__).resolve().parents[2] / "praxist" / "config.py"
        text = config_src.read_text(encoding="utf-8")
        for name in (
            "MAX_GENERATIONS",
            "COHORT_SIZE",
            "PER_GENERATION_HOURS",
            "PROMOTE_TOP_K",
            "FRONTIER_STRATEGY",
        ):
            with self.subTest(name=name):
                # The bare ``os.getenv("<NAME>"`` text must be gone;
                # the re-export ``DEFAULT_<NAME> as <NAME>`` survives.
                self.assertNotIn(f'os.getenv("{name}"', text)

    def test_dead_constants_removed_from_config_module(self) -> None:
        """``TASK_SPEC_PATH`` and ``SERVER_URL`` had zero readers; #75
        batch 8b deletes them.
        """
        from praxist import config

        self.assertFalse(hasattr(config, "TASK_SPEC_PATH"))
        self.assertFalse(hasattr(config, "SERVER_URL"))


class StartupAliasEnvOverridesTest(unittest.TestCase):
    """#75 batch 8b — ``PRAXIST_*`` preferred over legacy in startup overrides."""

    @staticmethod
    def _apply(env: dict[str, str]) -> list[dict[str, object]]:
        """Run the override path with a minimal task spec; return ``overrides_seen``."""
        from praxist.plugins.workflow_stages.research_loop.startup import (
            _apply_internal_env_overrides,
        )
        from praxist.task_spec import GenerationPolicy, MultiPIConfig, TaskSpec

        task_spec = TaskSpec(
            task_id="t",
            task_name="T",
            _task_dir="/tmp",
            generation_policy=GenerationPolicy(max_generations=1, cohort_size=2),
            multi_pi=MultiPIConfig(enabled=False, chair_peer_budget=0),
        )
        _, _, seen = _apply_internal_env_overrides(task_spec, {}, env)
        return seen

    def _env_name_for(self, seen: list[dict[str, object]], path: str) -> str | None:
        for row in seen:
            if row["path"] == path:
                return str(row["env"])
        return None

    def test_praxist_max_generations_wins_over_legacy(self) -> None:
        seen = self._apply({"PRAXIST_MAX_GENERATIONS": "7", "MAX_GENERATIONS": "3"})
        self.assertEqual(
            self._env_name_for(seen, "generation_policy.max_generations"), "PRAXIST_MAX_GENERATIONS"
        )
        self.assertEqual(
            next(r["value"] for r in seen if r["path"] == "generation_policy.max_generations"),
            7,
        )

    def test_legacy_name_used_when_praxist_name_absent(self) -> None:
        seen = self._apply({"MAX_GENERATIONS": "3"})
        self.assertEqual(
            self._env_name_for(seen, "generation_policy.max_generations"), "MAX_GENERATIONS"
        )

    def test_neither_set_yields_no_override(self) -> None:
        self.assertEqual(self._apply({}), [])

    def test_empty_string_treated_as_unset(self) -> None:
        seen = self._apply({"PRAXIST_MAX_GENERATIONS": "", "MAX_GENERATIONS": "3"})
        self.assertEqual(
            self._env_name_for(seen, "generation_policy.max_generations"), "MAX_GENERATIONS"
        )

    def test_cohort_size_alias_path(self) -> None:
        seen = self._apply({"PRAXIST_COHORT_SIZE": "8"})
        self.assertEqual(
            self._env_name_for(seen, "generation_policy.cohort_size"), "PRAXIST_COHORT_SIZE"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
