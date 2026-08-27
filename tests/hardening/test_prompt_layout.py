from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.core.prompt_layout import (
    PromptBlock,
    PromptLayout,
    audit_frozen_blocks,
    build_legacy_jinja_prompt_layout,
    find_dynamic_markers,
)
from praxist.core.replay import verify_run
from praxist.core.storage import ArtifactWriter
from praxist.plugins.workflow_stages.research_loop.backend.agent import (
    BaseAgent,
    resolve_prompt_with_layout,
)
from praxist.testing.fake_workflow_fixture import run_fake_workflow_fixture


class Step20PromptLayoutTests(unittest.TestCase):
    def test_legacy_jinja_layout_keeps_frozen_hash_stable_across_dynamic_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _write(root / "prompt_base.jinja2", "Base for {{ peer_id }} in {{ run_dir }}\n")
            task = _write(root / "prompt_task.jinja2", "Task: {{ task_name }}\n")
            generation = _write(
                root / "prompt_generation.jinja2",
                "Frontier={{ frontier_summary }}\nGraph={{ graph_session_context }}\n",
            )
            context_a = {
                "peer_id": "gen0_peer0",
                "run_dir": "/runs/a",
                "task_name": "sam",
                "frontier_summary": "frontier a",
                "graph_session_context": "graph a",
            }
            context_b = {
                **context_a,
                "peer_id": "gen3_peer7",
                "run_dir": "/runs/b",
                "frontier_summary": "frontier b",
                "graph_session_context": "graph b",
            }

            layout_a = build_legacy_jinja_prompt_layout(
                base_template_path=base,
                task_prompt_path=task,
                generation_template_path=generation,
                context=context_a,
                run_id="run_a",
                stage_id="research_loop",
                prompt_id="gen_0/peer_0",
                agent_runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:openrouter",
                repo_root=root,
            )
            layout_b = build_legacy_jinja_prompt_layout(
                base_template_path=base,
                task_prompt_path=task,
                generation_template_path=generation,
                context=context_b,
                run_id="run_b",
                stage_id="research_loop",
                prompt_id="gen_3/peer_7",
                agent_runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:openrouter",
                repo_root=root,
            )

            self.assertEqual(layout_a.frozen_prefix_hash, layout_b.frozen_prefix_hash)
            self.assertEqual(layout_a.semi_static_hash, layout_b.semi_static_hash)
            self.assertNotEqual(layout_a.dynamic_payload_hash, layout_b.dynamic_payload_hash)
            self.assertEqual(audit_frozen_blocks(layout_a.blocks)["status"], "pass")
            self.assertTrue(
                any(block.partition == "semi_static_run_context" for block in layout_a.blocks)
            )
            dynamic_blocks = [
                block for block in layout_a.blocks if block.partition == "dynamic_payload"
            ]
            self.assertTrue(dynamic_blocks)
            legacy_dynamic_blocks = [block for block in dynamic_blocks if block.legacy_renderer]
            self.assertTrue(legacy_dynamic_blocks)
            self.assertIn("run_dir", legacy_dynamic_blocks[0].dynamic_markers_in_template)
            self.assertEqual(dynamic_blocks[-1].block_id, "session_start_command")
            self.assertFalse(dynamic_blocks[-1].legacy_renderer)
            self.assertTrue(
                layout_a.prompt_text.rstrip().endswith("proceed with the research\nworkflow.")
            )
            self.assertEqual(layout_a.cache_mode, "runtime_auto_cache")
            self.assertEqual(layout_a.runtime_cache_strategy, "runtime_auto_cache")

    def test_provider_explicit_cache_is_only_selected_for_non_claude_sdk_anthropic_messages(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _write(root / "prompt_base.jinja2", "Base\n")
            layout = build_legacy_jinja_prompt_layout(
                base_template_path=base,
                task_prompt_path=None,
                generation_template_path=None,
                context={},
                run_id="run",
                stage_id="research_loop",
                prompt_id="prompt",
                agent_runtime_ref="agent_runtime:codex_sdk",
                model_provider_ref="model_provider:anthropic_messages",
                repo_root=root,
            )

            self.assertEqual(layout.cache_mode, "provider_explicit_cache")
            self.assertIsNone(layout.runtime_cache_strategy)
            self.assertEqual(layout.provider_cache_strategy, "anthropic_messages_cache_control")

    def test_peer_prompt_renders_bundled_role_descriptions_when_no_override(self) -> None:
        """Issue #85: with no ``peer_role_descriptions`` provided, the bundled

        five-bullet vocabulary block continues to render (regression guard
        for the in-place refactor that moved those bullets behind an
        ``{% include %}`` partial).
        """
        from praxist.plugins.workflow_stages.research_loop.backend.agent import (
            resolve_prompt_with_layout,
        )

        repo_root = (
            Path(__file__).resolve().parent.parent.parent
            / "praxist/plugins/workflow_stages/research_loop/backend"
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            prompt_text, _ = resolve_prompt_with_layout(
                base_template_path=out_dir / "missing_base.jinja2",
                task_prompt_path=None,
                generation_template_path=repo_root / "prompt_generation.jinja2",
                output_path=out_dir / "gen0_peer0_prompt.md",
                layout_output_path=out_dir / "gen0_peer0_layout.json",
                context={
                    "peer_id": "gen0_peer0",
                    "gen_id": 0,
                    "research_agenda": {
                        "peer_contracts": {
                            "gen0_peer0": {
                                "role": "exploit",
                                "target_hypothesis": "H1",
                                "success_signal": "ship a frontier-quality variant",
                            }
                        }
                    },
                },
            )
        # All five bundled role names appear when no override is provided.
        for role in ("exploit", "falsifier", "bridge", "anti_mainline"):
            self.assertIn(f"`{role}`", prompt_text)

    def test_peer_prompt_renders_task_local_role_description_when_provided(self) -> None:
        """Issue #85: when the panel topology supplied a description for the

        bound role, render that single bullet instead of the bundled
        five-bullet vocabulary block.
        """
        from praxist.plugins.workflow_stages.research_loop.backend.agent import (
            resolve_prompt_with_layout,
        )

        repo_root = (
            Path(__file__).resolve().parent.parent.parent
            / "praxist/plugins/workflow_stages/research_loop/backend"
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            prompt_text, _ = resolve_prompt_with_layout(
                base_template_path=out_dir / "missing_base.jinja2",
                task_prompt_path=None,
                generation_template_path=repo_root / "prompt_generation.jinja2",
                output_path=out_dir / "gen0_peer0_prompt.md",
                layout_output_path=out_dir / "gen0_peer0_layout.json",
                context={
                    "peer_id": "gen0_peer0",
                    "gen_id": 0,
                    "research_agenda": {
                        "peer_contracts": {
                            "gen0_peer0": {
                                "role": "specialist_a",
                                "target_hypothesis": "H1",
                                "success_signal": "lead the ablation arm",
                            }
                        }
                    },
                    "peer_role_descriptions": {
                        "specialist_a": "lead the ablation arm; publish T2 results only",
                    },
                },
            )
        self.assertIn("specialist_a", prompt_text)
        self.assertIn("lead the ablation arm; publish T2 results only", prompt_text)
        # Bundled five-bullet vocabulary must not leak when the bound role
        # has a task-local description.
        for role in ("exploit", "falsifier", "bridge", "anti_mainline"):
            self.assertNotIn(f"`{role}` —", prompt_text)

    def test_resolve_prompt_with_layout_writes_prompt_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _write(root / "prompt_base.jinja2", "Base {{ peer_id }}\n")
            output = root / "gen_0" / "peer_prompt.md"
            manifest_path = root / "gen_0" / "peer_prompt_layout.json"
            env = {
                "PRAXIST_RUN_ID": "run_step20",
                "PRAXIST_STAGE_ID": "research_loop",
                "PRAXIST_AGENT_RUNTIME_REF": "agent_runtime:claude_sdk",
                "PRAXIST_MODEL_PROVIDER_REF": "model_provider:openrouter",
                "PRAXIST_WORKSPACE_ROOT": str(root),
            }

            with patch.dict(os.environ, env, clear=False):
                prompt_text, manifest = resolve_prompt_with_layout(
                    base_template_path=base,
                    task_prompt_path=None,
                    generation_template_path=None,
                    output_path=output,
                    layout_output_path=manifest_path,
                    context={"peer_id": "gen0_peer0"},
                    prompt_id="gen_0/gen0_peer0",
                )

            self.assertTrue(output.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(output.read_text(encoding="utf-8"), prompt_text)
            self.assertTrue(prompt_text.startswith("# Praxist Research Agent Stable Contract"))
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest, manifest)
            self.assertEqual(manifest["schema_version"], "praxist.prompt_layout.v1")
            self.assertEqual(manifest["layout_version"], "praxist.prompt_layout.v1")
            self.assertEqual(manifest["cache_mode"], "runtime_auto_cache")
            self.assertEqual(manifest["frozen_audit"]["status"], "pass")
            self.assertTrue(str(manifest["rendered_prompt_hash"]).startswith("sha256:"))

    def test_base_agent_request_uses_prompt_layout_cache_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _write(root / "prompt_base.jinja2", "Base {{ peer_id }}\n")
            manifest = build_legacy_jinja_prompt_layout(
                base_template_path=base,
                task_prompt_path=None,
                generation_template_path=None,
                context={"peer_id": "gen0_peer0"},
                run_id="run_step20",
                stage_id="research_loop",
                prompt_id="prompt",
                agent_runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:openrouter",
                repo_root=root,
            ).manifest()
            env = {
                "PRAXIST_RUN_ID": "run_step20",
                "PRAXIST_STAGE_ID": "research_loop",
                "PRAXIST_AGENT_RUNTIME_REF": "agent_runtime:claude_sdk",
                "PRAXIST_MODEL_PROVIDER_REF": "model_provider:openrouter",
                "PRAXIST_MODEL_CREDENTIAL_KEY_ID": "openrouter:env:abc123",
                "PRAXIST_MODEL_PROFILE_REF": "cheap_peer",
            }

            with patch.dict(os.environ, env, clear=False):
                request = BaseAgent(
                    name="peer0",
                    allowed_tools=["Read"],
                    workspace=root,
                    mcp_servers={},
                    model="anthropic/claude-opus-4.7",
                    prompt_layout_manifest=manifest,
                )._build_agent_run_request("inspect task", {"ANTHROPIC_AUTH_TOKEN": "token"})

            self.assertEqual(request.prompt_ref["kind"], "prompt_layout_v1")
            self.assertEqual(request.prompt_ref["layout_hash"], manifest["layout_hash"])
            self.assertEqual(request.cache_policy.mode, "runtime_auto_cache")
            self.assertEqual(
                request.cache_policy.frozen_prefix_hash, manifest["frozen_prefix_hash"]
            )
            self.assertEqual(request.cache_policy.runtime_cache_strategy, "runtime_auto_cache")
            self.assertEqual(
                request.runtime_options["prompt_layout"]["layout_hash"], manifest["layout_hash"]
            )

    def test_replay_accepts_valid_prompt_layout_artifact_and_rejects_dynamic_frozen_block(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            valid_run_dir = workspace / "valid_layout"
            valid_result = run_fake_workflow_fixture(
                workspace=workspace,
                run_dir=valid_run_dir,
                credential_profile="fake_multi_key",
            )
            valid_run_id = json.loads((valid_run_dir / "run.json").read_text(encoding="utf-8"))[
                "run_id"
            ]
            valid_manifest, valid_prompt = _prompt_layout_fixture(workspace, valid_run_id)
            _persist_prompt_layout_artifacts(
                valid_run_dir, valid_run_id, valid_manifest, valid_prompt
            )

            valid_report = verify_run(Path(valid_result["run_dir"]))
            self.assertTrue(valid_report["success"], valid_report)

            invalid_run_dir = workspace / "invalid_layout"
            run_fake_workflow_fixture(
                workspace=workspace,
                run_dir=invalid_run_dir,
                credential_profile="fake_multi_key",
            )
            invalid_run_id = json.loads((invalid_run_dir / "run.json").read_text(encoding="utf-8"))[
                "run_id"
            ]
            bad_block = PromptBlock(
                block_id="bad_frozen",
                partition="frozen_prefix",
                renderer="static_text",
                text="This illegal frozen block mentions {{ run_dir }}.\n",
                dynamic_markers_in_rendered=find_dynamic_markers(
                    "This illegal frozen block mentions {{ run_dir }}."
                ),
            )
            bad_layout = PromptLayout(
                run_id=invalid_run_id,
                stage_id="research_loop",
                prompt_id="bad_prompt",
                agent_runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:openrouter",
                cache_mode="runtime_auto_cache",
                runtime_cache_strategy="runtime_auto_cache",
                provider_cache_strategy=None,
                blocks=[bad_block],
            )
            _persist_prompt_layout_artifacts(
                invalid_run_dir, invalid_run_id, bad_layout.manifest(), bad_layout.prompt_text
            )

            invalid_report = verify_run(invalid_run_dir)
            self.assertFalse(invalid_report["success"], invalid_report)
            self.assertTrue(
                any(
                    "frozen block bad_frozen contains dynamic markers" in error
                    for error in invalid_report["errors"]
                ),
                invalid_report,
            )


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _prompt_layout_fixture(root: Path, run_id: str) -> tuple[dict[str, object], str]:
    base = _write(root / "layout_fixture" / "prompt_base.jinja2", "Base {{ peer_id }}\n")
    layout = build_legacy_jinja_prompt_layout(
        base_template_path=base,
        task_prompt_path=None,
        generation_template_path=None,
        context={"peer_id": "gen0_peer0"},
        run_id=run_id,
        stage_id="research_loop",
        prompt_id="gen_0/gen0_peer0",
        agent_runtime_ref="agent_runtime:claude_sdk",
        model_provider_ref="model_provider:openrouter",
        repo_root=root,
    )
    return layout.manifest(), layout.prompt_text


def _compact_artifact_ref(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_id": artifact["artifact_id"],
        "payload_path": artifact["payload_path"],
        "content_hash": artifact["content_hash"],
    }


def _persist_prompt_layout_artifacts(
    run_dir: Path,
    run_id: str,
    manifest: dict[str, object],
    prompt_text: str,
) -> None:
    artifacts = ArtifactWriter(run_dir)
    artifacts.run_id = run_id
    prompt_artifact = artifacts.persist_text(
        "prompt.rendered",
        "prompts/gen_0/gen0_peer0.md",
        prompt_text,
        schema_ref="praxist.prompt.rendered.v1",
        producer={"stage_id": "research_loop", "role_ref": "role:fake_peer"},
        content_type="text/markdown",
    )
    manifest_with_ref = {
        **manifest,
        "rendered_prompt_ref": _compact_artifact_ref(prompt_artifact),
    }
    artifacts.persist_json(
        "prompt.layout_manifest",
        "prompts/gen_0/gen0_peer0_layout.json",
        manifest_with_ref,
        schema_ref="praxist.prompt_layout.v1",
        producer={"stage_id": "research_loop", "role_ref": "role:fake_peer"},
        source_artifact_ids=[str(prompt_artifact["artifact_id"])],
    )


if __name__ == "__main__":
    unittest.main()
