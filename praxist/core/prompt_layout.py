"""PromptLayout V1 helpers.

Step 20 keeps existing legacy prompt text runnable while making the
cache-relevant prompt boundaries explicit. Jinja remains a renderer for
individual blocks; it does not own the frozen/dynamic partition.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from praxist.core.storage import write_json

PromptPartition = Literal["frozen_prefix", "semi_static_run_context", "dynamic_payload"]

LAYOUT_VERSION = "praxist.prompt_layout.v1"

DYNAMIC_MARKERS = (
    "run_dir",
    "workspace_dir",
    "results_dir",
    "variants_dir",
    "findings_dir",
    "logs_dir",
    "notebook_path",
    "gen_id",
    "generation_id",
    "peer_id",
    "frontier_summary",
    "variant_hint",
    "graph_session_context",
    "research_agenda",
    "pi_memo",
    "pi_memos",
    "cross_reviews",
    "budget",
    "budget_grant",
    "timestamp",
    "orchestrator_status",
)

_DYNAMIC_RX = re.compile(r"\b(" + "|".join(re.escape(item) for item in DYNAMIC_MARKERS) + r")\b")


DEFAULT_FROZEN_PREFIX = """# Praxist Research Agent Stable Contract

You are an autonomous scientific research agent operating inside Praxist.
This stable prefix defines invariant behavior across peers, generations,
panel rounds, and runtime sessions. Dynamic run state is provided later.

Core invariants:
- Do rigorous science: implement ideas faithfully, report negative results,
  and do not cherry-pick seeds or fabricate measurements.
- Preserve useful work: keep experiments that may produce meaningful results
  alive when possible, and make intermediate results, findings, logs, and
  reproducible facts visible to the run.
- Treat graph, frontier, memory, and panel context as advisory evidence, not
  as scientific truth. Raw findings and benchmark outputs remain the source
  of truth.
- Use available tools by their declared names and publish structured findings
  with enough provenance for later peers and PI/Chair stages to reuse them.
- Treat runtime task notifications as the completion source for runtime-managed
  background commands. Never infer completion from whether a runtime-private
  `tasks/<id>.output` file is non-empty; successful commands may emit no text.
- Keep credentials, raw secrets, machine-specific environment dumps, and
  provider-private response details out of prompts, logs, artifacts, cache
  keys, and trajectories.
"""

DEFAULT_START_COMMAND = """# Praxist Session Start Command

Begin now. Treat the full rendered prompt above as the active user task, not
as background context, reminders, or documentation. Do not ask for
confirmation, do not wait for a human instruction, and do not summarize the
prompt. Start by using the available tools to read your peer notebook, inspect
shared findings, query the frontier, and then proceed with the research
workflow.
"""


@dataclass(frozen=True)
class PromptBlock:
    """Typed prompt segment with cache stability metadata and a deterministic content hash."""

    block_id: str
    partition: PromptPartition
    renderer: str
    text: str
    source_path: str | None = None
    source_hash: str | None = None
    legacy_renderer: bool = False
    dynamic_markers_in_template: list[str] = field(default_factory=list)
    dynamic_markers_in_rendered: list[str] = field(default_factory=list)

    @property
    def rendered_hash(self) -> str:
        return sha256_text(self.text)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "partition": self.partition,
            "renderer": self.renderer,
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "rendered_hash": self.rendered_hash,
            "legacy_renderer": self.legacy_renderer,
            "dynamic_markers_in_template": self.dynamic_markers_in_template,
            "dynamic_markers_in_rendered": self.dynamic_markers_in_rendered,
            "size_bytes": len(self.text.encode("utf-8")),
        }


@dataclass(frozen=True)
class PromptLayout:
    """Rendered prompt plus block-level hashes used for cache readiness and replay verification."""

    run_id: str
    stage_id: str
    prompt_id: str
    agent_runtime_ref: str
    model_provider_ref: str
    cache_mode: str
    runtime_cache_strategy: str | None
    provider_cache_strategy: str | None
    blocks: list[PromptBlock]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks if block.text).rstrip() + "\n"

    @property
    def frozen_prefix_hash(self) -> str:
        return _partition_hash(self.blocks, "frozen_prefix")

    @property
    def semi_static_hash(self) -> str:
        return _partition_hash(self.blocks, "semi_static_run_context")

    @property
    def dynamic_payload_hash(self) -> str:
        return _partition_hash(self.blocks, "dynamic_payload")

    @property
    def layout_hash(self) -> str:
        payload = {
            "layout_version": LAYOUT_VERSION,
            "frozen_prefix_hash": self.frozen_prefix_hash,
            "semi_static_hash": self.semi_static_hash,
            "dynamic_payload_hash": self.dynamic_payload_hash,
            "block_hashes": [block.rendered_hash for block in self.blocks],
        }
        return sha256_json(payload)

    def manifest(self, *, rendered_prompt_ref: dict[str, Any] | None = None) -> dict[str, Any]:
        frozen_audit = audit_frozen_blocks(self.blocks)
        return {
            "schema_version": LAYOUT_VERSION,
            "layout_version": LAYOUT_VERSION,
            "run_id": self.run_id,
            "stage_id": self.stage_id,
            "prompt_id": self.prompt_id,
            "agent_runtime_ref": self.agent_runtime_ref,
            "model_provider_ref": self.model_provider_ref,
            "cache_mode": self.cache_mode,
            "runtime_cache_strategy": self.runtime_cache_strategy,
            "provider_cache_strategy": self.provider_cache_strategy,
            "frozen_prefix_hash": self.frozen_prefix_hash,
            "semi_static_hash": self.semi_static_hash,
            "dynamic_payload_hash": self.dynamic_payload_hash,
            "layout_hash": self.layout_hash,
            "rendered_prompt_hash": sha256_text(self.prompt_text),
            "cache_breakpoints": ["frozen_prefix"],
            "cache_usage_status": "cache_usage_unknown",
            "rendered_prompt_ref": rendered_prompt_ref,
            "blocks": [block.to_manifest_dict() for block in self.blocks],
            "frozen_audit": frozen_audit,
            "legacy_jinja_renderers": [
                block.block_id for block in self.blocks if block.legacy_renderer
            ],
            "metadata": self.metadata,
        }


def build_legacy_jinja_prompt_layout(
    *,
    base_template_path: Path,
    task_prompt_path: Path | None,
    generation_template_path: Path | None,
    context: dict[str, Any],
    run_id: str,
    stage_id: str,
    prompt_id: str,
    agent_runtime_ref: str = "agent_runtime:claude_sdk",
    model_provider_ref: str = "",
    repo_root: Path | None = None,
    frozen_prefix_text: str = DEFAULT_FROZEN_PREFIX,
    start_command_text: str = DEFAULT_START_COMMAND,
    extra_dynamic_blocks: list[dict[str, Any]] | None = None,
) -> PromptLayout:
    """Render the legacy research-loop prompt as PromptLayout V1 blocks.

    Existing Jinja templates remain dynamic legacy blocks. A stable prefix is
    prepended so runtime auto-cache implementations have an invariant prefix
    to reuse while Step 20 separates the remaining legacy monolith.
    """

    blocks = [
        PromptBlock(
            block_id="identity_contract",
            partition="frozen_prefix",
            renderer="static_text",
            text=frozen_prefix_text.strip() + "\n",
            source_path=None,
            source_hash=None,
            legacy_renderer=False,
            dynamic_markers_in_template=[],
            dynamic_markers_in_rendered=find_dynamic_markers(frozen_prefix_text),
        )
    ]
    blocks.extend(
        _prompt_source_summary_block(
            base_template_path=base_template_path,
            task_prompt_path=task_prompt_path,
            generation_template_path=generation_template_path,
            repo_root=repo_root,
        )
    )
    blocks.extend(
        _render_optional_legacy_block(
            block_id="legacy_base_prompt",
            path=base_template_path,
            context=context,
            repo_root=repo_root,
        )
    )
    if task_prompt_path is not None:
        blocks.extend(
            _render_optional_legacy_block(
                block_id="legacy_task_prompt",
                path=task_prompt_path,
                context=context,
                repo_root=repo_root,
            )
        )
    if generation_template_path is not None:
        blocks.extend(
            _render_optional_legacy_block(
                block_id="legacy_generation_prompt",
                path=generation_template_path,
                context=context,
                repo_root=repo_root,
            )
        )
    for idx, extra_block in enumerate(extra_dynamic_blocks or []):
        text = str(extra_block.get("text") or "")
        if not text.strip():
            continue
        block_id = str(extra_block.get("block_id") or f"extra_dynamic_block_{idx}")
        blocks.append(
            PromptBlock(
                block_id=block_id,
                partition="dynamic_payload",
                renderer=str(extra_block.get("renderer") or "static_text"),
                text=text.strip() + "\n",
                source_path=extra_block.get("source_path"),
                source_hash=None,
                legacy_renderer=False,
                dynamic_markers_in_template=[],
                dynamic_markers_in_rendered=find_dynamic_markers(text),
            )
        )
    if start_command_text.strip():
        blocks.append(
            PromptBlock(
                block_id="session_start_command",
                partition="dynamic_payload",
                renderer="static_text",
                text=start_command_text.strip() + "\n",
                source_path=None,
                source_hash=None,
                legacy_renderer=False,
                dynamic_markers_in_template=[],
                dynamic_markers_in_rendered=find_dynamic_markers(start_command_text),
            )
        )

    cache_mode, runtime_cache_strategy, provider_cache_strategy = _cache_strategy(
        agent_runtime_ref,
        model_provider_ref,
    )
    return PromptLayout(
        run_id=run_id,
        stage_id=stage_id,
        prompt_id=prompt_id,
        agent_runtime_ref=agent_runtime_ref,
        model_provider_ref=model_provider_ref,
        cache_mode=cache_mode,
        runtime_cache_strategy=runtime_cache_strategy,
        provider_cache_strategy=provider_cache_strategy,
        blocks=blocks,
        metadata={
            "legacy_jinja_block_renderer": True,
            "prompt_layout_step": 20,
            "dynamic_payload_owned_by": "legacy_jinja_blocks",
        },
    )


def write_prompt_layout_files(
    *,
    layout: PromptLayout,
    prompt_path: Path,
    manifest_path: Path,
    rendered_prompt_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist prompt text and layout metadata artifacts for one agent session."""
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(layout.prompt_text, encoding="utf-8")
    manifest = layout.manifest(rendered_prompt_ref=rendered_prompt_ref)
    write_json(manifest_path, manifest)
    return manifest


def audit_frozen_blocks(blocks: list[PromptBlock]) -> dict[str, Any]:
    """Validate that frozen prompt blocks do not contain dynamic run, peer, or generation markers."""
    violations: list[dict[str, Any]] = []
    for block in blocks:
        if block.partition != "frozen_prefix":
            continue
        markers = sorted(set(block.dynamic_markers_in_template + block.dynamic_markers_in_rendered))
        if markers:
            violations.append({"block_id": block.block_id, "markers": markers})
    return {
        "status": "pass" if not violations else "fail",
        "violations": violations,
    }


def find_dynamic_markers(text: str) -> list[str]:
    """Return dynamic marker names detected in prompt text."""
    return sorted(set(match.group(1) for match in _DYNAMIC_RX.finditer(text or "")))


def sha256_text(text: str) -> str:
    """Return a SHA-256 digest for UTF-8 prompt text."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-serializable object."""
    normalized = json.dumps(
        _normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(normalized)


def _render_optional_legacy_block(
    *,
    block_id: str,
    path: Path,
    context: dict[str, Any],
    repo_root: Path | None,
) -> list[PromptBlock]:
    if not path.exists():
        return []
    from jinja2 import Environment, FileSystemLoader

    template_text = path.read_text(encoding="utf-8")
    # Use a FileSystemLoader rooted at the template's directory so legacy
    # Jinja templates can ``{% include %}`` sibling partials (issue #85's
    # ``prompt_generation_role_description.jinja2`` is the first use). The
    # default ``Template(...)`` constructor used here previously had no
    # loader, so any include raised ``TemplateNotFound`` at render time.
    env = Environment(
        loader=FileSystemLoader(str(path.parent)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    rendered = env.from_string(template_text).render(**context)
    return [
        PromptBlock(
            block_id=block_id,
            partition="dynamic_payload",
            renderer="jinja2",
            text=rendered,
            source_path=_display_path(path, repo_root),
            source_hash=sha256_text(template_text),
            legacy_renderer=True,
            dynamic_markers_in_template=find_dynamic_markers(template_text),
            dynamic_markers_in_rendered=find_dynamic_markers(rendered),
        )
    ]


def _prompt_source_summary_block(
    *,
    base_template_path: Path,
    task_prompt_path: Path | None,
    generation_template_path: Path | None,
    repo_root: Path | None,
) -> list[PromptBlock]:
    sources = [
        ("base", base_template_path),
        ("task", task_prompt_path),
        ("generation", generation_template_path),
    ]
    lines = [
        "# Prompt Source Contract",
        "",
        "The following source set is stable for this rendered prompt layout.",
        "Legacy Jinja renderers below are treated as dynamic payload until the native PromptBuilder migration.",
    ]
    any_existing = False
    for label, path in sources:
        if path is None or not path.exists():
            continue
        any_existing = True
        template_text = path.read_text(encoding="utf-8")
        lines.append(f"- {label}: {_display_path(path, repo_root)} ({sha256_text(template_text)})")
    if not any_existing:
        return []
    text = "\n".join(lines).rstrip() + "\n"
    return [
        PromptBlock(
            block_id="prompt_source_contract",
            partition="semi_static_run_context",
            renderer="static_source_summary",
            text=text,
            legacy_renderer=False,
            dynamic_markers_in_template=[],
            dynamic_markers_in_rendered=find_dynamic_markers(text),
        )
    ]


def _cache_strategy(
    agent_runtime_ref: str,
    model_provider_ref: str,
) -> tuple[str, str | None, str | None]:
    if (
        agent_runtime_ref == "agent_runtime:fake_runtime"
        or model_provider_ref == "model_provider:fake_provider"
    ):
        return "disabled", None, None
    if agent_runtime_ref == "agent_runtime:claude_sdk":
        return "runtime_auto_cache", "runtime_auto_cache", None
    if model_provider_ref == "model_provider:anthropic_messages":
        return "provider_explicit_cache", None, "anthropic_messages_cache_control"
    return "provider_default", None, None


def _partition_hash(blocks: list[PromptBlock], partition: PromptPartition) -> str:
    payload = [
        {
            "block_id": block.block_id,
            "rendered_hash": block.rendered_hash,
        }
        for block in blocks
        if block.partition == partition
    ]
    return sha256_json(payload)


def _display_path(path: Path, repo_root: Path | None) -> str:
    resolved = path.resolve()
    if repo_root is not None:
        try:
            return str(resolved.relative_to(repo_root.resolve()))
        except ValueError:
            pass
    return str(resolved)


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    return value
