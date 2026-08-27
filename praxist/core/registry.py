"""Run-local plugin discovery, resolution, and registry support."""

from __future__ import annotations

import glob
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

V1_STABLE_KINDS = (
    "role",
    "audit_rule",
    "evaluation",
    "panel_topology",
    "agent_runtime",
    "budget_policy",
    "workflow_stage",
)

EXPERIMENTAL_KINDS = (
    "tool_server",
    "ontology",
    "budget_unit",
    "model_provider",
    "graph_maintainer",
)

ALL_KINDS = V1_STABLE_KINDS + EXPERIMENTAL_KINDS
HARDCODED_EXECUTION_KINDS = set(ALL_KINDS)

KIND_DIRS = {
    "role": "roles",
    "audit_rule": "audit_rules",
    "evaluation": "evaluations",
    "panel_topology": "panel_topologies",
    "agent_runtime": "agent_runtimes",
    "budget_policy": "budget_policies",
    "workflow_stage": "workflow_stages",
    "tool_server": "tools",
    "ontology": "ontologies",
    "budget_unit": "budget_units",
    "model_provider": "model_providers",
    "graph_maintainer": "graph_maintainers",
}

SOURCE_PRIORITY = {"task_project": 0, "project": 1, "user": 2, "bundled": 3}
TRUSTED_EXECUTION_SOURCES = frozenset({"bundled", "task_project"})


@dataclass(frozen=True)
class PluginRef:
    """Parsed kind:name plugin reference with optional source, version, and requiredness constraints."""

    kind: str
    name: str
    version: str | None = None
    source: str | None = None
    required: bool = True

    @classmethod
    def parse(cls, value: str | dict[str, Any]) -> PluginRef:
        if isinstance(value, dict):
            return cls(
                kind=str(value["kind"]),
                name=str(value["name"]),
                version=value.get("version"),
                source=value.get("source"),
                required=bool(value.get("required", True)),
            )
        if ":" not in value:
            raise ValueError(f"PluginRef must be kind:name, got {value!r}")
        kind, name = value.split(":", 1)
        if not kind or not name:
            raise ValueError(f"PluginRef must be kind:name, got {value!r}")
        return cls(kind=kind, name=name)

    def as_string(self) -> str:
        return f"{self.kind}:{self.name}"


@dataclass(frozen=True)
class PluginMetadata:
    """Canonical plugin manifest data used for resolution, hashing, loading, and replay."""

    schema_version: int
    name: str
    kind: str
    version: str
    protocol_version: int
    stability: str
    description: str
    compatibility: dict[str, str]
    dependencies: list[dict[str, Any]]
    capabilities: list[str]
    entrypoint: str | None
    code: list[str]
    assets: list[str]

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> PluginMetadata:
        return cls(
            schema_version=int(manifest.get("schema_version", 1)),
            name=str(manifest["name"]),
            kind=str(manifest["kind"]),
            version=str(manifest.get("version", "0.1.0")),
            protocol_version=int(manifest.get("protocol_version", 1)),
            stability=str(manifest.get("stability", _stability_for_kind(str(manifest["kind"])))),
            description=str(manifest.get("description", "")),
            compatibility=dict(manifest.get("compatibility") or {}),
            dependencies=list(manifest.get("dependencies") or []),
            capabilities=list(manifest.get("capabilities") or []),
            entrypoint=manifest.get("entrypoint"),
            code=list(manifest.get("code") or []),
            assets=list(manifest.get("assets") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PluginIdentity:
    """Minimal identity tuple used during discovery before full metadata validation."""

    schema_version: int
    name: str
    kind: str
    version: str
    protocol_version: int
    stability: str

    def key(self) -> tuple[str, str]:
        return (self.kind, self.name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PluginCandidate:
    """Discovered plugin instance from one bundled, project, user, or task_project root."""

    identity: PluginIdentity
    source: str
    path: str
    manifest_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectedPlugin:
    """Resolved plugin descriptor with source, path, content hash, and selection provenance."""

    metadata: PluginMetadata
    source: str
    path: str
    content_hash: str
    selected_by: list[str]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["metadata"] = self.metadata.to_dict()
        return out


@dataclass(frozen=True)
class DiscoveryReport:
    """Plugin discovery output, including candidates and non-fatal manifest warnings."""

    candidates: list[PluginCandidate]
    warnings: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "praxist.plugin_discovery.v1",
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class PluginRoots:
    """Ordered plugin root set used by discovery for bundled, project, user, and task sources."""

    bundled: list[Path]
    project: list[Path]
    user: list[Path]
    task_project: list[Path] = field(default_factory=list)

    @classmethod
    def defaults(
        cls,
        workspace: Path | None = None,
        task_path: Path | None = None,
        *,
        extra_bundled: list[Path] | None = None,
        extra_project: list[Path] | None = None,
    ) -> PluginRoots:
        """Build the default plugin-root search set.

        ``extra_bundled`` / ``extra_project`` (issue #75 batch 6) let CLI
        entrypoints inject additional plugin search paths explicitly,
        bypassing the env-fallback. When either argument is ``None`` the
        helper falls back to ``PRAXIST_BUNDLED_PLUGIN_ROOTS`` /
        ``PRAXIST_PLUGIN_ROOTS`` so the test fixture path
        (``tests/__init__.py`` mutates ``PRAXIST_BUNDLED_PLUGIN_ROOTS`` to
        inject fixture plugins) and operator-level env extension both
        continue to work. Passing an empty list (``[]``) explicitly
        disables the env-fallback — useful for hermetic builds.
        """
        repo_root = Path(__file__).resolve().parents[2]
        project_root = workspace if workspace is not None else Path.cwd()
        if extra_bundled is None:
            extra_bundled = _plugin_roots_from_env("PRAXIST_BUNDLED_PLUGIN_ROOTS")
        if extra_project is None:
            extra_project = _plugin_roots_from_env("PRAXIST_PLUGIN_ROOTS")
        task_roots = (
            [Path(task_path).expanduser().resolve() / ".praxist" / "plugins"]
            if task_path is not None
            else []
        )
        return cls(
            bundled=[repo_root / "praxist" / "plugins", *extra_bundled],
            project=[project_root / ".praxist" / "plugins", *extra_project],
            user=[Path.home() / ".praxist" / "plugins"],
            task_project=task_roots,
        )


class PluginRegistry:
    """Frozen registry of selected plugin descriptors and optional instantiated objects."""

    def __init__(
        self, items: dict[tuple[str, str], Any], descriptors: dict[tuple[str, str], SelectedPlugin]
    ) -> None:
        self._items = dict(items)
        self._descriptors = dict(descriptors)

    def get(self, kind: str, name: str) -> Any | None:
        return self._items.get((kind, name))

    def require(self, kind: str, name: str) -> Any:
        key = (kind, name)
        if key not in self._items:
            raise KeyError(f"Plugin not registered: {kind}:{name}")
        return self._items[key]

    def descriptor(self, kind: str, name: str) -> SelectedPlugin:
        key = (kind, name)
        if key not in self._descriptors:
            raise KeyError(f"Plugin descriptor not registered: {kind}:{name}")
        return self._descriptors[key]

    def descriptor_for_ref(self, ref: str) -> SelectedPlugin:
        parsed = PluginRef.parse(ref)
        return self.descriptor(parsed.kind, parsed.name)

    def list(self, kind: str) -> list[SelectedPlugin]:
        return [
            descriptor
            for (item_kind, _), descriptor in self._descriptors.items()
            if item_kind == kind
        ]


class PluginRegistryBuilder:
    """Mutable registry builder used only during manifest loading."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], Any] = {}
        self._descriptors: dict[tuple[str, str], SelectedPlugin] = {}
        self._frozen = False

    def add(self, selected: SelectedPlugin, plugin: Any | None = None) -> None:
        if self._frozen:
            raise RuntimeError("PluginRegistryBuilder is frozen")
        key = (selected.metadata.kind, selected.metadata.name)
        if key in self._items:
            raise ValueError(f"Duplicate plugin: {selected.metadata.kind}:{selected.metadata.name}")
        self._items[key] = plugin if plugin is not None else selected.metadata.to_dict()
        self._descriptors[key] = selected

    def freeze(self) -> PluginRegistry:
        self._frozen = True
        return PluginRegistry(self._items, self._descriptors)


class PluginLoader:
    """Thin Gate B loader: discover, resolve, then freeze selected descriptors."""

    def __init__(self, roots: PluginRoots | None = None) -> None:
        self.roots = roots or PluginRoots.defaults()

    def discover(self) -> DiscoveryReport:
        candidates: list[PluginCandidate] = []
        warnings: list[dict[str, Any]] = []
        for source, roots in (
            ("task_project", self.roots.task_project),
            ("project", self.roots.project),
            ("user", self.roots.user),
            ("bundled", self.roots.bundled),
        ):
            for root in roots:
                if not root.exists():
                    continue
                for kind, dirname in KIND_DIRS.items():
                    kind_dir = root / dirname
                    if not kind_dir.exists():
                        continue
                    for manifest_path in sorted(kind_dir.glob("*/plugin.yaml")):
                        try:
                            manifest = _read_manifest(manifest_path)
                            identity = _identity_from_manifest(manifest)
                            if identity.kind != kind:
                                raise ValueError(
                                    f"manifest kind {identity.kind!r} does not match directory kind {kind!r}"
                                )
                            if identity.name != manifest_path.parent.name:
                                raise ValueError(
                                    f"manifest name {identity.name!r} does not match plugin directory "
                                    f"{manifest_path.parent.name!r}"
                                )
                            candidates.append(
                                PluginCandidate(
                                    identity=identity,
                                    source=source,
                                    path=str(manifest_path.parent),
                                    manifest_summary={
                                        "description": manifest.get("description", ""),
                                        "dependencies": manifest.get("dependencies") or [],
                                        "capabilities": manifest.get("capabilities") or [],
                                    },
                                )
                            )
                        except Exception as exc:  # noqa: BLE001 - discovery must be non-fatal.
                            warnings.append({"path": str(manifest_path), "error": str(exc)})
        return DiscoveryReport(candidates=candidates, warnings=warnings)

    def resolve(
        self,
        refs: Iterable[str | PluginRef],
        report: DiscoveryReport | None = None,
        *,
        run_id: str = "",
        root_task_ref: str = "",
        disabled_optional: list[dict[str, Any]] | None = None,
        enforce_bundled_execution: bool = False,
    ) -> dict[str, Any]:
        report = report or self.discover()
        index: dict[tuple[str, str], list[PluginCandidate]] = {}
        for candidate in report.candidates:
            index.setdefault(candidate.identity.key(), []).append(candidate)

        selected: dict[tuple[str, str], SelectedPlugin] = {}
        dependency_edges: list[dict[str, str]] = []
        resolving: list[tuple[str, str]] = []

        def select(ref: PluginRef, selected_by: str) -> None:
            if ref.kind not in ALL_KINDS:
                raise ValueError(f"Unknown plugin kind: {ref.kind}")
            key = (ref.kind, ref.name)
            if key in resolving:
                chain = " -> ".join(f"{kind}:{name}" for kind, name in [*resolving, key])
                raise ValueError(f"Plugin dependency cycle: {chain}")
            if key in selected:
                current = selected[key]
                if not _source_matches_selected(ref, current) or not _version_satisfies(
                    current.metadata.version, ref.version
                ):
                    if ref.required:
                        raise ValueError(
                            "Selected plugin does not satisfy required constraint: "
                            f"{ref.as_string()} {ref.version or ''} source={ref.source or 'any'}; "
                            f"selected {current.metadata.version} source={current.source}"
                        )
                    return
                if selected_by not in selected[key].selected_by:
                    selected[key] = SelectedPlugin(
                        metadata=current.metadata,
                        source=current.source,
                        path=current.path,
                        content_hash=current.content_hash,
                        selected_by=[*current.selected_by, selected_by],
                    )
                return
            candidates = [
                candidate for candidate in index.get(key, []) if _source_matches(ref, candidate)
            ]
            candidates = [
                candidate
                for candidate in candidates
                if _version_satisfies(candidate.identity.version, ref.version)
            ]
            if enforce_bundled_execution and ref.kind in HARDCODED_EXECUTION_KINDS:
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate.source in TRUSTED_EXECUTION_SOURCES
                ]
            if not candidates:
                if ref.required:
                    raise ValueError(f"Required plugin not found: {ref.as_string()}")
                return
            candidate = _choose_candidate(candidates)
            manifest = _read_manifest(Path(candidate.path) / "plugin.yaml")
            metadata = PluginMetadata.from_manifest(manifest)
            _validate_metadata(metadata, source=candidate.source)
            content_hash = compute_plugin_content_hash(Path(candidate.path), metadata)
            selected[key] = SelectedPlugin(
                metadata=metadata,
                source=candidate.source,
                path=candidate.path,
                content_hash=content_hash,
                selected_by=[selected_by],
            )
            resolving.append(key)
            for dep in metadata.dependencies:
                dep_ref = PluginRef.parse(dep)
                select(dep_ref, metadata.kind + ":" + metadata.name)
                if (dep_ref.kind, dep_ref.name) in selected:
                    dependency_edges.append(
                        {"from": metadata.kind + ":" + metadata.name, "to": dep_ref.as_string()}
                    )
            resolving.pop()

        for raw_ref in refs:
            ref = raw_ref if isinstance(raw_ref, PluginRef) else PluginRef.parse(raw_ref)
            select(ref, root_task_ref or "cli")

        selected_records = sorted(
            selected.values(), key=lambda item: (item.metadata.kind, item.metadata.name)
        )
        shadowed = _shadowed_records(report.candidates, selected_records)
        return {
            "schema_version": "praxist.plugin_resolution.v1",
            "run_id": run_id,
            "root_task_ref": root_task_ref,
            "algorithm_version": 2,
            "execution_source_policy": "bundled_only"
            if enforce_bundled_execution
            else "source_priority",
            "selected": [item.to_dict() for item in selected_records],
            "shadowed": shadowed,
            "disabled_optional": disabled_optional or [],
            "dependency_edges": dependency_edges,
            "content_hash_algorithm": "sha256",
            "warnings": report.warnings,
            "errors": [],
        }

    def load(self, manifest: dict[str, Any]) -> PluginRegistry:
        builder = PluginRegistryBuilder()
        for selected_record in manifest.get("selected", []):
            selected = selected_plugin_from_dict(selected_record)
            plugin = (
                instantiate_plugin_entrypoint(selected) if selected.metadata.entrypoint else None
            )
            builder.add(selected, plugin=plugin)
        return builder.freeze()


def selected_plugin_from_dict(value: dict[str, Any]) -> SelectedPlugin:
    """Rehydrate a SelectedPlugin from a resolution manifest record."""
    metadata = PluginMetadata.from_manifest(value["metadata"])
    return SelectedPlugin(
        metadata=metadata,
        source=str(value["source"]),
        path=str(value["path"]),
        content_hash=str(value["content_hash"]),
        selected_by=list(value.get("selected_by") or []),
    )


def require_execution_plugin(
    registry: PluginRegistry | None,
    ref: str,
    *,
    kind: str,
    capability: str | None = None,
) -> SelectedPlugin | None:
    """Validate that a selected plugin is executable under the current source policy."""
    if registry is None:
        return None
    selected = registry.descriptor_for_ref(ref)
    metadata = selected.metadata
    if metadata.kind != kind:
        raise ValueError(f"{ref} resolved as {metadata.kind}, expected {kind}")
    if selected.source not in TRUSTED_EXECUTION_SOURCES and not metadata.entrypoint:
        raise ValueError(
            f"{ref} selected from {selected.source}; non-bundled executable plugins must declare an entrypoint"
        )
    if capability and capability not in metadata.capabilities:
        raise ValueError(f"{ref} missing required execution capability: {capability}")
    return selected


def compute_plugin_content_hash(
    plugin_path: Path, metadata: PluginMetadata | dict[str, Any] | None = None
) -> str:
    """Hash a plugin manifest plus declared code and assets under its root."""
    if isinstance(metadata, dict):
        metadata = PluginMetadata.from_manifest(metadata)
    if metadata is None:
        metadata = PluginMetadata.from_manifest(_read_manifest(plugin_path / "plugin.yaml"))

    hasher = hashlib.sha256()
    paths = {plugin_path / "plugin.yaml"}
    for pattern in [*metadata.code, *metadata.assets]:
        paths.update(_safe_plugin_glob(plugin_path, pattern))
    plugin_root = plugin_path.resolve()
    for path in sorted(paths, key=lambda item: str(item.resolve().relative_to(plugin_root))):
        resolved = path.resolve()
        if not resolved.is_relative_to(plugin_root):
            raise ValueError(f"Plugin file escapes plugin root: {path}")
        rel = str(resolved.relative_to(plugin_root)).replace("\\", "/")
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(resolved.read_bytes())
        hasher.update(b"\0")
    return "sha256:" + hasher.hexdigest()


def read_plugin_metadata(plugin_path: Path) -> PluginMetadata:
    """Read canonical plugin metadata from the on-disk manifest."""
    return PluginMetadata.from_manifest(_read_manifest(plugin_path / "plugin.yaml"))


def _safe_plugin_glob(plugin_path: Path, pattern: str) -> set[Path]:
    pattern_path = Path(pattern)
    if pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise ValueError(f"Plugin code/assets path must stay inside plugin root: {pattern}")
    plugin_root = plugin_path.resolve()
    matches = set()
    raw_matches = glob.glob(str(plugin_path / pattern), recursive=True)
    for match in raw_matches:
        path = Path(match)
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(plugin_root):
            raise ValueError(f"Plugin code/assets path escapes plugin root: {pattern}")
        matches.add(path)
    if not matches and not glob.has_magic(pattern):
        raise ValueError(f"Declared plugin code/assets file not found: {pattern}")
    return matches


def static_selected_plugin(ref: PluginRef, selected_by: list[str]) -> SelectedPlugin:
    """Create a deterministic selected-plugin descriptor for legacy static manifests."""
    stability = _stability_for_kind(ref.kind)
    metadata = PluginMetadata(
        schema_version=1,
        name=ref.name,
        kind=ref.kind,
        version="0.1.0",
        protocol_version=1,
        stability=stability,
        description=f"Static plugin for {ref.as_string()}",
        compatibility={"praxist_core": ">=0.1.0", "python": ">=3.11"},
        dependencies=[],
        capabilities=[],
        entrypoint=None,
        code=[],
        assets=[],
    )
    digest = hashlib.sha256(f"{ref.kind}:{ref.name}:0.1.0".encode()).hexdigest()
    return SelectedPlugin(
        metadata=metadata,
        source="bundled",
        path=f"builtin://static/{ref.kind}/{ref.name}",
        content_hash=f"sha256:{digest}",
        selected_by=selected_by,
    )


def static_resolution_manifest(run_id: str, root_task_ref: str, refs: list[str]) -> dict[str, Any]:
    """Build a static plugin resolution manifest for compatibility-only paths."""
    selected = [
        static_selected_plugin(PluginRef.parse(ref), [root_task_ref]).to_dict() for ref in refs
    ]
    return {
        "schema_version": "praxist.plugin_resolution.v1",
        "run_id": run_id,
        "root_task_ref": root_task_ref,
        "algorithm_version": 1,
        "selected": selected,
        "shadowed": [],
        "disabled_optional": [],
        "dependency_edges": [],
        "content_hash_algorithm": "sha256",
        "warnings": [],
        "errors": [],
    }


def assert_bundled_execution_manifest(manifest: dict[str, Any]) -> None:
    """Verify executable plugin selections come from a trusted source (bundled or task_project)
    until dispatch is registry-backed."""
    bundled_only = manifest.get("execution_source_policy") == "bundled_only"
    for selected in manifest.get("selected", []):
        metadata = selected.get("metadata") or {}
        kind = metadata.get("kind")
        name = metadata.get("name")
        source = selected.get("source")
        entrypoint = metadata.get("entrypoint")
        if (
            kind in HARDCODED_EXECUTION_KINDS
            and source not in TRUSTED_EXECUTION_SOURCES
            and (bundled_only or not entrypoint)
        ):
            raise ValueError(
                f"{kind}:{name} selected from {source}; current execution path only supports bundled or task_project plugins"
            )


def load_plugin_entrypoint(selected: SelectedPlugin) -> Any:
    """Load a selected plugin's manifest-declared entrypoint factory."""
    metadata = selected.metadata
    if not metadata.entrypoint:
        raise ValueError(f"{metadata.kind}:{metadata.name} does not declare an entrypoint")
    module_name, sep, factory_name = metadata.entrypoint.partition(":")
    if not sep or not module_name or not factory_name:
        raise ValueError(f"{metadata.kind}:{metadata.name} entrypoint must be module:function")

    plugin_root = Path(selected.path).resolve()
    module_rel = Path(module_name.replace(".", "/") + ".py")
    module_path = (plugin_root / module_rel).resolve()
    if not module_path.is_relative_to(plugin_root):
        raise ValueError(f"{metadata.kind}:{metadata.name} entrypoint escapes plugin root")
    if not module_path.is_file():
        raise ValueError(
            f"{metadata.kind}:{metadata.name} entrypoint module not found: {module_rel}"
        )

    bundled_module = _bundled_plugin_module_name(plugin_root, module_name)
    if bundled_module:
        module = importlib.import_module(bundled_module)
        factory = getattr(module, factory_name, None)
        if not callable(factory):
            raise ValueError(
                f"{metadata.kind}:{metadata.name} entrypoint factory is not callable: {metadata.entrypoint}"
            )
        return factory

    digest = hashlib.sha256(
        f"{selected.path}:{selected.content_hash}:{metadata.entrypoint}".encode()
    ).hexdigest()[:16]
    spec_name = f"_praxist_plugin_{metadata.kind}_{metadata.name}_{digest}"
    spec = importlib.util.spec_from_file_location(spec_name, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{metadata.kind}:{metadata.name} entrypoint module cannot be imported")
    module = importlib.util.module_from_spec(spec)
    previous_path = list(sys.path)
    sys.path.insert(0, str(plugin_root))
    try:
        sys.modules[spec_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ValueError(
            f"{metadata.kind}:{metadata.name} entrypoint factory is not callable: {metadata.entrypoint}"
        )
    return factory


def _bundled_plugin_module_name(plugin_root: Path, module_name: str) -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    bundled_root = (repo_root / "praxist" / "plugins").resolve()
    try:
        rel = plugin_root.relative_to(bundled_root)
    except ValueError:
        return None
    rel_parts = ".".join(rel.parts)
    return f"praxist.plugins.{rel_parts}.{module_name}" if rel_parts else None


def instantiate_plugin_entrypoint(selected: SelectedPlugin) -> Any:
    """Instantiate a selected executable plugin using its entrypoint factory."""
    factory = load_plugin_entrypoint(selected)
    signature = inspect.signature(factory)
    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    if not required:
        return factory()
    if len(required) == 1:
        parameter = required[0]
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            return factory(**{parameter.name: selected})
        return factory(selected)
    raise ValueError(
        f"{selected.metadata.kind}:{selected.metadata.name} entrypoint factory must accept zero arguments "
        "or one SelectedPlugin argument"
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"plugin manifest must be an object: {path}")
    return value


def _plugin_roots_from_env(name: str) -> list[Path]:
    raw = os.environ.get(name, "")
    if not raw:
        return []
    return [Path(item).expanduser().resolve() for item in raw.split(os.pathsep) if item]


def _identity_from_manifest(manifest: dict[str, Any]) -> PluginIdentity:
    kind = str(manifest["kind"])
    return PluginIdentity(
        schema_version=int(manifest.get("schema_version", 1)),
        name=str(manifest["name"]),
        kind=kind,
        version=str(manifest.get("version", "0.1.0")),
        protocol_version=int(manifest.get("protocol_version", 1)),
        stability=str(manifest.get("stability", _stability_for_kind(kind))),
    )


def _validate_metadata(metadata: PluginMetadata, *, source: str | None = None) -> None:
    if metadata.kind not in ALL_KINDS:
        raise ValueError(f"Unknown plugin kind: {metadata.kind}")
    # Issue #80: stability is an interface contract for bundled plugins —
    # promising schema / prompt / role-API backward compatibility across
    # versions. For task-local plugins under ``<task>/.praxist/plugins/``,
    # the trust axis is source (already gated by ``TRUSTED_EXECUTION_SOURCES``),
    # not stability tier; task-local plugins are high-churn by design and
    # scope-isolated to a single task project. Stacking a stability check
    # on top of source trust would force task authors to either ship
    # ``v1_stable`` prematurely or carry a manual override. Skip the
    # mismatch check when the candidate came from a task project.
    if source != "task_project":
        expected = _stability_for_kind(metadata.kind)
        if metadata.stability != expected:
            raise ValueError(
                f"{metadata.kind}:{metadata.name} stability must be {expected}, got {metadata.stability}"
            )
    if metadata.protocol_version != 1:
        raise ValueError(f"{metadata.kind}:{metadata.name} protocol_version must be 1")
    if metadata.entrypoint:
        if ":" not in metadata.entrypoint:
            raise ValueError(f"{metadata.kind}:{metadata.name} entrypoint must be module:function")
        module_name = metadata.entrypoint.split(":", 1)[0].replace(".", "/") + ".py"
        if module_name not in metadata.code:
            raise ValueError(
                f"{metadata.kind}:{metadata.name} entrypoint module must be listed in code"
            )


def _stability_for_kind(kind: str) -> str:
    return "v1_stable" if kind in V1_STABLE_KINDS else "experimental"


def _source_matches(ref: PluginRef, candidate: PluginCandidate) -> bool:
    return ref.source in (None, "any", candidate.source)


def _source_matches_selected(ref: PluginRef, selected: SelectedPlugin) -> bool:
    return ref.source in (None, "any", selected.source)


def _choose_candidate(candidates: list[PluginCandidate]) -> PluginCandidate:
    return sorted(
        candidates,
        key=lambda candidate: (
            SOURCE_PRIORITY[candidate.source],
            tuple(-part for part in _version_tuple(candidate.identity.version)),
            candidate.path,
        ),
    )[0]


def _shadowed_records(
    candidates: list[PluginCandidate], selected: list[SelectedPlugin]
) -> list[dict[str, Any]]:
    selected_keys = {(item.metadata.kind, item.metadata.name): item for item in selected}
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        key = candidate.identity.key()
        selected_item = selected_keys.get(key)
        if selected_item is None:
            continue
        if candidate.path == selected_item.path:
            continue
        records.append(
            {
                "kind": candidate.identity.kind,
                "name": candidate.identity.name,
                "selected_source": selected_item.source,
                "selected_version": selected_item.metadata.version,
                "selected_path": selected_item.path,
                "shadowed_source": candidate.source,
                "shadowed_version": candidate.identity.version,
                "shadowed_path": candidate.path,
            }
        )
    return records


def _version_satisfies(version: str, constraint: str | None) -> bool:
    if not constraint:
        return True
    version_tuple = _version_tuple(version)
    for part in constraint.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith(">="):
            if version_tuple < _version_tuple(part[2:]):
                return False
        elif part.startswith(">"):
            if version_tuple <= _version_tuple(part[1:]):
                return False
        elif part.startswith("<="):
            if version_tuple > _version_tuple(part[2:]):
                return False
        elif part.startswith("<"):
            if version_tuple >= _version_tuple(part[1:]):
                return False
        elif part.startswith("=="):
            if version_tuple != _version_tuple(part[2:]):
                return False
        else:
            raise ValueError(f"Unsupported version constraint: {part}")
    return True


def _version_tuple(version: str) -> tuple[int, ...]:
    clean = version.split("-", 1)[0]
    parts: list[int] = []
    for part in clean.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def dumps_manifest(value: dict[str, Any]) -> str:
    """Serialize a manifest with stable ordering for human inspection and tests."""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
