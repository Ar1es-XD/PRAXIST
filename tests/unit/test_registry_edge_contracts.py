from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class RegistryEdgeContractsTest(unittest.TestCase):
    def test_registry_discovery_resolution_and_validation_edges(self) -> None:
        from praxist.core import registry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundled = root / "bundled"
            project = root / "project"
            user = root / "user"
            runtime = self._write_plugin(
                bundled,
                "agent_runtimes",
                "runtime_a",
                "agent_runtime",
                version="1.2.0",
                dependencies=[{"kind": "model_provider", "name": "provider_a"}],
                entrypoint="adapter:create_runtime",
                code={"adapter.py": "def create_runtime():\n    return {'runtime': True}\n"},
            )
            self._write_plugin(
                bundled,
                "model_providers",
                "provider_a",
                "model_provider",
                stability="experimental",
            )
            self._write_plugin(
                project,
                "agent_runtimes",
                "runtime_a",
                "agent_runtime",
                version="1.0.0",
            )
            self._write_plugin(
                user, "agent_runtimes", "runtime_a", "agent_runtime", version="0.9.0"
            )
            wrong_kind_dir = bundled / "agent_runtimes" / "bad_kind"
            wrong_kind_dir.mkdir(parents=True)
            wrong_kind_dir.joinpath("plugin.yaml").write_text(
                "name: bad_kind\nkind: model_provider\nprotocol_version: 1\n",
                encoding="utf-8",
            )
            wrong_name_dir = bundled / "agent_runtimes" / "bad_name"
            wrong_name_dir.mkdir(parents=True)
            wrong_name_dir.joinpath("plugin.yaml").write_text(
                "name: other\nkind: agent_runtime\nprotocol_version: 1\n",
                encoding="utf-8",
            )

            roots = registry.PluginRoots(bundled=[bundled], project=[project], user=[user])
            loader = registry.PluginLoader(roots)
            report = loader.discover()
            self.assertGreaterEqual(len(report.candidates), 4)
            self.assertGreaterEqual(len(report.warnings), 2)
            self.assertIn("schema_version", report.to_dict())
            self.assertIsInstance(report.candidates[0].to_dict(), dict)
            self.assertIsInstance(report.candidates[0].identity.to_dict(), dict)

            with self.assertRaisesRegex(ValueError, "Unknown plugin kind"):
                loader.resolve(["unknown:x"], report)
            manifest = loader.resolve(
                [registry.PluginRef("agent_runtime", "runtime_a", version=">=1.1,<2")],
                report,
                run_id="run",
                root_task_ref="task:test",
                enforce_bundled_execution=True,
            )
            selected_refs = {
                item["metadata"]["kind"] + ":" + item["metadata"]["name"]
                for item in manifest["selected"]
            }
            self.assertEqual(
                selected_refs, {"agent_runtime:runtime_a", "model_provider:provider_a"}
            )
            self.assertEqual(manifest["execution_source_policy"], "bundled_only")
            self.assertTrue(manifest["dependency_edges"])
            self.assertTrue(manifest["shadowed"])

            selected_runtime = next(
                item for item in manifest["selected"] if item["metadata"]["kind"] == "agent_runtime"
            )
            selected = registry.selected_plugin_from_dict(selected_runtime)
            self.assertEqual(selected.metadata.to_dict()["name"], "runtime_a")
            self.assertEqual(registry.compute_plugin_content_hash(runtime), selected.content_hash)
            self.assertEqual(
                registry.compute_plugin_content_hash(runtime, selected_runtime["metadata"]),
                selected.content_hash,
            )
            self.assertTrue(
                registry.dumps_manifest({"b": 1, "a": 2}).splitlines()[1].strip().startswith('"a"')
            )

            loaded = loader.load(manifest)
            self.assertEqual(loaded.get("missing", "x"), None)
            self.assertEqual(loaded.require("agent_runtime", "runtime_a"), {"runtime": True})
            self.assertEqual(
                loaded.descriptor_for_ref("agent_runtime:runtime_a").metadata.name, "runtime_a"
            )
            with self.assertRaises(KeyError):
                loaded.require("agent_runtime", "missing")
            with self.assertRaises(KeyError):
                loaded.descriptor("agent_runtime", "missing")

            builder = registry.PluginRegistryBuilder()
            builder.add(selected)
            with self.assertRaisesRegex(ValueError, "Duplicate plugin"):
                builder.add(selected)
            built = builder.freeze()
            self.assertEqual(len(built.list("agent_runtime")), 1)
            with self.assertRaises(RuntimeError):
                builder.add(selected)

            self.assertIs(
                registry.require_execution_plugin(
                    loaded, "agent_runtime:runtime_a", kind="agent_runtime"
                ),
                loaded.descriptor("agent_runtime", "runtime_a"),
            )
            with self.assertRaisesRegex(ValueError, "resolved as"):
                registry.require_execution_plugin(
                    loaded,
                    "agent_runtime:runtime_a",
                    kind="model_provider",
                )
            with self.assertRaisesRegex(ValueError, "missing required execution capability"):
                registry.require_execution_plugin(
                    loaded,
                    "agent_runtime:runtime_a",
                    kind="agent_runtime",
                    capability="missing",
                )
            self.assertIsNone(
                registry.require_execution_plugin(
                    None,
                    "agent_runtime:runtime_a",
                    kind="agent_runtime",
                )
            )

            with self.assertRaisesRegex(ValueError, "not found"):
                registry._safe_plugin_glob(runtime, "missing.py")
            with self.assertRaisesRegex(ValueError, "inside plugin root"):
                registry._safe_plugin_glob(runtime, "../escape.py")
            runtime.joinpath("subdir").mkdir()
            self.assertEqual(registry._safe_plugin_glob(runtime, "subdir/*"), set())

            manifest_only = self._write_plugin(
                project, "agent_runtimes", "manifest_only", "agent_runtime"
            )
            manifest_only_selected = registry.SelectedPlugin(
                metadata=registry.read_plugin_metadata(manifest_only),
                source="project",
                path=str(manifest_only),
                content_hash=registry.compute_plugin_content_hash(manifest_only),
                selected_by=["test"],
            )
            manifest_registry = registry.PluginRegistryBuilder()
            manifest_registry.add(manifest_only_selected)
            with self.assertRaisesRegex(ValueError, "must declare an entrypoint"):
                registry.require_execution_plugin(
                    manifest_registry.freeze(),
                    "agent_runtime:manifest_only",
                    kind="agent_runtime",
                )

    def test_entrypoint_loading_factories_and_version_constraints(self) -> None:
        from praxist.core import registry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaisesRegex(ValueError, "kind:name"):
                registry.PluginRef.parse("bad")
            with self.assertRaisesRegex(ValueError, "kind:name"):
                registry.PluginRef.parse(":bad")
            self.assertEqual(
                registry.PluginRef.parse(
                    {"kind": "agent_runtime", "name": "x", "required": False}
                ).required,
                False,
            )

            self.assertTrue(registry._version_satisfies("1.2.3", ">=1.0, <2.0, ==1.2.3"))
            self.assertFalse(registry._version_satisfies("1.2.3", ">1.2.3"))
            self.assertFalse(registry._version_satisfies("1.2.3", "<=1.2.2"))
            self.assertFalse(registry._version_satisfies("1.2.3", "<1.2.0"))
            self.assertEqual(registry._version_tuple("1.x-rc1"), (1, 0, 0))
            with self.assertRaisesRegex(ValueError, "Unsupported version constraint"):
                registry._version_satisfies("1.0.0", "~=1.0")

            zero_arg = self._write_plugin(
                root,
                "agent_runtimes",
                "zero",
                "agent_runtime",
                entrypoint="adapter:create_runtime",
                code={"adapter.py": "def create_runtime():\n    return 'zero'\n"},
            )
            one_arg = self._write_plugin(
                root,
                "agent_runtimes",
                "one",
                "agent_runtime",
                entrypoint="adapter:create_runtime",
                code={
                    "adapter.py": "def create_runtime(selected):\n    return selected.metadata.name\n"
                },
            )
            kw_arg = self._write_plugin(
                root,
                "agent_runtimes",
                "kw",
                "agent_runtime",
                entrypoint="adapter:create_runtime",
                code={
                    "adapter.py": "def create_runtime(*, selected):\n    return selected.metadata.kind\n"
                },
            )
            two_arg = self._write_plugin(
                root,
                "agent_runtimes",
                "two",
                "agent_runtime",
                entrypoint="adapter:create_runtime",
                code={"adapter.py": "def create_runtime(a, b):\n    return None\n"},
            )
            bad_factory = self._write_plugin(
                root,
                "agent_runtimes",
                "bad_factory",
                "agent_runtime",
                entrypoint="adapter:create_runtime",
                code={"adapter.py": "create_runtime = 1\n"},
            )
            missing_factory = self._write_plugin(
                root,
                "agent_runtimes",
                "missing_factory",
                "agent_runtime",
                entrypoint="adapter:missing",
                code={"adapter.py": "def other():\n    return None\n"},
            )

            def selected_for(path: Path) -> registry.SelectedPlugin:
                metadata = registry.read_plugin_metadata(path)
                return registry.SelectedPlugin(
                    metadata=metadata,
                    source="project",
                    path=str(path),
                    content_hash=registry.compute_plugin_content_hash(path, metadata),
                    selected_by=["test"],
                )

            self.assertEqual(registry.instantiate_plugin_entrypoint(selected_for(zero_arg)), "zero")
            self.assertEqual(registry.instantiate_plugin_entrypoint(selected_for(one_arg)), "one")
            self.assertEqual(
                registry.instantiate_plugin_entrypoint(selected_for(kw_arg)), "agent_runtime"
            )
            with self.assertRaisesRegex(ValueError, "zero arguments or one"):
                registry.instantiate_plugin_entrypoint(selected_for(two_arg))
            with self.assertRaisesRegex(ValueError, "factory is not callable"):
                registry.load_plugin_entrypoint(selected_for(bad_factory))
            with self.assertRaisesRegex(ValueError, "factory is not callable"):
                registry.load_plugin_entrypoint(selected_for(missing_factory))

            no_entry = self._write_plugin(root, "agent_runtimes", "no_entry", "agent_runtime")
            with self.assertRaisesRegex(ValueError, "does not declare an entrypoint"):
                registry.load_plugin_entrypoint(selected_for(no_entry))

            malformed = self._write_plugin(
                root,
                "agent_runtimes",
                "malformed",
                "agent_runtime",
                entrypoint="adapter",
                code={"adapter.py": "def create_runtime():\n    return None\n"},
            )
            malformed_selected = selected_for(malformed)
            with self.assertRaisesRegex(ValueError, "module:function"):
                registry.load_plugin_entrypoint(malformed_selected)

            missing_module = self._write_plugin(
                root,
                "agent_runtimes",
                "missing_module",
                "agent_runtime",
                entrypoint="adapter:create_runtime",
                code={"adapter.py": "def create_runtime():\n    return None\n"},
            )
            missing_module.joinpath("adapter.py").unlink()
            with self.assertRaisesRegex(ValueError, "file not found"):
                registry.load_plugin_entrypoint(selected_for(missing_module))

            not_object = root / "not_object" / "agent_runtimes" / "bad"
            not_object.mkdir(parents=True)
            not_object.joinpath("plugin.yaml").write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest must be an object"):
                registry.read_plugin_metadata(not_object)

            for manifest, message in (
                ({"name": "bad", "kind": "unknown"}, "Unknown plugin kind"),
                (
                    {"name": "bad", "kind": "agent_runtime", "stability": "experimental"},
                    "stability",
                ),
                (
                    {"name": "bad", "kind": "agent_runtime", "protocol_version": 2},
                    "protocol_version",
                ),
                (
                    {
                        "name": "bad",
                        "kind": "agent_runtime",
                        "entrypoint": "adapter:create_runtime",
                        "code": [],
                    },
                    "entrypoint module",
                ),
            ):
                with self.subTest(message=message):
                    metadata = registry.PluginMetadata.from_manifest(manifest)
                    with self.assertRaisesRegex(ValueError, message):
                        registry._validate_metadata(metadata)

    def test_task_project_plugins_discoverable_under_bundled_execution(self) -> None:
        """task-supplied plugins (--task-path) must be discoverable and selectable when
        enforce_bundled_execution=True, and should outrank a same-named bundled plugin.

        Regression coverage for a task-local plugin under
        ``<task-project>/.praxist/plugins/`` being omitted when plugin roots only
        include workspace-side projects and bundled execution rejects project-source
        candidates.
        """
        from praxist.core import registry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            task_path = root / "task_project"
            task_plugins = task_path / ".praxist" / "plugins"
            bundled = root / "bundled"

            self._write_plugin(
                bundled,
                "panel_topologies",
                "rocket_panel",
                "panel_topology",
                version="0.1.0",
            )
            self._write_plugin(
                task_plugins,
                "panel_topologies",
                "rocket_panel",
                "panel_topology",
                version="1.0.0",
            )
            self._write_plugin(
                task_plugins,
                "panel_topologies",
                "task_only_panel",
                "panel_topology",
            )

            defaults = registry.PluginRoots.defaults(workspace, task_path=task_path)
            self.assertEqual(
                defaults.task_project,
                [(task_path / ".praxist" / "plugins").resolve()],
            )

            roots = registry.PluginRoots(
                bundled=[bundled],
                project=[],
                user=[],
                task_project=[task_plugins],
            )
            loader = registry.PluginLoader(roots)
            report = loader.discover()

            sources_by_key: dict[tuple[str, str], list[str]] = {}
            for candidate in report.candidates:
                sources_by_key.setdefault(candidate.identity.key(), []).append(candidate.source)
            self.assertEqual(
                set(sources_by_key[("panel_topology", "rocket_panel")]),
                {"bundled", "task_project"},
            )
            self.assertEqual(
                sources_by_key[("panel_topology", "task_only_panel")], ["task_project"]
            )

            manifest = loader.resolve(
                ["panel_topology:rocket_panel", "panel_topology:task_only_panel"],
                report,
                run_id="run",
                root_task_ref="task:test",
                enforce_bundled_execution=True,
            )

            selected = {
                (item["metadata"]["kind"], item["metadata"]["name"]): item
                for item in manifest["selected"]
            }
            self.assertEqual(selected[("panel_topology", "rocket_panel")]["source"], "task_project")
            self.assertEqual(
                selected[("panel_topology", "task_only_panel")]["source"], "task_project"
            )
            self.assertEqual(manifest["execution_source_policy"], "bundled_only")
            registry.assert_bundled_execution_manifest(manifest)

    def test_plugin_roots_defaults_extra_kwargs_bypass_env_reads(self) -> None:
        """Issue #75 batch 6: explicit ``extra_bundled`` / ``extra_project``

        kwargs win over ``PRAXIST_BUNDLED_PLUGIN_ROOTS`` / ``PRAXIST_PLUGIN_ROOTS``
        env reads. The env-fallback only kicks in when the kwarg is
        ``None``; passing an empty list explicitly disables the fallback
        (useful for hermetic builds).
        """
        import os
        from unittest.mock import patch

        from praxist.core import registry

        with tempfile.TemporaryDirectory() as tmp:
            # ``.resolve()`` is needed on macOS where ``/var`` is a
            # symlink to ``/private/var`` — ``_plugin_roots_from_env``
            # resolves its paths, so we have to too for membership
            # assertions to match.
            root = Path(tmp).resolve()
            explicit_bundled = root / "explicit_bundled"
            explicit_project = root / "explicit_project"
            env_bundled = root / "env_bundled"
            env_project = root / "env_project"
            for d in (explicit_bundled, explicit_project, env_bundled, env_project):
                d.mkdir()

            workspace = root / "workspace"
            workspace.mkdir()

            with patch.dict(
                os.environ,
                {
                    "PRAXIST_BUNDLED_PLUGIN_ROOTS": str(env_bundled),
                    "PRAXIST_PLUGIN_ROOTS": str(env_project),
                },
                clear=False,
            ):
                # Explicit kwargs win over env values.
                explicit_roots = registry.PluginRoots.defaults(
                    workspace=workspace,
                    extra_bundled=[explicit_bundled],
                    extra_project=[explicit_project],
                )
                self.assertIn(explicit_bundled, explicit_roots.bundled)
                self.assertIn(explicit_project, explicit_roots.project)
                self.assertNotIn(env_bundled, explicit_roots.bundled)
                self.assertNotIn(env_project, explicit_roots.project)

                # None kwargs → env fallback (legacy / fixture / operator path).
                env_roots = registry.PluginRoots.defaults(workspace=workspace)
                self.assertIn(env_bundled, env_roots.bundled)
                self.assertIn(env_project, env_roots.project)

                # Empty list explicitly disables the env-fallback —
                # hermetic-build mode.
                hermetic = registry.PluginRoots.defaults(
                    workspace=workspace,
                    extra_bundled=[],
                    extra_project=[],
                )
                self.assertNotIn(env_bundled, hermetic.bundled)
                self.assertNotIn(env_project, hermetic.project)
                # Bundled / project still include their default roots
                # (repo_root and workspace's .praxist/plugins).
                self.assertEqual(len(hermetic.bundled), 1)
                self.assertTrue(str(hermetic.bundled[0]).endswith("praxist/plugins"))

    def test_validate_metadata_skips_stability_check_for_task_project_source(self) -> None:
        """Issue #80: task-local plugins are gated by source trust, not the

        stability tier. ``_validate_metadata`` must accept any stability
        value when ``source=='task_project'`` so authors can keep their
        in-flight task-local plugins at ``v0_experimental`` without a
        manual workaround.
        """
        from praxist.core import registry

        # Same manifest that the bundled path rejects (see the existing
        # subtests around ``"stability"`` above) — only ``source`` changes.
        metadata = registry.PluginMetadata.from_manifest(
            {
                "schema_version": 1,
                "name": "task_local",
                "kind": "agent_runtime",
                "version": "0.1.0",
                "protocol_version": 1,
                "stability": "v0_experimental",
            }
        )
        # Bundled / project / user sources continue to enforce the contract.
        for non_task_source in ("bundled", "project", "user", None):
            with (
                self.subTest(source=non_task_source),
                self.assertRaisesRegex(ValueError, "stability"),
            ):
                registry._validate_metadata(metadata, source=non_task_source)
        # task_project source skips the stability mismatch check.
        registry._validate_metadata(metadata, source="task_project")
        # Other field validations still apply for task_project sources too —
        # e.g. an unknown kind is still rejected.
        bad_kind = registry.PluginMetadata.from_manifest(
            {
                "schema_version": 1,
                "name": "task_local_bad",
                "kind": "definitely_not_a_kind",
                "version": "0.1.0",
                "protocol_version": 1,
            }
        )
        with self.assertRaisesRegex(ValueError, "Unknown plugin kind"):
            registry._validate_metadata(bad_kind, source="task_project")

    def test_task_project_plugin_with_experimental_stability_resolves(self) -> None:
        """End-to-end: a task-local plugin shipped with ``stability:

        v0_experimental`` resolves successfully under
        ``enforce_bundled_execution=True``, while a bundled plugin with the
        same stability still fails the strict gate (#80 keeps the bundled
        contract intact).
        """
        from praxist.core import registry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            task_path = root / "task_project"
            task_plugins = task_path / ".praxist" / "plugins"
            bundled = root / "bundled"

            self._write_plugin(
                task_plugins,
                "panel_topologies",
                "task_v0_panel",
                "panel_topology",
                stability="v0_experimental",
            )

            roots = registry.PluginRoots(
                bundled=[bundled],
                project=[],
                user=[],
                task_project=[task_plugins],
            )
            loader = registry.PluginLoader(roots)
            report = loader.discover()

            manifest = loader.resolve(
                ["panel_topology:task_v0_panel"],
                report,
                run_id="run",
                root_task_ref="task:test",
                enforce_bundled_execution=True,
            )
            selected = {
                (item["metadata"]["kind"], item["metadata"]["name"]): item
                for item in manifest["selected"]
            }
            self.assertEqual(
                selected[("panel_topology", "task_v0_panel")]["source"], "task_project"
            )
            self.assertEqual(
                selected[("panel_topology", "task_v0_panel")]["metadata"]["stability"],
                "v0_experimental",
            )

            # Bundled plugin with same non-compliant stability is still
            # rejected — task_project is the only relaxed surface.
            self._write_plugin(
                bundled,
                "panel_topologies",
                "bundled_v0_panel",
                "panel_topology",
                stability="v0_experimental",
            )
            bundled_loader = registry.PluginLoader(
                registry.PluginRoots(
                    bundled=[bundled],
                    project=[],
                    user=[],
                    task_project=[],
                )
            )
            bundled_report = bundled_loader.discover()
            with self.assertRaisesRegex(ValueError, "stability"):
                bundled_loader.resolve(
                    ["panel_topology:bundled_v0_panel"],
                    bundled_report,
                    run_id="run",
                    root_task_ref="task:test",
                    enforce_bundled_execution=True,
                )

    def _write_plugin(
        self,
        root: Path,
        kind_dir: str,
        name: str,
        kind: str,
        *,
        version: str = "0.1.0",
        stability: str | None = None,
        dependencies: list[dict[str, object]] | None = None,
        entrypoint: str | None = None,
        code: dict[str, str] | None = None,
    ) -> Path:
        from praxist.core.registry import _stability_for_kind

        plugin_dir = root / kind_dir / name
        plugin_dir.mkdir(parents=True, exist_ok=True)
        code = code or {}
        for rel, text in code.items():
            path = plugin_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        code_list = list(code)
        manifest = {
            "schema_version": 1,
            "name": name,
            "kind": kind,
            "version": version,
            "protocol_version": 1,
            "stability": stability or _stability_for_kind(kind),
            "description": f"{name} test plugin",
            "compatibility": {"praxist_core": ">=0.1.0", "python": ">=3.11"},
            "dependencies": dependencies or [],
            "capabilities": ["runtime.test"] if kind == "agent_runtime" else [],
            "entrypoint": entrypoint,
            "code": code_list,
            "assets": [],
        }
        plugin_dir.joinpath("plugin.yaml").write_text(
            "\n".join(
                [
                    "schema_version: 1",
                    f"name: {name}",
                    f"kind: {kind}",
                    f"version: {version}",
                    "protocol_version: 1",
                    f"stability: {manifest['stability']}",
                    f"description: {name} test plugin",
                    "compatibility:",
                    '  praxist_core: ">=0.1.0"',
                    '  python: ">=3.11"',
                    "dependencies:",
                    *[
                        f"  - kind: {dep['kind']}\n    name: {dep['name']}"
                        for dep in manifest["dependencies"]
                    ],
                    "capabilities:",
                    *[f"  - {cap}" for cap in manifest["capabilities"]],
                    f"entrypoint: {entrypoint or ''}",
                    "code:",
                    *[f"  - {rel}" for rel in code_list],
                    "assets: []",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return plugin_dir


if __name__ == "__main__":
    unittest.main()
