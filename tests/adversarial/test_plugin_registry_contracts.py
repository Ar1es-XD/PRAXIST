from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from praxist.core.registry import PluginLoader, PluginRoots


class PluginRegistryAdversarialContracts(unittest.TestCase):
    def test_declared_concrete_code_files_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin_dir = root / ".praxist" / "plugins" / "agent_runtimes" / "declared_missing"
            plugin_dir.mkdir(parents=True)
            plugin_dir.joinpath("adapter.py").write_text(
                "def create_runtime():\n    return object()\n",
                encoding="utf-8",
            )
            plugin_dir.joinpath("plugin.yaml").write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        "name: declared_missing",
                        "kind: agent_runtime",
                        "version: 0.1.0",
                        "protocol_version: 1",
                        "stability: v1_stable",
                        "description: Missing declared file should fail resolution.",
                        "compatibility:",
                        "  praxist_core: '>=0.1.0,<1.0'",
                        "  python: '>=3.11'",
                        "dependencies: []",
                        "capabilities: []",
                        "entrypoint: adapter:create_runtime",
                        "code:",
                        "  - adapter.py",
                        "  - missing_helper.py",
                        "assets: []",
                    ]
                ),
                encoding="utf-8",
            )

            loader = PluginLoader(PluginRoots.defaults(root))
            with self.assertRaises(ValueError):
                loader.resolve(
                    ["agent_runtime:declared_missing"],
                    run_id="run_adversarial_missing_code",
                    root_task_ref="test:adversarial",
                )

    def test_manifest_name_must_match_plugin_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin_dir = root / ".praxist" / "plugins" / "agent_runtimes" / "innocent_dir"
            plugin_dir.mkdir(parents=True)
            plugin_dir.joinpath("adapter.py").write_text(
                "def create_runtime():\n    return object()\n",
                encoding="utf-8",
            )
            plugin_dir.joinpath("plugin.yaml").write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        "name: spoofed_runtime",
                        "kind: agent_runtime",
                        "version: 0.1.0",
                        "protocol_version: 1",
                        "stability: v1_stable",
                        "description: Directory/name mismatch should fail.",
                        "compatibility:",
                        "  praxist_core: '>=0.1.0,<1.0'",
                        "  python: '>=3.11'",
                        "dependencies: []",
                        "capabilities: []",
                        "entrypoint: adapter:create_runtime",
                        "code:",
                        "  - adapter.py",
                        "assets: []",
                    ]
                ),
                encoding="utf-8",
            )

            loader = PluginLoader(PluginRoots.defaults(root))
            report = loader.discover()
            discovered = [
                candidate
                for candidate in report.candidates
                if candidate.identity.name == "spoofed_runtime"
            ]
            self.assertEqual(
                discovered,
                [],
                "plugin identity should be bound to its directory name to avoid misleading shadowing",
            )
