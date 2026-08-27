from __future__ import annotations

import importlib
import pkgutil
import sys
import unittest

import praxist


class ModuleReloadSmokeTest(unittest.TestCase):
    def test_all_praxist_modules_reload_under_coverage(self) -> None:
        """Keep import-time contracts visible to the default coverage profile.

        unittest imports test modules before coverage starts, so ordinary
        module-level imports do not count definition-time failures or manifest
        entrypoint imports. Reloading the system package tree inside a test
        method makes importability an explicit contract and records top-level
        definitions under coverage.
        """

        failures: list[tuple[str, str, str]] = []
        module_names = sorted(
            module_info.name
            for module_info in pkgutil.walk_packages(praxist.__path__, praxist.__name__ + ".")
        )
        original_modules = {name: sys.modules.get(name) for name in module_names}
        original_namespaces = {
            name: dict(module.__dict__)
            for name, module in original_modules.items()
            if module is not None
        }
        try:
            for module_name in module_names:
                try:
                    module = importlib.import_module(module_name)
                    importlib.reload(module)
                except Exception as exc:  # noqa: BLE001 - report every broken module.
                    failures.append((module_name, type(exc).__name__, str(exc)))
        finally:
            # Reloading mutates module objects in place. Restore every original
            # namespace so this coverage smoke test cannot alter later tests.
            for module_name in reversed(module_names):
                original = original_modules[module_name]
                if original is None:
                    sys.modules.pop(module_name, None)
                    continue
                sys.modules[module_name] = original
                original.__dict__.clear()
                original.__dict__.update(original_namespaces[module_name])

        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
