from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class RuntimeFailureClassificationTest(unittest.TestCase):
    def test_identifiers_and_ordinary_words_do_not_become_auth_errors(self) -> None:
        from praxist.core.runtimes import classify_runtime_failure

        self.assertEqual(
            classify_runtime_failure("shared_core_id: 8105d401c78e3d8b"),
            "runtime_error",
        )
        self.assertEqual(classify_runtime_failure("authoritative result"), "runtime_error")
        self.assertEqual(classify_runtime_failure("artifact-429abc"), "runtime_error")
        self.assertEqual(classify_runtime_failure("sandbox write forbidden"), "runtime_error")

    def test_explicit_provider_statuses_keep_their_failure_classes(self) -> None:
        from praxist.core.runtimes import classify_runtime_failure

        self.assertEqual(classify_runtime_failure("HTTP 401 Unauthorized"), "auth_error")
        self.assertEqual(classify_runtime_failure("403 Forbidden"), "auth_error")
        self.assertEqual(classify_runtime_failure("authentication_error"), "auth_error")
        self.assertEqual(classify_runtime_failure("auth_error"), "auth_error")
        self.assertEqual(classify_runtime_failure("OAuth token expired"), "auth_error")
        self.assertEqual(classify_runtime_failure("oauth_error"), "auth_error")
        self.assertEqual(classify_runtime_failure("authorizationError"), "auth_error")
        self.assertEqual(classify_runtime_failure("HTTP 429 rate limited"), "rate_limited")


class RuntimeManagedCredentialForRefTest(unittest.TestCase):
    def setUp(self) -> None:
        from praxist.core.registry import PluginRegistry

        self.runtime_ref = "agent_runtime:fake_runtime"
        self.model_provider_ref = "model_provider:openai_compatible"
        self.registry = MagicMock(spec=PluginRegistry)

    def test_returns_none_without_hook_or_when_hook_returns_none(self) -> None:
        from praxist.core.runtimes import runtime_managed_credential_for_ref

        for runtime in (
            SimpleNamespace(),
            SimpleNamespace(discover_managed_credential=MagicMock(return_value=None)),
        ):
            with (
                self.subTest(runtime=runtime),
                patch("praxist.core.runtimes.runtime_for_ref", return_value=runtime),
            ):
                self.assertIsNone(
                    runtime_managed_credential_for_ref(
                        self.runtime_ref,
                        self.model_provider_ref,
                        self.registry,
                    )
                )

    def test_returns_valid_provider_scoped_reference(self) -> None:
        from praxist.core.credentials import CredentialRef
        from praxist.core.runtimes import runtime_managed_credential_for_ref

        expected = CredentialRef(
            scope="model_provider",
            provider="openai_compatible",
            target_ref=self.model_provider_ref,
            key_id="openai_compatible:runtime:account",
            source="runtime_session",
        )
        runtime = SimpleNamespace(discover_managed_credential=MagicMock(return_value=expected))
        with patch("praxist.core.runtimes.runtime_for_ref", return_value=runtime):
            observed = runtime_managed_credential_for_ref(
                self.runtime_ref,
                self.model_provider_ref,
                self.registry,
            )

        self.assertIs(observed, expected)

    def test_rejects_invalid_type_scope_provider_or_target(self) -> None:
        from praxist.core.credentials import CredentialRef
        from praxist.core.runtimes import runtime_managed_credential_for_ref

        invalid_values = (
            object(),
            CredentialRef(
                scope="tool_server",
                provider="openai_compatible",
                target_ref=self.model_provider_ref,
                key_id="wrong-scope",
                source="runtime_session",
            ),
            CredentialRef(
                scope="model_provider",
                provider="other_provider",
                target_ref=self.model_provider_ref,
                key_id="wrong-provider",
                source="runtime_session",
            ),
            CredentialRef(
                scope="model_provider",
                provider="openai_compatible",
                target_ref="model_provider:other_provider",
                key_id="wrong-target",
                source="runtime_session",
            ),
        )
        for invalid in invalid_values:
            runtime = SimpleNamespace(discover_managed_credential=MagicMock(return_value=invalid))
            expected_error = TypeError if not isinstance(invalid, CredentialRef) else ValueError
            with (
                self.subTest(invalid=invalid),
                patch("praxist.core.runtimes.runtime_for_ref", return_value=runtime),
                self.assertRaises(expected_error),
            ):
                runtime_managed_credential_for_ref(
                    self.runtime_ref,
                    self.model_provider_ref,
                    self.registry,
                )


class ResolveModelCredentialForRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        from praxist.core.registry import PluginRegistry

        self.runtime_ref = "agent_runtime:codex_sdk"
        self.provider_ref = "model_provider:openai_compatible"
        self.registry = MagicMock(spec=PluginRegistry)

    def test_env_credential_wins_without_runtime_probe(self) -> None:
        from praxist.core.credentials import CredentialRef, CredentialSet
        from praxist.core.runtimes import resolve_model_credential_for_runtime

        credential = CredentialRef(
            scope="model_provider",
            provider="openai_compatible",
            target_ref=self.provider_ref,
            key_id="openai_compatible:env:test",
            source="env",
        )
        original = CredentialSet(mode="single", credentials=[credential])
        with patch("praxist.core.runtimes.runtime_managed_credential_for_ref") as discover:
            effective, selected = resolve_model_credential_for_runtime(
                original,
                self.runtime_ref,
                self.provider_ref,
                self.registry,
                resolve_only=False,
            )

        self.assertIs(effective, original)
        self.assertIs(selected, credential)
        discover.assert_not_called()

    def test_resolve_only_never_probes_or_requires_auth(self) -> None:
        from praxist.core.credentials import CredentialSet
        from praxist.core.runtimes import resolve_model_credential_for_runtime

        original = CredentialSet(mode="single", credentials=[])
        with patch("praxist.core.runtimes.runtime_managed_credential_for_ref") as discover:
            effective, selected = resolve_model_credential_for_runtime(
                original,
                self.runtime_ref,
                self.provider_ref,
                self.registry,
                resolve_only=True,
            )

        self.assertIs(effective, original)
        self.assertIsNone(selected)
        discover.assert_not_called()

    def test_managed_reference_is_appended_without_mutating_input(self) -> None:
        from praxist.core.credentials import CredentialRef, CredentialSet
        from praxist.core.runtimes import resolve_model_credential_for_runtime

        managed = CredentialRef(
            scope="model_provider",
            provider="openai_compatible",
            target_ref=self.provider_ref,
            key_id="openai_compatible:codex_sdk:chatgpt:abc123",
            source="runtime_session",
        )
        original = CredentialSet(mode="single", credentials=[])
        with patch(
            "praxist.core.runtimes.runtime_managed_credential_for_ref",
            return_value=managed,
        ):
            effective, selected = resolve_model_credential_for_runtime(
                original,
                self.runtime_ref,
                self.provider_ref,
                self.registry,
                resolve_only=False,
            )

        self.assertEqual(original.credentials, [])
        self.assertEqual(effective.credentials, [managed])
        self.assertEqual(effective.mode, "single")
        self.assertIs(selected, managed)

    def test_missing_env_and_managed_auth_keeps_existing_error(self) -> None:
        from praxist.core.credentials import CredentialSet
        from praxist.core.runtimes import resolve_model_credential_for_runtime

        with (
            patch(
                "praxist.core.runtimes.runtime_managed_credential_for_ref",
                return_value=None,
            ),
            self.assertRaisesRegex(ValueError, "requires a matching active"),
        ):
            resolve_model_credential_for_runtime(
                CredentialSet(mode="single", credentials=[]),
                self.runtime_ref,
                self.provider_ref,
                self.registry,
                resolve_only=False,
            )


if __name__ == "__main__":
    unittest.main()
