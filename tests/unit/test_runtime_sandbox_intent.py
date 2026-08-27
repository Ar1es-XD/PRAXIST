"""Unit tests for :class:`praxist.core.protocol.RuntimeSandboxIntent`.

The sandbox intent is a pure-data contract; tests verify the value
vocabularies are accepted, invalid values are rejected with informative
errors, and the JSON-serializable view round-trips cleanly.
"""

from __future__ import annotations

import unittest

from praxist.core.protocol import (
    SANDBOX_APPROVAL_VALUES,
    SANDBOX_ENFORCEMENT_APPROVAL_GATE,
    SANDBOX_ENFORCEMENT_OS_SANDBOX,
    SANDBOX_FILESYSTEM_VALUES,
    SANDBOX_NETWORK_VALUES,
    RuntimeSandboxIntent,
)


class RuntimeSandboxIntentTest(unittest.TestCase):
    """Validation contract for :class:`RuntimeSandboxIntent`."""

    def test_valid_intent_constructs_and_serializes(self) -> None:
        """A well-formed intent constructs and yields a stable dict view."""
        intent = RuntimeSandboxIntent(filesystem="workspace_write", network="off", approval="auto")
        self.assertEqual(
            intent.to_dict(),
            {"filesystem": "workspace_write", "network": "off", "approval": "auto"},
        )

    def test_intent_is_hashable_and_frozen(self) -> None:
        """The dataclass is frozen, so it works as a dict key / set member."""
        intent = RuntimeSandboxIntent(filesystem="full", network="on", approval="auto")
        bucket = {intent: "ok"}
        self.assertEqual(bucket[intent], "ok")
        with self.assertRaises(AttributeError):
            intent.filesystem = "read_only"  # type: ignore[misc]

    def test_invalid_filesystem_value_raises(self) -> None:
        """A filesystem value outside the vocabulary is rejected."""
        with self.assertRaisesRegex(ValueError, "filesystem"):
            RuntimeSandboxIntent(filesystem="bogus", network="off", approval="auto")  # type: ignore[arg-type]

    def test_invalid_network_value_raises(self) -> None:
        """A network value outside the vocabulary is rejected."""
        with self.assertRaisesRegex(ValueError, "network"):
            RuntimeSandboxIntent(filesystem="full", network="maybe", approval="auto")  # type: ignore[arg-type]

    def test_invalid_approval_value_raises(self) -> None:
        """An approval value outside the vocabulary is rejected."""
        with self.assertRaisesRegex(ValueError, "approval"):
            RuntimeSandboxIntent(filesystem="full", network="off", approval="sometimes")  # type: ignore[arg-type]

    def test_each_filesystem_value_is_accepted(self) -> None:
        """Every documented filesystem value constructs."""
        for value in SANDBOX_FILESYSTEM_VALUES:
            intent = RuntimeSandboxIntent(filesystem=value, network="on", approval="auto")  # type: ignore[arg-type]
            self.assertEqual(intent.filesystem, value)

    def test_each_network_value_is_accepted(self) -> None:
        """Every documented network value constructs."""
        for value in SANDBOX_NETWORK_VALUES:
            intent = RuntimeSandboxIntent(filesystem="full", network=value, approval="auto")  # type: ignore[arg-type]
            self.assertEqual(intent.network, value)

    def test_each_approval_value_is_accepted(self) -> None:
        """Every documented approval value constructs."""
        for value in SANDBOX_APPROVAL_VALUES:
            intent = RuntimeSandboxIntent(filesystem="full", network="on", approval=value)  # type: ignore[arg-type]
            self.assertEqual(intent.approval, value)


class SandboxEnforcementConstantsTest(unittest.TestCase):
    """The enforcement-mode constants are stable string identifiers."""

    def test_enforcement_constants_are_distinct_strings(self) -> None:
        """``approval_gate`` and ``os_sandbox`` are the only enforcement modes today."""
        self.assertEqual(SANDBOX_ENFORCEMENT_APPROVAL_GATE, "approval_gate")
        self.assertEqual(SANDBOX_ENFORCEMENT_OS_SANDBOX, "os_sandbox")
        self.assertNotEqual(SANDBOX_ENFORCEMENT_APPROVAL_GATE, SANDBOX_ENFORCEMENT_OS_SANDBOX)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
