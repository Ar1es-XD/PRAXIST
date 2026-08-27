"""Best-effort SDK facade; no collection failure escapes into a Research Run."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal
from uuid import UUID

from .batching import encode_next_batch
from .consent import (
    ConsentDecision,
    ConsentStatus,
    ConsentStore,
    parse_agent_reply,
)
from .identity import EnvironmentIdentityStore
from .outbox import Outbox
from .ports import BatchSender
from .protocol import UsageEvent


class UsageSdk:
    """Consent-gated local capture with lazy outbox creation."""

    def __init__(
        self,
        consent_store: ConsentStore | None = None,
        *,
        identity_store: EnvironmentIdentityStore | None = None,
        _outbox_factory: Callable[[], Outbox] = Outbox,
    ) -> None:
        self._consent = consent_store or ConsentStore()
        self._identity = identity_store or EnvironmentIdentityStore()
        self._outbox_factory = _outbox_factory
        self._outbox: Outbox | None = None

    @property
    def consent_status(self) -> ConsentStatus:
        return self._consent.status()

    @property
    def consent_grant_id(self) -> str | None:
        """Return an atomic-file snapshot for observer initialization."""

        try:
            return self._consent.grant_id()
        except Exception:
            return None

    @property
    def environment_id(self) -> UUID | None:
        status = self._consent.status()
        if getattr(status, "value", status) != ConsentStatus.GRANTED.value:
            return None
        try:
            return self._identity.get_or_create()
        except Exception:
            return None

    def record_direct_choice(self, choice: str) -> ConsentStatus:
        """Record an explicit unselected UI choice; other input leaves state unset."""

        if choice == "Yes":
            decision = ConsentDecision.GRANTED
        elif choice == "No":
            decision = ConsentDecision.DENIED
        else:
            return self._consent.status()
        return self._record_decision(decision, source="direct")

    def record_agent_reply(self, reply: str) -> ConsentStatus:
        """Record only an approved explicit Agent-interaction keyword."""

        decision = parse_agent_reply(reply)
        if decision is None:
            return self._consent.status()
        return self._record_decision(decision, source="agent")

    def capture(self, event: UsageEvent, *, expected_grant_id: str | None = None) -> bool:
        """Queue a safe event if granted; return false on every failure mode."""

        try:
            with self._consent.capture_access():
                current_grant_id = self._consent.grant_id()
                if current_grant_id is None:
                    return False
                if expected_grant_id is not None and current_grant_id != expected_grant_id:
                    return False
                outbox = self._get_outbox()
                outbox.discard_other_grants(current_grant_id)
                return outbox.enqueue(event, grant_id=current_grant_id)
        except Exception:
            return False

    def withdraw(self) -> bool:
        """Fail closed, stop collection, and delete unsent local events."""

        return self._deny_and_delete(source="withdrawal")

    def close(self) -> None:
        if self._outbox is not None:
            self._outbox.close()
            self._outbox = None

    def _record_decision(
        self,
        decision: ConsentDecision,
        *,
        source: Literal["direct", "agent"],
    ) -> ConsentStatus:
        decision_value = getattr(decision, "value", decision)
        if decision_value == ConsentDecision.DENIED.value:
            self._deny_and_delete(source=source)
            return self._consent.status()
        try:
            self._consent.write(decision, source=source)
        except Exception:
            return ConsentStatus.UNSET
        return ConsentStatus(decision_value)

    def _deny_and_delete(
        self,
        *,
        source: Literal["direct", "agent", "withdrawal"],
    ) -> bool:
        deleted = False

        def purge_outbox() -> None:
            nonlocal deleted
            deleted = self._delete_outbox_best_effort()
            if not deleted:
                raise OSError("unsent product-usage events could not be removed")

        try:
            self._consent.write(
                ConsentDecision.DENIED,
                source=source,
                _while_locked=purge_outbox,
            )
        except Exception:
            return False
        return deleted

    def _get_outbox(self) -> Outbox:
        if self._outbox is None:
            self._outbox = self._outbox_factory()
        return self._outbox

    def _delete_outbox_best_effort(self) -> bool:
        deleted = False
        try:
            outbox = self._outbox or self._outbox_factory()
            outbox.close_and_delete()
            deleted = True
        except Exception:
            pass
        finally:
            self._outbox = None
        return deleted


class UploadCoordinator:
    """Perform one failure-isolated upload attempt through an injected sender.

    Endpoint selection and HTTP policy belong to the sender. Praxist invokes
    this coordinator from an isolated worker so collection cannot control a
    Research Run.
    """

    def __init__(
        self,
        consent_store: ConsentStore,
        outbox: Outbox,
        sender: BatchSender,
    ) -> None:
        self._consent = consent_store
        self._outbox = outbox
        self._sender = sender

    def flush_once(self) -> int:
        try:
            with self._consent.exclusive_access():
                grant_id = self._consent.grant_id()
                if grant_id is None:
                    return 0
                self._outbox.discard_other_grants(grant_id)
                batch = encode_next_batch(self._outbox.fetch_oldest(grant_id=grant_id))
                if batch is None:
                    return 0
                # The bounded sender and withdrawal share one linearization
                # boundary: a completed withdrawal can never be followed by a
                # delayed request created under the withdrawn grant.
                acknowledged = set(self._sender.send(batch.body))
                safe_acknowledgements = acknowledged.intersection(batch.event_ids)
                return self._outbox.acknowledge(
                    safe_acknowledgements,
                    grant_id=grant_id,
                )
        except Exception:
            return 0
