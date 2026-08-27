"""Consent management for Praxist's optional pseudonymous product usage."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from typing import Any, TextIO

from praxist.cli._terminal_ui import (
    Choice,
    TerminalInteractionCancelled,
    TerminalInteractionError,
    select_choice,
    view_scrollable_text,
)
from praxist.cli.docs import DOCUMENTATION_URL

PRODUCT_USAGE_NOTICE_URL = f"{DOCUMENTATION_URL.rstrip('/')}/legal/product-usage-data-notice.html"


class ProductUsageUnavailableError(RuntimeError):
    """Raised when the built-in product-usage component cannot be loaded."""


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register ``praxist product-usage``."""

    parser = subparsers.add_parser(
        "product-usage",
        help="Review or change pseudonymous product-usage consent.",
        description=(
            "Review or change pseudonymous V2 product-usage consent. "
            "Withdrawal stops future capture and deletes unsent local events; "
            "delivered events expire through scheduled retention."
        ),
    )
    commands = parser.add_subparsers(dest="product_usage_command", metavar="<command>")

    notice = commands.add_parser("notice", help="Show the complete consent notice.")
    notice.set_defaults(func=cmd_notice)

    consent = commands.add_parser("consent", help="Record an explicit Yes or No choice.")
    consent.add_argument(
        "--agent-reply",
        default=None,
        help="Explicit Agent-assisted reply: Yes, No, Agree, or Disagree.",
    )
    consent.set_defaults(func=cmd_consent)

    status = commands.add_parser("status", help="Show the current consent state.")
    status.add_argument("--json", action="store_true", dest="json_output")
    status.set_defaults(func=cmd_status)

    withdraw = commands.add_parser(
        "withdraw",
        help="Deny future collection and delete unsent events.",
    )
    withdraw.set_defaults(func=cmd_withdraw)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the product-usage subcommand in isolation."""

    parser = argparse.ArgumentParser(prog="praxist product-usage")
    commands = parser.add_subparsers(dest="product_usage_command", metavar="<command>")
    notice = commands.add_parser("notice")
    notice.set_defaults(func=cmd_notice)
    consent = commands.add_parser("consent")
    consent.add_argument("--agent-reply", default=None)
    consent.set_defaults(func=cmd_consent)
    status = commands.add_parser("status")
    status.add_argument("--json", action="store_true", dest="json_output")
    status.set_defaults(func=cmd_status)
    withdraw = commands.add_parser("withdraw")
    withdraw.set_defaults(func=cmd_withdraw)
    args = parser.parse_args(list(argv or ()))
    if not args.product_usage_command:
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


def cmd_notice(_args: argparse.Namespace) -> int:
    """Print the current notice for the V2 product-usage protocol."""

    try:
        notice = _product_usage_notice()
    except ProductUsageUnavailableError:
        print(
            "praxist product-usage: the built-in product-usage component is unavailable",
            file=sys.stderr,
        )
        return 1
    print(notice)
    return 0


def cmd_consent(args: argparse.Namespace) -> int:
    """Record one explicit direct or Agent-assisted consent decision."""

    try:
        sdk = _usage_sdk()
    except ProductUsageUnavailableError as exc:
        print(f"praxist product-usage: {exc}", file=sys.stderr)
        return 1
    current = sdk.consent_status
    if not _collection_transport_available():
        value = getattr(current, "value", str(current))
        print(f"Product-usage collection is unavailable in this build; stored consent is {value}.")
        return 0
    if getattr(current, "value", str(current)) == "granted":
        print(f"Product-usage consent is already {current.value}.")
        return 0

    try:
        _product_usage_notice()
    except ProductUsageUnavailableError as exc:
        print(f"praxist product-usage: {exc}", file=sys.stderr)
        return 1

    reply = args.agent_reply
    if reply is not None:
        status = sdk.record_agent_reply(reply)
    elif not sys.stdin.isatty():
        print("Consent remains unset; no product-usage data will be collected.")
        return 0
    else:
        try:
            choice = _select_consent_choice()
        except (TerminalInteractionCancelled, TerminalInteractionError):
            print("Consent remains unset; no product-usage data will be collected.")
            return 0
        status = sdk.record_direct_choice(choice)

    value = getattr(status, "value", str(status))
    if value == "unset":
        print("No explicit supported choice was recorded; consent remains unset.")
    else:
        print(f"Product-usage consent recorded: {value}.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Print the current V2 consent state."""

    try:
        payload = read_product_usage_status()
    except ProductUsageUnavailableError as exc:
        print(f"praxist product-usage: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(payload["status"])
    return 0


def read_product_usage_status() -> dict[str, object]:
    """Return the current optional collection state without changing it."""

    status = _usage_sdk().consent_status
    return {
        "collection_available": _collection_transport_available(),
        "status": getattr(status, "value", str(status)),
    }


def cmd_withdraw(_args: argparse.Namespace) -> int:
    """Stop future capture and remove unsent local events."""

    try:
        withdrawn = _usage_sdk().withdraw()
    except ProductUsageUnavailableError as exc:
        print(f"praxist product-usage: {exc}", file=sys.stderr)
        return 1
    if not withdrawn:
        print(
            "Withdrawal was not fully persisted or unsent events could not be removed; "
            "collection is disabled in this process, but local permissions need repair.",
            file=sys.stderr,
        )
        return 1
    print("Product-usage consent withdrawn; unsent events deleted.")
    return 0


def prompt_for_consent_if_unset(*, output_stream: TextIO | None = None) -> bool:
    """Offer the current notice and report whether setup may continue."""

    if not _collection_transport_available():
        if output_stream is not None:
            print(
                "Product-usage collection is unavailable in this build; privacy setup was "
                "skipped and no usage data will be collected.",
                file=output_stream,
            )
        return True
    target = output_stream or sys.stdout
    try:
        sdk = _usage_sdk()
    except ProductUsageUnavailableError:
        return True
    current = sdk.consent_status
    if getattr(current, "value", str(current)) != "unset" or not sys.stdin.isatty():
        return True
    try:
        _product_usage_notice()
    except ProductUsageUnavailableError:
        return True
    try:
        choice = _select_consent_choice()
    except (TerminalInteractionCancelled, TerminalInteractionError):
        return False
    status = sdk.record_direct_choice(choice)
    value = getattr(status, "value", str(status))
    if value == "unset":
        print("No explicit supported choice was recorded; consent remains unset.", file=target)
    else:
        print(f"Product-usage consent recorded: {value}.", file=target)
    return True


def _select_consent_choice() -> str:
    """Return an explicit supported consent token from a local TTY choice."""

    while True:
        choice = select_choice(
            "Optional product-usage data",
            (
                Choice(
                    "review",
                    "Review the complete data notice",
                    "opens a temporary scroll view; this does not grant consent",
                ),
                Choice("Yes", "Share product usage", "bounded pseudonymized lifecycle data"),
                Choice("No", "Skip collection", "research works normally without collection"),
            ),
            default=None,
            input_stream=sys.stdin,
            output_stream=sys.stderr,
        )
        if choice != "review":
            return choice
        try:
            view_scrollable_text(
                "Praxist User Data Collection Notice",
                _product_usage_notice(),
                input_stream=sys.stdin,
                output_stream=sys.stderr,
            )
        except (ProductUsageUnavailableError, TerminalInteractionError):
            print(f"Review the complete notice at: {PRODUCT_USAGE_NOTICE_URL}", file=sys.stderr)


def _usage_sdk() -> Any:
    try:
        usage_client = importlib.import_module("praxist.product_usage.client")
    except ImportError as exc:
        raise ProductUsageUnavailableError(
            "the built-in product-usage component is unavailable"
        ) from exc
    return usage_client.UsageSdk()


def _product_usage_notice() -> str:
    try:
        integration = importlib.import_module("praxist.infrastructure.product_usage")
        return str(integration.product_usage_notice())
    except Exception as exc:
        raise ProductUsageUnavailableError(
            "the built-in product-usage component is unavailable"
        ) from exc


def _collection_transport_available() -> bool:
    try:
        from praxist import __version__

        transport = importlib.import_module("praxist.product_usage.transport")
        transport.default_batch_sender(__version__)
    except Exception:
        return False
    return True
