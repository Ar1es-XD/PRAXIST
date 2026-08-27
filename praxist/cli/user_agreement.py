"""``praxist user-agreement`` — review and accept current legal terms."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import TextIO

from praxist.cli._terminal_ui import (
    Choice,
    TerminalInteractionCancelled,
    TerminalInteractionError,
    interactive_terminal_available,
    select_choice,
    view_scrollable_text,
)
from praxist.cli.docs import DOCUMENTATION_URL
from praxist.user_agreement import (
    FAIR_SOURCE_LICENSE_VERSION,
    USER_AGREEMENT_VERSION,
    acceptance_path,
    current_acceptance,
    record_acceptance,
    user_agreement_sha256,
    user_agreement_text,
)

USER_AGREEMENT_URL = f"{DOCUMENTATION_URL.rstrip('/')}/legal/user-agreement.html"
FAIR_SOURCE_LICENSE_URL = "https://github.com/sapientinc/praxist/blob/main/LICENSE.md"


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register ``praxist user-agreement``."""

    parser = subparsers.add_parser(
        "user-agreement",
        help="Review the Praxist License and User Agreement or inspect acceptance status.",
    )
    commands = parser.add_subparsers(dest="user_agreement_command", metavar="<command>")

    review = commands.add_parser("review", help="Review the complete legal terms.")
    review.add_argument(
        "--print",
        action="store_true",
        dest="print_full",
        help="Print the full text instead of using a temporary scroll view.",
    )
    review.set_defaults(func=cmd_review)

    accept = commands.add_parser("accept", help="Record explicit acceptance.")
    accept.add_argument(
        "--agent-reply",
        default=None,
        help="Exact Agent-assisted reply; only Agree is accepted.",
    )
    accept.set_defaults(func=cmd_accept)

    status = commands.add_parser("status", help="Show acceptance for the current legal terms.")
    status.add_argument("--json", action="store_true", dest="json_output")
    status.set_defaults(func=cmd_status)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone User Agreement command group."""

    parser = argparse.ArgumentParser(prog="praxist user-agreement")
    commands = parser.add_subparsers(dest="user_agreement_command", metavar="<command>")
    review = commands.add_parser("review")
    review.add_argument("--print", action="store_true", dest="print_full")
    review.set_defaults(func=cmd_review)
    accept = commands.add_parser("accept")
    accept.add_argument("--agent-reply", default=None)
    accept.set_defaults(func=cmd_accept)
    status = commands.add_parser("status")
    status.add_argument("--json", action="store_true", dest="json_output")
    status.set_defaults(func=cmd_status)
    args = parser.parse_args(list(argv or ()))
    if not args.user_agreement_command:
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


def cmd_review(args: argparse.Namespace) -> int:
    """Review without changing acceptance state."""

    if args.print_full:
        print(user_agreement_text())
        return 0
    if not interactive_terminal_available():
        print(FAIR_SOURCE_LICENSE_URL)
        print(USER_AGREEMENT_URL)
        print(
            "Open the URLs above to review the complete legal terms, or run "
            "`praxist user-agreement review --print`.",
            file=sys.stderr,
        )
        return 0
    try:
        _review_in_terminal()
    except (TerminalInteractionCancelled, TerminalInteractionError) as exc:
        print(f"praxist user-agreement: {exc}", file=sys.stderr)
        return 130
    return 0


def cmd_accept(args: argparse.Namespace) -> int:
    """Record one direct or explicitly relayed Agent acceptance."""

    existing = current_acceptance()
    if existing is not None:
        print(f"Praxist legal terms {USER_AGREEMENT_VERSION} are already accepted.")
        return 0
    if args.agent_reply is not None:
        if args.agent_reply.strip() != "Agree":
            print(
                "praxist user-agreement: only the operator's exact `Agree` reply can be recorded",
                file=sys.stderr,
            )
            return 2
        accepted = record_acceptance(source="agent")
        print(f"Praxist legal terms accepted at {accepted.accepted_at}.")
        return 0
    if not interactive_terminal_available():
        print(
            "praxist user-agreement: a local interactive terminal is required; review the "
            f"License at {FAIR_SOURCE_LICENSE_URL} and User Agreement at {USER_AGREEMENT_URL}",
            file=sys.stderr,
        )
        return 2
    try:
        accepted = prompt_for_acceptance_if_needed()
    except (TerminalInteractionCancelled, TerminalInteractionError) as exc:
        print(f"praxist user-agreement: {exc}", file=sys.stderr)
        return 130
    return 0 if accepted else 130


def cmd_status(args: argparse.Namespace) -> int:
    """Report whether the exact current legal terms have been accepted."""

    record = current_acceptance()
    payload = {
        "accepted": record is not None,
        "agreement_version": USER_AGREEMENT_VERSION,
        "agreement_sha256": user_agreement_sha256(),
        "license_version": FAIR_SOURCE_LICENSE_VERSION,
        "accepted_at": record.accepted_at if record is not None else None,
        "source": record.source if record is not None else None,
        "record_path": str(acceptance_path()),
        "review_url": USER_AGREEMENT_URL,
        "license_url": FAIR_SOURCE_LICENSE_URL,
    }
    if args.json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif record is None:
        print(f"not accepted (current version: {USER_AGREEMENT_VERSION})")
    else:
        print(f"accepted {record.accepted_at} ({record.source})")
    return 0


def prompt_for_acceptance_if_needed(*, output_stream: TextIO | None = None) -> bool:
    """Run the compact first-use legal gate and return whether setup may continue."""

    if current_acceptance() is not None:
        return True
    target = output_stream or sys.stderr
    while True:
        choice = select_choice(
            "Praxist License and User Agreement",
            (
                Choice(
                    "review",
                    "Review the complete legal terms",
                    "opens the License, User Agreement, and data notice; returning does not accept",
                ),
                Choice(
                    "agree",
                    "I have reviewed and agree",
                    "records this exact legal bundle and continues setup",
                ),
                Choice(
                    "cancel",
                    "Cancel setup",
                    "does not accept the legal terms or change runtime configuration",
                ),
            ),
            default=0,
            input_stream=sys.stdin,
            output_stream=target,
        )
        if choice == "review":
            try:
                _review_in_terminal(output_stream=target)
            except TerminalInteractionError:
                print(
                    f"Review the License at: {FAIR_SOURCE_LICENSE_URL}\n"
                    f"Review the User Agreement at: {USER_AGREEMENT_URL}",
                    file=target,
                )
            continue
        if choice == "cancel":
            return False
        accepted = record_acceptance(source="direct")
        print(f"Praxist legal terms accepted at {accepted.accepted_at}.", file=target)
        return True


def _review_in_terminal(*, output_stream: TextIO | None = None) -> None:
    view_scrollable_text(
        f"Praxist Legal Terms {USER_AGREEMENT_VERSION}",
        user_agreement_text(),
        input_stream=sys.stdin,
        output_stream=output_stream or sys.stderr,
    )
