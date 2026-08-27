"""``praxist docs`` — open the hosted Praxist documentation."""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from collections.abc import Mapping

DOCUMENTATION_URL = "https://praxist.sapient.inc/en/docs"


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``praxist docs`` subcommand."""
    parser = subparsers.add_parser(
        "docs",
        help="Open or print the hosted Praxist documentation.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Print the documentation URL without opening a browser.",
    )
    parser.set_defaults(func=cmd_docs)


def cmd_docs(args: argparse.Namespace) -> int:
    """Print the canonical URL and open it when the host has a local browser."""

    url = os.environ.get("PRAXIST_DOCS_URL", "").strip() or DOCUMENTATION_URL
    opened = False
    if not args.no_open and _can_open_browser(os.environ, sys.platform):
        try:
            opened = bool(webbrowser.open_new_tab(url))
        except (OSError, webbrowser.Error):
            opened = False

    print(url)
    if opened:
        print("Opened Praxist documentation in the default browser.", file=sys.stderr)
    elif not args.no_open:
        print("No local browser detected; open the documentation URL above.", file=sys.stderr)
    return 0


def _can_open_browser(environment: Mapping[str, str], platform: str) -> bool:
    """Return whether browser launch is appropriate for this process."""

    if _truthy(environment.get("PRAXIST_DOCS_NO_OPEN")) or _truthy(environment.get("CI")):
        return False
    if environment.get("SSH_CONNECTION") or environment.get("SSH_TTY"):
        return False
    if platform == "darwin":
        return True
    if platform.startswith("linux"):
        return bool(environment.get("DISPLAY") or environment.get("WAYLAND_DISPLAY"))
    return False


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
