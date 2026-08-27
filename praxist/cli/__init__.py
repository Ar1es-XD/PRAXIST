"""``praxist`` — Praxist command-line interface (package).

The package exposes :func:`main` as the ``praxist`` console-script entry point
and registers subcommand modules.  Subcommand implementations live in
sibling modules under this package.

The dispatcher contract (tests in ``tests/unit/test_cli.py``):

* ``main(argv=None)`` is the console-script entry point.
* No subcommand → print top-level help and exit 0 (so ``praxist`` alone is a
  no-op rather than an error).
* Unknown subcommand → argparse's standard ``invalid choice`` error
  (exit 2).

Output discipline: data → stdout, decorations and hints → stderr.

CLI compatibility window: ``praxist <verb>`` subcommands coexist with the
lower-level ``python -m praxist.run`` entry point. Adding a new ``praxist``
subcommand does not deprecate that Python path until the v1.0 public release
window (see AGENTS.md §16, §34).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from praxist import __version__
from praxist.cli import configure_llm as _configure_llm_module
from praxist.cli import docs as _docs_module
from praxist.cli import doctor as _doctor_module
from praxist.cli import examples as _examples_module
from praxist.cli import install_skills as _install_skills_module
from praxist.cli import product_usage as _product_usage_module
from praxist.cli import resolve as _resolve_module
from praxist.cli import resume as _resume_module
from praxist.cli import setup as _setup_module
from praxist.cli import start as _start_module
from praxist.cli import status as _status_module
from praxist.cli import stop as _stop_module
from praxist.cli import takeover as _takeover_module
from praxist.cli import uninstall as _uninstall_module
from praxist.cli import user_agreement as _user_agreement_module

SubcommandHandler = Callable[[argparse.Namespace], int | None]

__all__ = ["main", "SubcommandHandler"]


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``praxist`` argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="praxist",
        description=(
            "Praxist — multi-agent autonomous research system.\n\n"
            "Designed for both human operators and AI research agents.\n"
            "Data goes to stdout; decorations and hints go to stderr.\n\n"
            "Live dashboard: praxist --monitor [--run-id RUN_ID | --latest]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    _configure_llm_module.register(subparsers)
    _docs_module.register(subparsers)
    _doctor_module.register(subparsers)
    _examples_module.register(subparsers)
    _install_skills_module.register(subparsers)
    _product_usage_module.register(subparsers)
    from praxist.cli import monitor as _monitor_module

    _monitor_module.register(subparsers)
    _resolve_module.register(subparsers)
    _resume_module.register(subparsers)
    _setup_module.register(subparsers)
    _start_module.register(subparsers)
    _status_module.register(subparsers)
    _stop_module.register(subparsers)
    _takeover_module.register(subparsers)
    _uninstall_module.register(subparsers)
    _user_agreement_module.register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the top-level ``praxist`` console-script dispatcher.

    Args:
        argv: Optional argument vector for tests or embedded callers.  When
            omitted, argparse reads from ``sys.argv``.
    """
    parser = _build_parser()
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv and effective_argv[0] == "--monitor":
        effective_argv[0] = "monitor"
    if effective_argv and effective_argv[0] == "--takeover":
        effective_argv[0] = "takeover"
    args = parser.parse_args(effective_argv)
    if not args.command:
        parser.print_help()
        sys.exit(0)
    handler: SubcommandHandler | None = getattr(args, "func", None)
    if handler is None:  # pragma: no cover - guard against a future registration bug
        parser.error(f"subcommand {args.command!r} is not yet implemented")
    sys.exit(handler(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    main()
