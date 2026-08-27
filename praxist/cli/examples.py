"""Install bundled examples as writable projects outside Praxist source."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

BUNDLED_EXAMPLES = (
    "rocket_booster_recovery",
    "rocket_booster_recovery_rust",
)
EXAMPLES_HOME_ENV = "PRAXIST_EXAMPLES_HOME"


class ExampleInstallError(RuntimeError):
    """Raised when a bundled example cannot be materialized safely."""


@dataclass(frozen=True)
class ExampleInstallResult:
    """Result of materializing one bundled example."""

    name: str
    destination: str
    status: str
    dry_run: bool


def default_examples_home() -> Path:
    """Return the user-owned directory for writable example projects."""

    configured = os.environ.get(EXAMPLES_HOME_ENV, "").strip()
    return Path(configured).expanduser() if configured else Path.home() / "PraxistExamples"


def _bundled_examples_root() -> Traversable:
    packaged = resources.files("praxist").joinpath("resources/examples")
    if packaged.is_dir():
        return packaged
    return Path(__file__).resolve().parents[2] / "examples"


def _ensure_external_destination(target: Path) -> None:
    """Keep writable example projects outside source and installed packages."""

    installation_root = Path(__file__).resolve().parents[2]
    try:
        target.relative_to(installation_root)
    except ValueError:
        return
    raise ExampleInstallError(
        "example destination must live outside the Praxist source or installation "
        f"directory: {target}"
    )


def materialize_example(
    name: str,
    *,
    destination: Path | None = None,
    dry_run: bool = False,
    source_root: Path | None = None,
) -> ExampleInstallResult:
    """Copy one read-only bundled example to a user-owned working directory.

    Existing destinations are preserved byte-for-byte. This keeps installer
    upgrades from replacing research work or run artifacts.
    """

    if name not in BUNDLED_EXAMPLES:
        raise ExampleInstallError(f"unknown bundled example: {name}")
    target = (destination or (default_examples_home() / name)).expanduser().resolve()
    _ensure_external_destination(target)
    if target.exists():
        if not target.is_dir():
            raise ExampleInstallError(f"example destination is not a directory: {target}")
        return ExampleInstallResult(name, str(target), "preserved_existing", dry_run)
    if dry_run:
        return ExampleInstallResult(name, str(target), "would_install", True)

    source_ref = (source_root or _bundled_examples_root()).joinpath(name)
    with resources.as_file(source_ref) as source:
        if not source.is_dir():
            raise ExampleInstallError(f"bundled example is missing: {source}")
        git_artifacts = [path for path in source.rglob("*") if path.name.lower().startswith(".git")]
        if git_artifacts:
            raise ExampleInstallError("bundled example contains forbidden Git metadata")

        target.parent.mkdir(parents=True, exist_ok=True)
        stage = target.parent / f".{target.name}.install-{uuid.uuid4().hex}"
        try:
            shutil.copytree(source, stage)
            if target.exists():
                shutil.rmtree(stage)
                return ExampleInstallResult(name, str(target), "preserved_existing", False)
            stage.rename(target)
        except OSError as exc:
            shutil.rmtree(stage, ignore_errors=True)
            raise ExampleInstallError(f"could not install example at {target}: {exc}") from exc
    return ExampleInstallResult(name, str(target), "installed", False)


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``praxist examples`` command group."""

    parser = subparsers.add_parser(
        "examples",
        help="List or install complete writable example projects.",
    )
    actions = parser.add_subparsers(dest="examples_action", metavar="<action>", required=True)

    list_parser = actions.add_parser("list", help="List bundled complete examples.")
    list_parser.add_argument("--json", action="store_true", dest="json_output")
    list_parser.set_defaults(func=cmd_examples_list)

    install_parser = actions.add_parser(
        "install",
        help="Copy a bundled example outside the Praxist installation.",
    )
    install_parser.add_argument("name", choices=BUNDLED_EXAMPLES)
    install_parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help=(
            "Final project directory (default: ${PRAXIST_EXAMPLES_HOME:-~/PraxistExamples}/NAME)."
        ),
    )
    install_parser.add_argument("--dry-run", action="store_true")
    install_parser.add_argument("--json", action="store_true", dest="json_output")
    install_parser.set_defaults(func=cmd_examples_install)


def cmd_examples_list(args: argparse.Namespace) -> int:
    """List complete examples available in this Praxist installation."""

    payload = [
        {
            "name": name,
            "default_destination": str((default_examples_home() / name).resolve()),
        }
        for name in BUNDLED_EXAMPLES
    ]
    if args.json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for item in payload:
            print(f"{item['name']}\t{item['default_destination']}")
    return 0


def cmd_examples_install(args: argparse.Namespace) -> int:
    """Materialize one complete example without replacing an existing copy."""

    try:
        result = materialize_example(
            args.name,
            destination=args.destination,
            dry_run=args.dry_run,
        )
    except ExampleInstallError as exc:
        print(f"praxist examples: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print(result.destination)
        if result.status == "installed":
            print("Installed writable Praxist example project.", file=sys.stderr)
        elif result.status == "preserved_existing":
            print("Existing example project preserved unchanged.", file=sys.stderr)
        else:
            print("Would install writable Praxist example project.", file=sys.stderr)
    return 0
