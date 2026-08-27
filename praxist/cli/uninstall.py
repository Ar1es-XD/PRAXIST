"""Remove a user-level Praxist installation without touching research projects."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from praxist.cli._setup_common import default_skills_dir
from praxist.cli.install_skills import InstallSkillsError, uninstall_codex_skills

MANAGED_VENV_MARKER = ".praxist-managed-venv"
MANAGED_VENV_MARKER_CONTENT = "managed_by=praxist\n"
ENTRYPOINT_NAMES = ("praxist", "praxist-uninstall")


class UninstallError(RuntimeError):
    """Raised when an installation cannot be removed safely."""


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register ``praxist uninstall``."""

    parser = subparsers.add_parser(
        "uninstall",
        help="Remove the user-level Praxist installation.",
        description=(
            "Remove Praxist-managed CLI files, runtime environment, agent skills, "
            "configuration, state, and cache. Research projects, task environments, "
            "run directories, agent CLIs, Python, and uv are never removed."
        ),
    )
    _add_arguments(parser)
    parser.set_defaults(func=cmd_uninstall)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--venv-dir",
        default=None,
        help="Override the Praxist-managed virtualenv path.",
    )
    parser.add_argument(
        "--bin-dir",
        default=None,
        help="Override the user bin directory containing Praxist entrypoints.",
    )
    parser.add_argument(
        "--skills-dir",
        default=None,
        help="Override skill removal with one explicit managed directory.",
    )
    parser.add_argument(
        "--keep-user-data",
        action="store_true",
        help="Keep Praxist configuration, registry state, product-usage state, and cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate ownership and report removals without changing files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit one machine-readable result document.",
    )


def cmd_uninstall(args: argparse.Namespace) -> int:
    """Handle ``praxist uninstall``."""

    try:
        result = uninstall_installation(
            venv_dir=_optional_path(args.venv_dir),
            bin_dir=_optional_path(args.bin_dir),
            skills_dir=_optional_path(args.skills_dir),
            keep_user_data=bool(args.keep_user_data),
            dry_run=bool(args.dry_run),
        )
    except UninstallError as exc:
        print(f"praxist uninstall: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        verb = "would remove" if result["dry_run"] else "removed"
        print(
            f"Praxist uninstall: {verb} {len(result['removed'])} managed path(s).", file=sys.stderr
        )
        for item in result["preserved"]:
            print(f"  preserved: {item['path']} ({item['reason']})", file=sys.stderr)
        print("Research projects and run directories were not modified.", file=sys.stderr)
        print("Agent CLIs, Python, uv, and task dependencies were not modified.", file=sys.stderr)
    return 0


def uninstall_installation(
    *,
    venv_dir: Path | None = None,
    bin_dir: Path | None = None,
    skills_dir: Path | None = None,
    keep_user_data: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove one user-level Praxist installation.

    The operation fails closed while a local run is active or process state
    cannot be inspected. File removal is limited to Praxist application roots,
    installer-owned entrypoint symlinks, and a proven managed virtualenv.
    """

    _assert_no_active_runs()
    paths = _installation_paths(venv_dir=venv_dir, bin_dir=bin_dir)
    skill_targets = _skill_targets(skills_dir)
    skill_results: list[dict[str, Any]] = []
    for host, target_skills in skill_targets:
        try:
            skill_result = uninstall_codex_skills(target_dir=target_skills, dry_run=dry_run)
        except InstallSkillsError as exc:
            raise UninstallError(f"could not verify managed {host} skills: {exc}") from exc
        if skill_result["refused"]:
            refused = ", ".join(str(path) for path in skill_result["refused"])
            raise UninstallError(f"refusing to remove modified or unmanaged skill paths: {refused}")
        skill_results.append(skill_result)

    removed = [str(path) for result in skill_results for path in result["removed"]]
    missing: list[str] = [str(path) for result in skill_results for path in result["missing"]]
    preserved: list[dict[str, str]] = []

    for name in ENTRYPOINT_NAMES:
        _remove_entrypoint(
            paths["bin_dir"] / name,
            paths["venv_dir"] / _venv_scripts_dir() / name,
            dry_run=dry_run,
            removed=removed,
            missing=missing,
            preserved=preserved,
        )

    app_roots = [] if keep_user_data else _application_roots()
    for root in app_roots:
        _remove_application_root(
            root,
            dry_run=dry_run,
            removed=removed,
            missing=missing,
        )

    if not any(_is_within(paths["venv_dir"], root) for root in app_roots):
        _remove_managed_venv(
            paths["venv_dir"],
            default_venv=paths["default_venv"],
            dry_run=dry_run,
            removed=removed,
            missing=missing,
            preserved=preserved,
        )

    custom_config = os.environ.get("PRAXIST_CONFIG_FILE", "").strip()
    if custom_config:
        config_path = _absolute(Path(custom_config))
        if not any(_is_within(config_path, root) for root in app_roots):
            preserved.append(
                {
                    "path": str(config_path),
                    "reason": "custom PRAXIST_CONFIG_FILE is outside Praxist application roots",
                }
            )

    return {
        "dry_run": dry_run,
        "keep_user_data": keep_user_data,
        "removed": _dedupe_strings(removed),
        "missing": _dedupe_strings(missing),
        "preserved": _dedupe_preserved(preserved),
        "skills_dir": str(skill_targets[0][1]),
        "skills_dirs": [str(path) for _host, path in skill_targets],
        "venv_dir": str(paths["venv_dir"]),
        "bin_dir": str(paths["bin_dir"]),
    }


def _skill_targets(skills_dir: Path | None) -> list[tuple[str, Path]]:
    """Select explicit or manifest-backed skill hosts without another state file."""

    if skills_dir is not None:
        return [("custom", _absolute(skills_dir))]
    candidates = [(host, _absolute(default_skills_dir(host))) for host in ("codex", "claude")]
    managed = [
        (host, path) for host, path in candidates if (path / ".praxist-skills.json").is_file()
    ]
    return managed or [candidates[0]]


def _assert_no_active_runs() -> None:
    from praxist.cli import status

    errors: list[str] = []
    rows = status.collect_status_rows(
        errors=errors,
        include_peer_health=False,
        process_probe_timeout=3.0,
    )
    if errors:
        raise UninstallError(
            "could not verify that all local runs are stopped; run `praxist status` "
            "and resolve its process or registry warnings first"
        )
    active = [row for row in rows if row.source in {status.SOURCE_REGISTRY, status.SOURCE_PS_ONLY}]
    if active:
        labels = ", ".join(row.run_id or f"pid={row.pid}" for row in active[:5])
        suffix = "" if len(active) <= 5 else f" (+{len(active) - 5} more)"
        raise UninstallError(
            f"active Praxist run(s) detected: {labels}{suffix}; stop them before uninstalling"
        )


def _installation_paths(*, venv_dir: Path | None, bin_dir: Path | None) -> dict[str, Path]:
    home = Path.home()
    data_home = _absolute(Path(os.environ.get("XDG_DATA_HOME", home / ".local/share")))
    default_venv = data_home / "praxist" / "venv"
    selected_bin = bin_dir or Path(os.environ.get("XDG_BIN_HOME", home / ".local/bin"))
    return {
        "default_venv": _absolute(default_venv),
        "venv_dir": _absolute(venv_dir or default_venv),
        "bin_dir": _absolute(selected_bin),
    }


def _application_roots() -> list[Path]:
    home = Path.home()
    roots = [
        Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")) / "praxist",
        Path(os.environ.get("XDG_DATA_HOME", home / ".local/share")) / "praxist",
        Path(os.environ.get("XDG_STATE_HOME", home / ".local/state")) / "praxist",
        Path(os.environ.get("XDG_CACHE_HOME", home / ".cache")) / "praxist",
        home / ".config" / "praxist",
        home / ".local" / "share" / "praxist",
        home / ".local" / "state" / "praxist",
        home / ".cache" / "praxist",
    ]
    if sys.platform == "darwin":
        roots.append(home / "Library" / "Application Support" / "Praxist")
    elif os.name == "nt":  # pragma: no cover - managed-venv cleanup targets POSIX hosts.
        roots.append(home / "AppData" / "Local" / "Praxist")
    return _dedupe_paths(_absolute(path) for path in roots)


def _remove_entrypoint(
    path: Path,
    expected_target: Path,
    *,
    dry_run: bool,
    removed: list[str],
    missing: list[str],
    preserved: list[dict[str, str]],
) -> None:
    if not path.exists() and not path.is_symlink():
        missing.append(str(path))
        return
    if not path.is_symlink():
        preserved.append({"path": str(path), "reason": "entrypoint is not an installer symlink"})
        return
    if os.path.realpath(path) != os.path.realpath(expected_target):
        preserved.append({"path": str(path), "reason": "entrypoint points outside this install"})
        return
    removed.append(str(path))
    if not dry_run:
        try:
            path.unlink()
        except OSError as exc:
            raise UninstallError(f"could not remove entrypoint {path}: {exc}") from exc


def _remove_application_root(
    path: Path,
    *,
    dry_run: bool,
    removed: list[str],
    missing: list[str],
) -> None:
    _assert_application_root(path)
    _remove_path(path, dry_run=dry_run, removed=removed, missing=missing)


def _remove_managed_venv(
    path: Path,
    *,
    default_venv: Path,
    dry_run: bool,
    removed: list[str],
    missing: list[str],
    preserved: list[dict[str, str]],
) -> None:
    if not path.exists() and not path.is_symlink():
        missing.append(str(path))
        return
    if path.is_symlink():
        if not _is_managed_venv(path, default_venv=default_venv):
            preserved.append(
                {"path": str(path), "reason": "virtualenv link is not proven installer-managed"}
            )
            return
        removed.append(str(path))
        if not dry_run:
            try:
                path.unlink()
            except OSError as exc:
                raise UninstallError(f"could not remove virtualenv link {path}: {exc}") from exc
        preserved.append(
            {"path": str(path), "reason": "virtualenv symlink target was not followed"}
        )
        return
    if not _is_managed_venv(path, default_venv=default_venv):
        preserved.append(
            {"path": str(path), "reason": "virtualenv is not proven installer-managed"}
        )
        return
    _remove_path(path, dry_run=dry_run, removed=removed, missing=missing)


def _is_managed_venv(path: Path, *, default_venv: Path) -> bool:
    marker = path / MANAGED_VENV_MARKER
    try:
        if marker.is_file() and marker.read_text(encoding="utf-8") == MANAGED_VENV_MARKER_CONTENT:
            return True
    except OSError:
        return False
    scripts = path / _venv_scripts_dir()
    return (
        path == default_venv
        and (path / "pyvenv.cfg").is_file()
        and ((scripts / "praxist").exists() or (scripts / "praxist.exe").exists())
    )


def _remove_path(
    path: Path,
    *,
    dry_run: bool,
    removed: list[str],
    missing: list[str],
) -> None:
    if not path.exists() and not path.is_symlink():
        missing.append(str(path))
        return
    removed.append(str(path))
    if dry_run:
        return
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    except OSError as exc:
        raise UninstallError(f"could not remove {path}: {exc}") from exc


def _assert_application_root(path: Path) -> None:
    if path.name.lower() != "praxist" or path == Path.home() or path.parent == path:
        raise UninstallError(f"refusing unsafe application root: {path}")
    if len(path.parts) < 3:
        raise UninstallError(f"refusing shallow application root: {path}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _venv_scripts_dir() -> str:
    return "Scripts" if os.name == "nt" else "bin"


def _optional_path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _dedupe_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _dedupe_preserved(values: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, str]] = []
    for item in values:
        key = (item["path"], item["reason"])
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Run the standalone ``praxist-uninstall`` console entrypoint."""

    parser = argparse.ArgumentParser(
        prog="praxist-uninstall",
        description="Safely remove the current user's Praxist installation.",
    )
    _add_arguments(parser)
    raise SystemExit(cmd_uninstall(parser.parse_args(argv)))


if __name__ == "__main__":  # pragma: no cover
    main()
