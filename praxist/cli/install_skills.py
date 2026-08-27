"""``praxist install-skills`` — register bundled Praxist skills for agent CLIs."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from praxist.cli._setup_common import (
    bundled_skill_dirs,
    default_codex_skills_dir,
    default_skills_dir,
    skill_tree_digest,
    write_skill_marker,
)

OWNERSHIP_MANIFEST = ".praxist-skills.json"
OWNERSHIP_SCHEMA_VERSION = 1


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register agent-host skill installation and removal subcommands."""
    parser = subparsers.add_parser(
        "install-skills",
        help="Install bundled Praxist skills for Codex or Claude Code.",
    )
    parser.add_argument(
        "--target",
        choices=("codex", "claude"),
        default="codex",
        help="Skill host. Default: codex.",
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="Override the target skill directory.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Register skills by copying package content or linking a source checkout.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Refresh existing Praxist-managed entries; unmanaged paths require --force-unmanaged."
        ),
    )
    parser.add_argument(
        "--force-unmanaged",
        action="store_true",
        help=(
            "With --replace, back up and replace unmanaged entries whose names exactly "
            "match bundled Praxist skills. Unrelated skills are untouched."
        ),
    )
    parser.add_argument(
        "--migrate-legacy-symlinks",
        action="store_true",
        help=(
            "With --replace, explicitly adopt old Praxist repo-style symlinks "
            "that predate the ownership manifest."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report actions without changing the target directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit the result as JSON.",
    )
    parser.set_defaults(func=cmd_install_skills)

    uninstall = subparsers.add_parser(
        "uninstall-skills",
        help="Remove Praxist-managed agent skill registrations.",
    )
    uninstall.add_argument(
        "--target",
        choices=("codex", "claude"),
        default="codex",
        help="Skill host. Default: codex.",
    )
    uninstall.add_argument(
        "--target-dir",
        default=None,
        help="Override the target skill directory.",
    )
    uninstall.add_argument(
        "--dry-run",
        action="store_true",
        help="Report removals without changing the target directory.",
    )
    uninstall.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit the result as JSON.",
    )
    uninstall.set_defaults(func=cmd_uninstall_skills)


def cmd_install_skills(args: argparse.Namespace) -> int:
    """Install bundled Praxist skills for the requested target."""
    try:
        result = install_skills(
            target=args.target,
            target_dir=Path(args.target_dir).expanduser() if args.target_dir else None,
            mode=args.mode,
            replace=args.replace,
            dry_run=args.dry_run,
            migrate_legacy_symlinks=args.migrate_legacy_symlinks,
            force_unmanaged=args.force_unmanaged,
        )
    except InstallSkillsError as exc:
        print(f"praxist install-skills: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"installed {len(result['installed'])} skill(s) into {result['target_dir']}",
            file=sys.stderr,
        )
        for entry in result["installed"]:
            print(f"  {entry['name']} ({entry['mode']})", file=sys.stderr)
        for backup in result.get("backups", []):
            print(f"  preserved previous skill at {backup}", file=sys.stderr)
    return 0


def cmd_uninstall_skills(args: argparse.Namespace) -> int:
    """Remove only skill entries that can be proven Praxist-managed."""

    try:
        result = uninstall_skills(
            target=args.target,
            target_dir=Path(args.target_dir).expanduser() if args.target_dir else None,
            dry_run=args.dry_run,
        )
    except InstallSkillsError as exc:
        print(f"praxist uninstall-skills: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"removed {len(result['removed'])} Praxist skill(s) from {result['target_dir']}",
            file=sys.stderr,
        )
        for path in result["refused"]:
            print(f"  refused unmanaged path: {path}", file=sys.stderr)
    return 1 if result["refused"] else 0


class InstallSkillsError(RuntimeError):
    """Raised when skill installation cannot proceed."""


class SkillConflictError(InstallSkillsError):
    """Raised when bundled names collide with operator-owned skill paths."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = tuple(paths)
        rendered = "\n  ".join(str(path) for path in self.paths)
        super().__init__(
            "existing skill paths are not Praxist-managed; refusing to overwrite:\n"
            f"  {rendered}\n"
            "rerun with --replace --force-unmanaged to preserve backups and replace them"
        )


def install_skills(
    *,
    target: str,
    target_dir: Path | None,
    mode: str,
    replace: bool,
    dry_run: bool,
    migrate_legacy_symlinks: bool = False,
    force_unmanaged: bool = False,
    skip_unmanaged: bool = False,
) -> dict[str, Any]:
    """Install bundled skills for one supported agent host."""

    if target not in {"codex", "claude"}:
        raise InstallSkillsError(f"unsupported skill host: {target}")
    resolved_target = target_dir or default_skills_dir(target)
    result = install_codex_skills(
        target_dir=resolved_target,
        mode=mode,
        replace=replace,
        dry_run=dry_run,
        migrate_legacy_symlinks=migrate_legacy_symlinks,
        force_unmanaged=force_unmanaged,
        skip_unmanaged=skip_unmanaged,
    )
    result["target"] = target
    return result


def uninstall_skills(
    *,
    target: str,
    target_dir: Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Remove Praxist-managed skills for one supported agent host."""

    if target not in {"codex", "claude"}:
        raise InstallSkillsError(f"unsupported skill host: {target}")
    resolved_target = target_dir or default_skills_dir(target)
    result = uninstall_codex_skills(target_dir=resolved_target, dry_run=dry_run)
    result["target"] = target
    return result


@contextlib.contextmanager
def _target_lock(target: Path) -> Iterator[None]:
    """Serialize Praxist skill lifecycle operations without another state file."""

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Praxist operators use POSIX hosts.
        raise InstallSkillsError("skill lifecycle locking requires a POSIX host") from exc
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(target, flags)
    except OSError as exc:
        raise InstallSkillsError(f"could not lock skill target directory {target}: {exc}") from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def install_codex_skills(
    *,
    target_dir: Path | None,
    mode: str,
    replace: bool,
    dry_run: bool,
    migrate_legacy_symlinks: bool = False,
    force_unmanaged: bool = False,
    skip_unmanaged: bool = False,
) -> dict[str, Any]:
    """Install bundled skills under one target-directory lifecycle lock."""

    target = (target_dir or default_codex_skills_dir()).resolve()
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        return _install_codex_skills_locked(
            target_dir=target,
            mode=mode,
            replace=replace,
            dry_run=dry_run,
            migrate_legacy_symlinks=migrate_legacy_symlinks,
            force_unmanaged=force_unmanaged,
            skip_unmanaged=skip_unmanaged,
        )
    with _target_lock(target):
        return _install_codex_skills_locked(
            target_dir=target,
            mode=mode,
            replace=replace,
            dry_run=dry_run,
            migrate_legacy_symlinks=migrate_legacy_symlinks,
            force_unmanaged=force_unmanaged,
            skip_unmanaged=skip_unmanaged,
        )


def _install_codex_skills_locked(
    *,
    target_dir: Path,
    mode: str,
    replace: bool,
    dry_run: bool,
    migrate_legacy_symlinks: bool,
    force_unmanaged: bool,
    skip_unmanaged: bool,
) -> dict[str, Any]:
    """Install bundled skills into one locked agent skills directory."""
    if mode not in {"copy", "symlink"}:
        raise InstallSkillsError(f"unsupported install mode: {mode}")
    if migrate_legacy_symlinks and not replace:
        raise InstallSkillsError("--migrate-legacy-symlinks requires --replace")
    if force_unmanaged and not replace:
        raise InstallSkillsError("--force-unmanaged requires --replace")
    if force_unmanaged and skip_unmanaged:
        raise InstallSkillsError("cannot both skip and force unmanaged skill paths")
    target = (target_dir or default_codex_skills_dir()).resolve()
    skills = bundled_skill_dirs()
    if not skills:
        raise InstallSkillsError("no bundled Praxist skills found")

    installed: list[dict[str, str]] = []
    conflicts: list[Path] = []
    skipped: list[str] = []
    skipped_names: set[str] = set()
    backups: list[str] = []
    actions: list[
        tuple[
            Path,
            Path,
            tuple[int, int, int, int, int] | None,
            dict[str, str] | None,
            bool,
            bool,
        ]
    ] = []
    stale_paths: list[
        tuple[
            Path,
            tuple[int, int, int, int, int] | None,
            Path | None,
            dict[str, str] | None,
        ]
    ] = []
    manifest_identity = _path_identity(target / OWNERSHIP_MANIFEST)
    ownership = _read_ownership_manifest(target)
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)

    current_names = {source.name for source in skills}
    for name, record in ownership.items():
        if name in current_names:
            continue
        dest = target / name
        if not (dest.exists() or dest.is_symlink()):
            stale_paths.append((dest, None, Path(record["source"]), record))
            continue
        if not _is_replaceable(
            dest,
            expected_source=Path(record["source"]),
            ownership=record,
            allow_legacy_symlink=False,
        ):
            raise InstallSkillsError(
                "removed bundled skill has an unverified destination; "
                f"refusing to modify it: {dest}"
            )
        stale_paths.append((dest, _path_identity(dest), Path(record["source"]), record))

    for source in skills:
        dest = target / source.name
        replace_unmanaged = False
        if dest.exists() or dest.is_symlink():
            replaceable = _is_replaceable(
                dest,
                expected_source=source,
                ownership=ownership.get(source.name),
                allow_legacy_symlink=replace and migrate_legacy_symlinks,
            )
            if not replaceable and not force_unmanaged:
                if skip_unmanaged:
                    skipped.append(str(dest))
                    skipped_names.add(source.name)
                    continue
                conflicts.append(dest)
                continue
            replace_unmanaged = not replaceable
            if not replace and _installation_is_current(dest, source=source, mode=mode):
                installed.append({"name": source.name, "path": str(dest), "mode": mode})
                continue
        actions.append(
            (
                source,
                dest,
                _path_identity(dest),
                ownership.get(source.name),
                replace and migrate_legacy_symlinks,
                replace_unmanaged,
            )
        )
        installed.append({"name": source.name, "path": str(dest), "mode": mode})

    if conflicts:
        raise SkillConflictError(conflicts)

    if not dry_run:
        for path, identity, expected_source, record in stale_paths:
            _remove_managed_skill(
                path,
                expected_identity=identity,
                expected_source=expected_source,
                ownership=record,
            )
        try:
            for source, dest, identity, record, allow_legacy, replace_unmanaged in actions:
                backup = _replace_managed_skill(
                    source=source,
                    dest=dest,
                    mode=mode,
                    expected_identity=identity,
                    ownership=record,
                    allow_legacy_symlink=allow_legacy,
                    replace_unmanaged=replace_unmanaged,
                )
                if backup is not None:
                    backups.append(str(backup))
            updated_ownership = {
                name: record
                for name, record in ownership.items()
                if name in current_names and name not in skipped_names
            }
            for source in skills:
                if source.name in skipped_names:
                    continue
                updated_ownership[source.name] = {
                    "managed_by": "praxist",
                    "mode": mode,
                    "source": str(source.resolve()),
                }
            _write_ownership_manifest(
                target,
                updated_ownership,
                expected_identity=manifest_identity,
            )
        except InstallSkillsError as exc:
            if backups:
                rendered = "\n  ".join(backups)
                raise InstallSkillsError(
                    f"{exc}\noperator-owned skills remain recoverable at:\n  {rendered}"
                ) from exc
            raise

    return {
        "target": "codex",
        "target_dir": str(target),
        "installed": installed,
        "removed_stale": [str(path) for path, *_rest in stale_paths],
        "backups": backups,
        "skipped": skipped,
    }


def uninstall_codex_skills(
    *,
    target_dir: Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Remove managed skills under one target-directory lifecycle lock."""

    target = (target_dir or default_codex_skills_dir()).resolve()
    if not target.exists():
        return _uninstall_codex_skills_locked(target_dir=target, dry_run=dry_run)
    with _target_lock(target):
        return _uninstall_codex_skills_locked(target_dir=target, dry_run=dry_run)


def _uninstall_codex_skills_locked(
    *,
    target_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Remove bundled skill names only when ownership can be proven."""

    target = (target_dir or default_codex_skills_dir()).resolve()
    skills = bundled_skill_dirs()
    removed: list[str] = []
    missing: list[str] = []
    refused: list[str] = []
    manifest_identity = _path_identity(target / OWNERSHIP_MANIFEST)
    ownership = _read_ownership_manifest(target)
    if not skills and not ownership:
        raise InstallSkillsError("no bundled or manifest-owned Praxist skills found")
    current_sources = {source.name: source for source in skills}
    for name in sorted(set(current_sources) | set(ownership)):
        source = current_sources.get(name)
        record = ownership.get(name)
        dest = target / name
        if not (dest.exists() or dest.is_symlink()):
            missing.append(str(dest))
            ownership.pop(name, None)
            continue
        if not _is_replaceable(
            dest,
            expected_source=source or (Path(record["source"]) if record else None),
            ownership=record,
            allow_legacy_symlink=False,
        ):
            refused.append(str(dest))
            continue
        removed.append(str(dest))
        if dry_run:
            continue
        _remove_managed_skill(
            dest,
            expected_identity=_path_identity(dest),
            expected_source=source or (Path(record["source"]) if record else None),
            ownership=record,
        )
        ownership.pop(name, None)
    if not dry_run:
        _write_ownership_manifest(
            target,
            ownership,
            expected_identity=manifest_identity,
        )
    return {
        "target": "codex",
        "target_dir": str(target),
        "removed": removed,
        "missing": missing,
        "refused": refused,
        "dry_run": dry_run,
    }


def _is_replaceable(
    path: Path,
    *,
    expected_source: Path | None = None,
    ownership: dict[str, str] | None = None,
    allow_legacy_symlink: bool = False,
    expected_name: str | None = None,
) -> bool:
    """Return True when an existing target is safe for Praxist to replace."""
    if path.is_symlink():
        if ownership and _symlink_matches_ownership(path, ownership):
            return True
        if expected_source is None:
            return False
        try:
            if path.resolve(strict=True) == expected_source.resolve(strict=True):
                return True
        except OSError:
            pass
        return allow_legacy_symlink and _looks_like_legacy_praxist_symlink(
            path,
            skill_name=expected_source.name,
        )
    marker = path / ".praxist-skill.json"
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    recorded_digest = data.get("tree_digest")
    try:
        current_digest = skill_tree_digest(path)
    except OSError:
        return False
    return (
        data.get("managed_by") == "praxist"
        and data.get("package", "praxist") == "praxist"
        and data.get("skill_name", expected_name or path.name) == (expected_name or path.name)
        and isinstance(recorded_digest, str)
        and bool(recorded_digest)
        and recorded_digest == current_digest
    )


def _read_ownership_manifest(target: Path) -> dict[str, dict[str, str]]:
    """Read and validate the target-level Praxist ownership manifest."""

    return _read_ownership_manifest_file(target / OWNERSHIP_MANIFEST)


def _read_ownership_manifest_file(path: Path) -> dict[str, dict[str, str]]:
    """Read one ownership manifest, including a private isolated copy."""

    if not path.exists() and not path.is_symlink():
        return {}
    if path.is_symlink() or not path.is_file():
        raise InstallSkillsError(
            f"skill ownership manifest is not a regular Praxist-managed file: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallSkillsError(
            f"skill ownership manifest is unreadable or invalid; refusing to replace it: {path}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != OWNERSHIP_SCHEMA_VERSION
        or payload.get("managed_by") != "praxist"
        or not isinstance(payload.get("skills"), dict)
    ):
        raise InstallSkillsError(
            f"skill ownership manifest is not owned by Praxist; refusing to replace it: {path}"
        )
    skills: dict[str, dict[str, str]] = {}
    for name, value in payload["skills"].items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
            or not isinstance(value, dict)
            or value.get("managed_by") != "praxist"
            or value.get("mode") not in {"copy", "symlink"}
            or not isinstance(value.get("source"), str)
            or not value["source"].strip()
        ):
            raise InstallSkillsError(
                "skill ownership manifest contains an invalid or unowned entry; "
                f"refusing to replace it: {path}"
            )
        skills[name] = dict(value)
    return skills


def _path_identity(path: Path) -> tuple[int, int, int, int, int] | None:
    """Return an inode/content identity without following symlinks."""

    try:
        stat = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise InstallSkillsError(f"could not inspect skill path {path}: {exc}") from exc
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_mode,
        stat.st_size,
        stat.st_mtime_ns,
    )


def _assert_managed_path_unchanged(
    path: Path,
    *,
    expected_identity: tuple[int, int, int, int, int] | None,
    expected_source: Path | None,
    ownership: dict[str, str] | None,
    allow_legacy_symlink: bool = False,
    expected_name: str | None = None,
) -> None:
    """Refuse a destination changed after ownership validation."""

    current_identity = _path_identity(path)
    if current_identity != expected_identity:
        raise InstallSkillsError(
            f"skill destination changed during lifecycle operation; refusing to modify: {path}"
        )
    if current_identity is not None and not _is_replaceable(
        path,
        expected_source=expected_source,
        ownership=ownership,
        allow_legacy_symlink=allow_legacy_symlink,
        expected_name=expected_name,
    ):
        raise InstallSkillsError(
            f"skill destination is no longer Praxist-managed; refusing to modify: {path}"
        )


def _remove_managed_skill(
    path: Path,
    *,
    expected_identity: tuple[int, int, int, int, int] | None,
    expected_source: Path | None,
    ownership: dict[str, str] | None,
) -> None:
    """Quarantine and remove one destination only after post-move validation."""

    try:
        quarantine = _quarantine_managed_path(
            path,
            expected_identity=expected_identity,
            expected_source=expected_source,
            ownership=ownership,
        )
        if quarantine is not None:
            _remove_path(quarantine)
    except OSError as exc:
        raise InstallSkillsError(f"could not remove managed skill {path}: {exc}") from exc


def _write_ownership_manifest(
    target: Path,
    skills: dict[str, dict[str, str]],
    *,
    expected_identity: tuple[int, int, int, int, int] | None,
) -> None:
    """Persist or remove the manifest without overwriting concurrent content."""

    path = target / OWNERSHIP_MANIFEST
    stage = target / f".{OWNERSHIP_MANIFEST}.{uuid.uuid4().hex}.tmp"
    backup: Path | None = None
    try:
        if skills:
            payload = {
                "schema_version": OWNERSHIP_SCHEMA_VERSION,
                "managed_by": "praxist",
                "skills": skills,
            }
            stage.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.chmod(stage, 0o600)
        backup = _quarantine_ownership_manifest(
            path,
            expected_identity=expected_identity,
        )
        if skills:
            os.link(stage, path, follow_symlinks=False)
        if backup is not None:
            _remove_path(backup)
    except BaseException as exc:
        if backup is not None and (backup.exists() or backup.is_symlink()):
            try:
                _restore_quarantined_path(backup, path)
            except InstallSkillsError as restore_exc:
                raise InstallSkillsError(f"{exc}; {restore_exc}") from exc
        if isinstance(exc, InstallSkillsError):
            raise
        raise InstallSkillsError(f"could not update skill ownership manifest: {exc}") from exc
    finally:
        stage.unlink(missing_ok=True)


def _quarantine_ownership_manifest(
    path: Path,
    *,
    expected_identity: tuple[int, int, int, int, int] | None,
) -> Path | None:
    """Isolate and revalidate a Praxist-owned manifest before replacing it."""

    if _path_identity(path) != expected_identity:
        raise InstallSkillsError(
            "skill ownership manifest changed during lifecycle operation; "
            f"refusing to replace it: {path}"
        )
    if expected_identity is None:
        return None
    backup = path.parent / f".{path.name}.praxist-backup-{uuid.uuid4().hex}"
    try:
        os.replace(path, backup)
        if _path_identity(backup) != expected_identity:
            raise InstallSkillsError(
                "skill ownership manifest changed during lifecycle operation; "
                f"refusing to replace it: {path}"
            )
        _read_ownership_manifest_file(backup)
    except BaseException as exc:
        if backup.exists() or backup.is_symlink():
            try:
                _restore_quarantined_path(backup, path)
            except InstallSkillsError as restore_exc:
                raise InstallSkillsError(f"{exc}; {restore_exc}") from exc
        raise
    return backup


def _looks_like_legacy_praxist_symlink(path: Path, *, skill_name: str) -> bool:
    """Recognize only the narrow symlink shape emitted by older repo scripts."""

    try:
        raw_target = Path(os.readlink(path))
    except OSError:
        return False
    if raw_target.name != skill_name or raw_target.parent.name != "skills":
        return False
    skill_file = path / "SKILL.md"
    try:
        text = skill_file.read_text(encoding="utf-8")
    except OSError:
        # A moved checkout leaves a broken absolute link. The lexical
        # ``.../skills/<known-name>`` shape is the only recoverable proof.
        return True
    return any(line.strip() == f"name: {skill_name}" for line in text.splitlines()[:20])


def _symlink_matches_ownership(path: Path, ownership: dict[str, str]) -> bool:
    """Verify that a manifest-owned symlink still points to its recorded source."""

    if ownership.get("managed_by") != "praxist":
        return False
    recorded = str(ownership.get("source") or "").strip()
    if not recorded:
        return False
    try:
        raw_target = Path(os.readlink(path)).expanduser()
    except OSError:
        return False
    if not raw_target.is_absolute():
        raw_target = path.parent / raw_target
    return raw_target.resolve(strict=False) == Path(recorded).expanduser().resolve(strict=False)


def _installation_is_current(path: Path, *, source: Path, mode: str) -> bool:
    if mode == "symlink":
        return path.is_symlink() and _is_replaceable(path, expected_source=source)
    if path.is_symlink() or not path.is_dir():
        return False
    marker = path / ".praxist-skill.json"
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        installed_digest = skill_tree_digest(path)
        source_digest = skill_tree_digest(source)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        data.get("managed_by") == "praxist"
        and data.get("skill_name", path.name) == path.name
        and data.get("tree_digest") == installed_digest == source_digest
        and (path / "SKILL.md").is_file()
    )


def _replace_managed_skill(
    *,
    source: Path,
    dest: Path,
    mode: str,
    expected_identity: tuple[int, int, int, int, int] | None,
    ownership: dict[str, str] | None,
    allow_legacy_symlink: bool,
    replace_unmanaged: bool = False,
) -> Path | None:
    """Stage one skill and atomically replace its validated destination."""

    stage = dest.parent / f".{dest.name}.praxist-stage-{uuid.uuid4().hex}"
    backup = dest.parent / f".{dest.name}.praxist-backup-{uuid.uuid4().hex}"
    try:
        if mode == "symlink":
            os.symlink(source.resolve(), stage, target_is_directory=True)
        else:
            shutil.copytree(source, stage)
            write_skill_marker(
                stage,
                source="package-resource",
                skill_name=dest.name,
            )
        had_existing = expected_identity is not None
        if had_existing:
            if replace_unmanaged:
                backup = (
                    _quarantine_unmanaged_path(
                        dest,
                        expected_identity=expected_identity,
                        quarantine=backup,
                    )
                    or backup
                )
            else:
                backup = (
                    _quarantine_managed_path(
                        dest,
                        expected_identity=expected_identity,
                        expected_source=source,
                        ownership=ownership,
                        allow_legacy_symlink=allow_legacy_symlink,
                        quarantine=backup,
                    )
                    or backup
                )
        else:
            _assert_managed_path_unchanged(
                dest,
                expected_identity=None,
                expected_source=source,
                ownership=ownership,
                allow_legacy_symlink=allow_legacy_symlink,
            )
        try:
            _publish_staged_skill(stage, dest)
            if mode == "copy" and not _is_replaceable(dest, expected_name=dest.name):
                raise InstallSkillsError(
                    f"installed skill changed during publication; refusing ownership: {dest}"
                )
        except BaseException:
            if had_existing and (backup.exists() or backup.is_symlink()):
                _restore_quarantined_path(backup, dest)
            raise
        if had_existing and not replace_unmanaged and (backup.exists() or backup.is_symlink()):
            _remove_path(backup)
    except OSError as exc:
        raise InstallSkillsError(f"could not install {source.name}: {exc}") from exc
    finally:
        with contextlib.suppress(OSError):
            _remove_path(stage)
    return backup if replace_unmanaged and (backup.exists() or backup.is_symlink()) else None


def _quarantine_unmanaged_path(
    path: Path,
    *,
    expected_identity: tuple[int, int, int, int, int] | None,
    quarantine: Path,
) -> Path | None:
    """Atomically isolate an explicitly forced same-name skill destination."""

    if expected_identity is None:
        return None
    if _path_identity(path) != expected_identity:
        raise InstallSkillsError(f"skill destination changed before forced replacement: {path}")
    try:
        os.replace(path, quarantine)
        if _path_identity(quarantine) != expected_identity:
            raise InstallSkillsError(f"skill destination changed during forced replacement: {path}")
    except BaseException as exc:
        if quarantine.exists() or quarantine.is_symlink():
            try:
                _restore_quarantined_path(quarantine, path)
            except InstallSkillsError as restore_exc:
                raise InstallSkillsError(f"{exc}; {restore_exc}") from exc
        raise
    return quarantine


def _quarantine_managed_path(
    path: Path,
    *,
    expected_identity: tuple[int, int, int, int, int] | None,
    expected_source: Path | None,
    ownership: dict[str, str] | None,
    allow_legacy_symlink: bool = False,
    quarantine: Path | None = None,
) -> Path | None:
    """Atomically isolate a public destination, then verify what was moved."""

    _assert_managed_path_unchanged(
        path,
        expected_identity=expected_identity,
        expected_source=expected_source,
        ownership=ownership,
        allow_legacy_symlink=allow_legacy_symlink,
    )
    if expected_identity is None:
        return None
    isolated = quarantine or (path.parent / f".{path.name}.praxist-quarantine-{uuid.uuid4().hex}")
    try:
        os.replace(path, isolated)
        _assert_managed_path_unchanged(
            isolated,
            expected_identity=expected_identity,
            expected_source=expected_source,
            ownership=ownership,
            allow_legacy_symlink=allow_legacy_symlink,
            expected_name=path.name,
        )
    except BaseException as exc:
        if isolated.exists() or isolated.is_symlink():
            try:
                _restore_quarantined_path(isolated, path)
            except InstallSkillsError as restore_exc:
                raise InstallSkillsError(f"{exc}; {restore_exc}") from exc
        raise
    return isolated


def _publish_staged_skill(stage: Path, dest: Path) -> None:
    """Publish a complete stage without overwriting a concurrently created path."""

    if stage.is_symlink():
        os.symlink(os.readlink(stage), dest, target_is_directory=True)
        return
    if stage.is_dir():
        _copy_tree_exclusive(stage, dest)
        return
    raise InstallSkillsError(f"invalid staged skill path: {stage}")


def _restore_quarantined_path(quarantine: Path, dest: Path) -> None:
    """Restore a preserved path without overwriting a newly created destination."""

    try:
        if quarantine.is_symlink():
            os.symlink(os.readlink(quarantine), dest, target_is_directory=True)
        elif quarantine.is_dir():
            _copy_tree_exclusive(quarantine, dest)
        elif quarantine.is_file():
            with quarantine.open("rb") as source, dest.open("xb") as target:
                shutil.copyfileobj(source, target)
        else:
            raise InstallSkillsError(f"unsupported quarantined skill path: {quarantine}")
    except (OSError, InstallSkillsError) as exc:
        raise InstallSkillsError(
            f"could not restore {dest}; preserved original content at {quarantine}: {exc}"
        ) from exc
    _remove_path(quarantine)


def _copy_tree_exclusive(source: Path, dest: Path) -> None:
    """Copy a private complete tree without overwriting concurrent public files."""

    marker_name = ".praxist-skill.json"

    def ignore_marker(directory: str, names: list[str]) -> set[str]:
        if Path(directory) == source and marker_name in names:
            return {marker_name}
        return set()

    shutil.copytree(
        source,
        dest,
        symlinks=True,
        copy_function=_copy_file_exclusive,
        ignore=ignore_marker,
    )
    marker = source / marker_name
    if marker.is_file():
        _copy_file_exclusive(marker, dest / marker_name)
    shutil.copystat(source, dest, follow_symlinks=False)


def _copy_file_exclusive(source: str | Path, dest: str | Path) -> str:
    """Copy one regular file using exclusive destination creation."""

    source_path = Path(source)
    dest_path = Path(dest)
    with source_path.open("rb") as source_file, dest_path.open("xb") as dest_file:
        shutil.copyfileobj(source_file, dest_file)
    shutil.copystat(source_path, dest_path, follow_symlinks=False)
    return str(dest_path)


def _remove_path(path: Path) -> None:
    """Remove one known private path without following symlinks."""

    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)
