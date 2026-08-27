from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class RepositoryCleanupTest(unittest.TestCase):
    def test_only_canonical_product_identity_is_present(self) -> None:
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO_ROOT).split(b"\0")
        tracked = [item for item in tracked if item]
        forbidden_literals = (
            b"sapi" + b"ent",
            b"ai" + b"_scientist",
            b"ai" + b"-scientist",
            b"ai" + b" scientist",
        )
        allowed_identity_literals = (
            b"https://praxist." + b"sapi" + b"ent.inc/en/docs",
            b"https://discord.gg/" + b"sapi" + b"ent",
            b"https://github.com/" + b"sapi" + b"entinc/praxist",
            b"ghcr.io/" + b"sapi" + b"entinc/praxist-collector",
            b"praxist@" + b"sapi" + b"ent.inc",
            b"praxist by " + b"sapi" + b"ent intelligence",
            b"sapi" + b"ent intelligence pte ltd",
        )
        legal_identity_paths = {
            b"LICENSE.md",
            b"docs/legal/PRIVACY.md",
            b"docs/legal/user-agreement.md",
        }
        short_identity = b"a" + b"is"
        short_pattern = re.compile(
            rb"(?<![A-Za-z0-9])" + re.escape(short_identity) + rb"(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        offenders: list[str] = []

        for relative_bytes in tracked:
            if relative_bytes in legal_identity_paths:
                continue
            relative = relative_bytes.decode("utf-8")
            path = REPO_ROOT / relative
            content = path.read_bytes()
            lowered_path = relative_bytes.lower()
            lowered_content = content.lower()
            for allowed_literal in allowed_identity_literals:
                lowered_content = lowered_content.replace(allowed_literal, b"")
            if any(
                literal in lowered_path or literal in lowered_content
                for literal in forbidden_literals
            ) or short_pattern.search(relative_bytes):
                offenders.append(relative)
                continue
            if short_pattern.search(lowered_content):
                offenders.append(relative)

        self.assertEqual(offenders, [])

    def test_legacy_archives_and_operator_scripts_are_not_bundled(self) -> None:
        forbidden_paths = [
            "auto_research",
            "old",
            "experiments/MCI-SAM-deliverables",
            "experiments/MCI-SAM-deliverablesi",
            "kill_auto_research.sh",
            "kill_run.sh",
            "analyze_auto_research_bugs.py",
            "check_auto_research_progress.py",
        ]

        existing = [path for path in forbidden_paths if (REPO_ROOT / path).exists()]

        self.assertEqual(existing, [])

    def test_root_kill_scripts_are_not_bundled(self) -> None:
        kill_scripts = sorted(path.name for path in REPO_ROOT.glob("kill*.sh"))

        self.assertEqual(kill_scripts, [])


if __name__ == "__main__":
    unittest.main()
