"""Contracts for the versioned Praxist documentation system."""

from __future__ import annotations

import os
import re
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_docs_site

REPO_ROOT = Path(__file__).resolve().parents[2]
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


class DocumentationSiteContracts(unittest.TestCase):
    def test_generated_references_match_live_cli_and_skill_metadata(self) -> None:
        for path, expected in build_docs_site.generated_reference_docs().items():
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.read_text(encoding="utf-8"), expected)

        cli_reference = (REPO_ROOT / "docs" / "reference" / "cli.md").read_text(encoding="utf-8")
        self.assertNotIn("No additional description.", cli_reference)

    def test_cli_reference_is_independent_of_terminal_width(self) -> None:
        with patch.dict(os.environ, {"COLUMNS": "40"}):
            narrow = build_docs_site.render_cli_reference()
        with patch.dict(os.environ, {"COLUMNS": "220"}):
            wide = build_docs_site.render_cli_reference()
        self.assertEqual(narrow, wide)

    def test_navigation_owns_every_source_page_once_and_links_resolve(self) -> None:
        build_docs_site.validate_docs_sources()

        nav_paths = [path for _, path in build_docs_site.nav_entries()]
        source_paths = sorted(
            path.relative_to(REPO_ROOT / "docs").as_posix()
            for path in (REPO_ROOT / "docs").rglob("*.md")
        )
        self.assertEqual(sorted(nav_paths), source_paths)
        self.assertEqual(len(nav_paths), len(set(nav_paths)))

    def test_documentation_is_english_only(self) -> None:
        sources = [REPO_ROOT / "README.md", REPO_ROOT / "AGENTS.md"]
        for root_name in ("docs", "skills", "templates", "examples"):
            sources.extend((REPO_ROOT / root_name).rglob("*.md"))

        offenders = [
            path.relative_to(REPO_ROOT).as_posix()
            for path in sources
            if CJK_RE.search(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(offenders, [])
        self.assertFalse((REPO_ROOT / "README.zh-CN.md").exists())
        self.assertFalse((REPO_ROOT / "skills" / "README.zh-CN.md").exists())
        localized_pages = [
            path.relative_to(REPO_ROOT).as_posix()
            for root_name in ("docs", "skills", "templates", "examples")
            for path in (REPO_ROOT / root_name).rglob("*")
            if path.is_file()
            and any(
                marker in path.name.lower() for marker in (".zh.", ".zh-", "_zh.", "_zh-", "zh-cn")
            )
        ]
        self.assertEqual(localized_pages, [])

    def test_quickstart_owns_the_two_lane_oobe_contract(self) -> None:
        quickstart = (REPO_ROOT / "docs" / "getting-started" / "quickstart.md").read_text(
            encoding="utf-8"
        )
        local_heading = "## Local-Terminal Setup"
        agent_lane_heading = "## Agent-Managed Setup"
        codex_heading = "### Codex-Native Mode: No API Key"
        api_profiles_heading = "### API-Backed Profiles: Long Runs"
        self.assertLess(quickstart.index(local_heading), quickstart.index(agent_lane_heading))
        self.assertLess(quickstart.index(codex_heading), quickstart.index(api_profiles_heading))
        self.assertIn("praxist --takeover", quickstart)
        self.assertIn("praxist setup --interactive", quickstart)
        self.assertIn("praxist setup --agent-managed", quickstart)
        self.assertIn("[Installation](installation.md#install-and-configure)", quickstart)
        self.assertNotIn("pip install", quickstart)
        self.assertNotIn("praxist examples install", quickstart)
        self.assertIn("praxist --takeover --operator claude", quickstart)
        self.assertIn("docs/agents/oobe-install.md", quickstart)
        self.assertIn("one `*`", quickstart)
        self.assertIn("Nothing is preselected", quickstart)
        self.assertIn("not share an interaction controller", quickstart)
        self.assertIn(
            "Report current research progress and list the strongest variant",
            quickstart,
        )
        self.assertNotIn("codex login", quickstart)
        self.assertNotIn("<YOUR_DEEPSEEK_API_KEY>", quickstart)
        self.assertNotIn("read -rsp", quickstart)

    def test_homepage_moves_from_boundary_to_action_and_research_brief(self) -> None:
        homepage = (REPO_ROOT / "docs" / "index.md").read_text(encoding="utf-8")

        boundary = homepage.index("## Research Infrastructure, Explicit Science")
        installation = homepage.index("## Install Praxist")
        profiles = homepage.index("## Choose A Runtime Profile")
        next_steps = homepage.index("## Continue After Setup")
        prompt = homepage.index("## The Research Brief Is the Control Surface")
        self.assertLess(boundary, installation)
        self.assertLess(installation, profiles)
        self.assertLess(profiles, next_steps)
        self.assertLess(next_steps, prompt)
        self.assertIn(
            'python3 -m pip install --index-url https://pypi.org/simple "praxist[agents,codex]"',
            homepage,
        )
        self.assertIn("packaged OOBE runbook", homepage)
        self.assertIn("The task project owns", homepage)
        for role in ("Researcher", "Codex", "Claude Code", "Praxist", "Task project"):
            self.assertIn(role, homepage)
        self.assertIn("the sole source of domain meaning", homepage)
        self.assertNotIn('??? example "Detailed takeover brief"', homepage)

    def test_public_oobe_docs_use_the_pip_install_boundary(self) -> None:
        install_sources = (
            REPO_ROOT / "README.md",
            REPO_ROOT / "docs" / "index.md",
            REPO_ROOT / "docs" / "getting-started" / "installation.md",
            REPO_ROOT / "docs" / "agents" / "oobe-install.md",
            REPO_ROOT / "skills" / "README.md",
        )
        quickstart = REPO_ROOT / "docs" / "getting-started" / "quickstart.md"
        for path in (*install_sources, quickstart):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("bash praxist-install.sh", text, path)

        for path in install_sources:
            text = path.read_text(encoding="utf-8")
            self.assertIn(
                'pip install --index-url https://pypi.org/simple "praxist[agents,codex]"',
                text,
                path,
            )

        quickstart_text = quickstart.read_text(encoding="utf-8")
        self.assertNotIn("pip install", quickstart_text)
        self.assertIn("[Installation](installation.md#install-and-configure)", quickstart_text)

        installation = install_sources[2].read_text(encoding="utf-8")
        self.assertIn(
            "praxist setup --interactive --install-skills codex",
            installation,
        )
        self.assertIn(
            "praxist setup --interactive --install-skills claude",
            installation,
        )
        for path in install_sources:
            for line in path.read_text(encoding="utf-8").splitlines():
                if "pip install" in line and "praxist" in line:
                    self.assertNotIn("takeover", line, path)

    def test_terms_and_procedures_have_clear_documentation_owners(self) -> None:
        glossary = (REPO_ROOT / "docs" / "about" / "glossary.md").read_text(encoding="utf-8")
        for term in (
            "Principal Investigator (PI)",
            "Deep Innovation Gate (DIG)",
            "Quality-Diversity (QD)",
            "Herfindahl-Hirschman Index (HHI)",
        ):
            self.assertIn(term, glossary)

        qd = (REPO_ROOT / "docs" / "guides" / "qdig-cohort-allocator.md").read_text(
            encoding="utf-8"
        )
        qd_compact = " ".join(qd.split())
        self.assertIn("https://doi.org/10.3389/frobt.2016.00040", qd)
        self.assertIn(
            "Praxist applies that principle to candidate-plan allocation, "
            "not evolutionary genotype search.",
            qd_compact,
        )

        authored = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (REPO_ROOT / "docs").rglob("*.md")
            if path.name not in {"cli.md", "skills.md"}
        )
        for stale_phrase in (
            "DIG-Lite",
            "QD-DIG",
            "dig_lite",
            "the v0.1 migration is partial",
            "## Robust Mode",
            "three more generations have committed",
        ):
            self.assertNotIn(stale_phrase, authored)

        reports = (REPO_ROOT / "docs" / "guides" / "user-facing-reports-and-init.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("completed-generation count reaches a multiple of three", reports)

        credentials = (REPO_ROOT / "docs" / "guides" / "credentials.md").read_text(encoding="utf-8")
        self.assertIn("## Credential Failover Boundary", credentials)
        self.assertIn(
            "not a user-selectable runtime mode",
            " ".join(credentials.split()),
        )

        installation = (REPO_ROOT / "docs" / "getting-started" / "installation.md").read_text(
            encoding="utf-8"
        )
        examples = (REPO_ROOT / "docs" / "guides" / "examples-and-templates.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("praxist examples install", installation)
        self.assertIn("praxist examples install rocket_booster_recovery", examples)

        rocket = (REPO_ROOT / "docs" / "examples" / "rocket-booster-recovery.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("| `task_GPU_server` |", rocket)
        self.assertIn("| `task_PC` |", rocket)

    def test_first_task_explains_inputs_outputs_and_transparent_takeover(self) -> None:
        first_task = (REPO_ROOT / "docs" / "getting-started" / "first-task.md").read_text(
            encoding="utf-8"
        )

        for phrase in (
            "## What Must Exist Before Takeover",
            "## Write the Research Brief",
            '??? example "Detailed takeover brief"',
            "## What Praxist Adds",
            "Objective:",
            "Evidence:",
            "Execution:",
            "Exploration:",
            "Operation:",
            "## What Happens During Takeover",
            "observes the unchanged baseline execution path",
        ):
            self.assertIn(phrase, first_task)
        self.assertLess(
            first_task.index("## Start Takeover"),
            first_task.index("## What Happens During Takeover"),
        )

    def test_authored_docs_do_not_duplicate_long_paragraphs(self) -> None:
        excluded = {
            REPO_ROOT / "docs" / "legal" / "user-agreement.md",
            REPO_ROOT / "docs" / "legal" / "PRIVACY.md",
            REPO_ROOT / "docs" / "legal" / "product-usage-data-notice.md",
            REPO_ROOT / "docs" / "reference" / "cli.md",
            REPO_ROOT / "docs" / "reference" / "skills.md",
        }
        owners: dict[str, list[str]] = {}
        for path in sorted((REPO_ROOT / "docs").rglob("*.md")):
            if path in excluded:
                continue
            for paragraph in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
                normalized = " ".join(paragraph.split())
                if len(normalized) < 180 or normalized.startswith(("```", "|", "<div", "![")):
                    continue
                owners.setdefault(normalized, []).append(path.relative_to(REPO_ROOT).as_posix())

        duplicates = {text: paths for text, paths in owners.items() if len(set(paths)) > 1}
        self.assertEqual(duplicates, {})

    def test_key_documentation_contracts_have_one_owner(self) -> None:
        sources = {
            path.relative_to(REPO_ROOT).as_posix(): path.read_text(encoding="utf-8")
            for path in [REPO_ROOT / "README.md", *(REPO_ROOT / "docs").rglob("*.md")]
        }
        markers = {
            '??? example "Detailed takeover brief"': "docs/getting-started/first-task.md",
            "| Stage | Local interaction | Result |": "docs/getting-started/quickstart.md",
            "literature_search(query, sources, max_results)": (
                "docs/guides/scientific-literature-lookup.md"
            ),
            "## Automatic Reports": "docs/guides/user-facing-reports-and-init.md",
            "uncached_input_tokens = input_tokens": "docs/guides/costs.md",
            "PRAXIST_CONTEXT_EFFICIENCY_MODE": "docs/guides/cost-optimization.md",
        }
        for marker, expected_owner in markers.items():
            actual = [relative for relative, source in sources.items() if marker in source]
            self.assertEqual(actual, [expected_owner], marker)

    def test_readme_directs_users_to_hosted_documentation(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("## Documentation", readme)
        self.assertIn("praxist docs", readme)
        self.assertIn("No local documentation server is required.", readme)
        self.assertNotIn("## Browse the Documentation Site", readme)
        self.assertNotIn("uv run mkdocs serve", readme)

    def test_public_community_entry_points_are_present(self) -> None:
        required = (
            ".github/CODE_OF_CONDUCT.md",
            ".github/CONTRIBUTING.md",
            ".github/CODEOWNERS",
            ".github/PULL_REQUEST_TEMPLATE.md",
            ".github/ISSUE_TEMPLATE/bug_report.yml",
            ".github/ISSUE_TEMPLATE/feature_request.yml",
        )

        for relative in required:
            self.assertTrue((REPO_ROOT / relative).is_file(), relative)

        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        for relative in required[:2]:
            self.assertIn(f"({relative})", readme)

    def test_documentation_urls_follow_release_contracts(self) -> None:
        from praxist.cli.docs import DOCUMENTATION_URL  # noqa: PLC0415

        public_readme_url = "https://praxist.sapient.inc/en/docs"
        project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        config = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        workflow = (REPO_ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")

        self.assertEqual(project["project"]["urls"]["Documentation"], DOCUMENTATION_URL)
        self.assertEqual(DOCUMENTATION_URL, public_readme_url)
        self.assertIn(f"site_url: {DOCUMENTATION_URL}", config)
        self.assertIn("repo_url: https://github.com/sapientinc/praxist", config)
        self.assertIn("repo: fontawesome/brands/github", config)
        self.assertIn(public_readme_url, readme)
        self.assertIn("github.event_name == 'workflow_dispatch'", workflow)
        self.assertIn("vars.PRAXIST_PAGES_ENABLED == 'true'", workflow)

    def test_header_glass_effect_has_cross_platform_fallbacks(self) -> None:
        stylesheet = (REPO_ROOT / "docs" / "assets" / "stylesheets" / "extra.css").read_text(
            encoding="utf-8"
        )

        self.assertIn("-webkit-backdrop-filter: blur(12px) saturate(135%)", stylesheet)
        self.assertIn("backdrop-filter: blur(12px) saturate(135%)", stylesheet)
        self.assertIn("@media (prefers-reduced-transparency: reduce)", stylesheet)
        self.assertIn("@media (forced-colors: active)", stylesheet)
        self.assertIn("--praxist-shell-overscroll-reserve: 8rem", stylesheet)
        self.assertIn("position: sticky", stylesheet)
        self.assertIn(
            "height: calc(4.8rem + var(--praxist-shell-overscroll-reserve))",
            stylesheet,
        )
        self.assertIn(
            "transform: translateY(calc(-1 * var(--praxist-shell-overscroll-reserve)))",
            stylesheet,
        )
        self.assertIn("background-color: var(--praxist-shell-bg)", stylesheet)
        self.assertIn("body::before", stylesheet)

    def test_docs_dependencies_have_one_project_owned_definition(self) -> None:
        project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        docs_extra = project["project"]["optional-dependencies"]["docs"]
        self.assertIn("mkdocs-material>=9.7,<10", docs_extra)
        self.assertIn("mkdocstrings[python]>=1.0,<2", docs_extra)
        self.assertFalse((REPO_ROOT / "docs" / "requirements.txt").exists())

    def test_homepage_material_icon_shortcodes_have_a_renderer(self) -> None:
        homepage = (REPO_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
        config = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")

        self.assertIn(":material-rocket-launch:", homepage)
        self.assertIn("- pymdownx.emoji:", config)
        self.assertIn("material.extensions.emoji.twemoji", config)
        self.assertIn("material.extensions.emoji.to_svg", config)

    def test_documentation_header_uses_the_vector_logo(self) -> None:
        config = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        logo = REPO_ROOT / "docs" / "Praxist.svg"

        self.assertIn("logo: Praxist.svg", config)
        self.assertTrue(logo.is_file())
        self.assertIn("<svg", logo.read_text(encoding="utf-8"))

    def test_readme_uses_the_praxist_banner(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        banner = REPO_ROOT / "docs" / "assets" / "brand" / "praxist-banner.svg"

        self.assertIn('src="docs/assets/brand/praxist-banner.svg"', readme)
        self.assertTrue(banner.is_file())
        self.assertIn('viewBox="0 0 800 185"', banner.read_text(encoding="utf-8"))

    def test_documentation_favicon_uses_the_standalone_vector_mark(self) -> None:
        config = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        mark = REPO_ROOT / "docs" / "assets" / "brand" / "praxist-mark.svg"

        self.assertIn("favicon: assets/brand/praxist-mark.svg", config)
        self.assertTrue(mark.is_file())
        mark_text = mark.read_text(encoding="utf-8")
        self.assertIn('viewBox="0 0 203.33 203.33"', mark_text)
        self.assertIn('aria-label="Praxist mark"', mark_text)

    def test_material_diagrams_and_unbranded_footer_are_configured(self) -> None:
        config = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        docs_with_diagrams = (
            "docs/index.md",
            "docs/getting-started/first-task.md",
            "docs/guides/operators.md",
            "docs/concepts/architecture.md",
            "docs/guides/task-projects.md",
            "docs/guides/research-loop-variant-generation-flow.md",
            "docs/guides/user-facing-reports-and-init.md",
        )

        self.assertIn("generator: false", config)
        self.assertIn("name: mermaid", config)
        self.assertIn("pymdownx.superfences.fence_code_format", config)
        for relative in docs_with_diagrams:
            source = (REPO_ROOT / relative).read_text(encoding="utf-8")
            blocks = re.findall(r"```mermaid\n(.*?)\n```", source, flags=re.DOTALL)
            self.assertTrue(blocks, relative)
            self.assertEqual(source.count('class="praxist-diagram"'), len(blocks), relative)
            for block in blocks:
                self.assertIn("flowchart LR", block, relative)
                self.assertNotRegex(block, r"flowchart\s+(?:TD|TB|BT|RL)")

    def test_diagram_theme_is_semantic_and_adapts_to_both_color_schemes(self) -> None:
        stylesheet = (REPO_ROOT / "docs" / "assets" / "stylesheets" / "extra.css").read_text(
            encoding="utf-8"
        )

        for token in (
            "--md-mermaid-node-bg-color: #f4f6f8;",
            "--md-mermaid-node-bg-color: #2b313c;",
            "--md-mermaid-node-fg-color: #788493;",
            "--md-mermaid-node-fg-color: #8190a4;",
            "--md-mermaid-edge-color: #7d8794;",
            "--md-mermaid-edge-color: #8c97a7;",
            "overflow-x: auto;",
            "overscroll-behavior-inline: contain;",
            "min-width: 40rem;",
            ".node.system > :is(rect, circle, ellipse, polygon, path)",
            ".node.task > :is(rect, circle, ellipse, polygon, path)",
        ):
            self.assertIn(token, stylesheet)

    def test_operator_guide_is_direct_cli_first(self) -> None:
        operators = (REPO_ROOT / "docs" / "guides" / "operators.md").read_text(encoding="utf-8")

        self.assertTrue(operators.startswith("# Direct CLI Operations\n"))
        for phrase in (
            "## Command Map",
            "## Validate and Start",
            "## Observe a Run",
            "## Stop a Run",
            "## Resume a Run",
            "## Agent-Assisted Operation",
        ):
            self.assertIn(phrase, operators)

    def test_documentation_header_preserves_compact_glass_surfaces(self) -> None:
        stylesheet = (REPO_ROOT / "docs" / "assets" / "stylesheets" / "extra.css").read_text(
            encoding="utf-8"
        )

        for declaration in (
            "--praxist-shell-bg: #eceef1;",
            "--praxist-shell-bg: #2a3040;",
            "--praxist-shell-glass: rgb(236 239 243 / 42%);",
            "--praxist-shell-glass: rgb(35 42 56 / 48%);",
            "--praxist-search-bg: rgb(255 255 255 / 46%);",
            "--praxist-search-bg: rgb(12 17 27 / 38%);",
            "height: 1.5rem;",
        ):
            self.assertIn(declaration, stylesheet)
        self.assertNotIn("--praxist-header-logo-width", stylesheet)
        self.assertNotRegex(
            stylesheet,
            r"(?s)\.md-header__button\.md-logo\s*\{[^}]*position:\s*absolute",
        )

    def test_llm_exports_are_derived_from_navigation_order(self) -> None:
        with tempfile.TemporaryDirectory(prefix="praxist_docs_") as tmp_raw:
            output = Path(tmp_raw)
            with patch.object(build_docs_site, "BUILD_DIR", output):
                build_docs_site.generate_llm_exports()

            compact = (output / "llms.txt").read_text(encoding="utf-8")
            full = (output / "llms-full.txt").read_text(encoding="utf-8")
            self.assertEqual(compact, (output / "docs" / "llms.txt").read_text(encoding="utf-8"))
            self.assertEqual(
                full,
                (output / "docs" / "llms-full.txt").read_text(encoding="utf-8"),
            )
            self.assertIn("[Quickstart](getting-started/quickstart.html)", compact)
            for _, relative in build_docs_site.nav_entries():
                self.assertIn(f"<!-- Source: {relative} -->", full)

    def test_docs_workflow_builds_and_deploys_only_derived_output(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
        self.assertIn("uv sync --extra docs", workflow)
        self.assertIn("scripts/build_docs_site.py", workflow)
        self.assertIn('cache-dependency-glob: "pyproject.toml"', workflow)
        self.assertNotIn('cache-dependency-glob: "uv.lock"', workflow)
        self.assertIn("vars.PRAXIST_PAGES_ENABLED == 'true'", workflow)
        self.assertIn("actions/upload-pages-artifact@v3", workflow)
        self.assertIn("pages: write", workflow)
        self.assertIn("id-token: write", workflow)
        self.assertIn("actions/deploy-pages@v4", workflow)


if __name__ == "__main__":
    unittest.main()
