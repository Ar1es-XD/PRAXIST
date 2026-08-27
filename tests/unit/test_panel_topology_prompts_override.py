"""Tests for the panel-topology ``prompts_dir`` override hook.

The PR that introduced these tests adds an optional ``prompts_dir`` field on
``PanelTopologySpec`` and threads it through ``BasePI`` and ``ChairArbiter``
so panel-topology plugins can ship their own Jinja templates without forking
the framework's bundled ``multi_pi/prompts/`` directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from praxist.core.panel_topology import (
    PanelTopologySpec,
    panel_topology_from_manifest,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.chair_arbiter import (
    ChairArbiter,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles._base_pi import (
    BasePI,
    _default_prompts_dir,
)

# ---------------------------------------------------------------------------
# PanelTopologySpec / manifest resolution
# ---------------------------------------------------------------------------


def _minimal_topology(prompts_dir_value: object | None = None) -> dict[str, object]:
    topology: dict[str, object] = {
        "modes": {"mini": ["foo"]},
        "rounds": [{"round_id": "r1"}],
        "roles": [{"role_id": "foo", "role_ref": "task_role:foo"}],
    }
    if prompts_dir_value is not None:
        topology["prompts_dir"] = prompts_dir_value
    return {"topology": topology}


def test_panel_topology_spec_default_prompts_dir_is_none() -> None:
    spec = PanelTopologySpec(topology_ref="panel_topology:test", modes={}, rounds=[])
    assert spec.prompts_dir is None
    assert spec.to_dict()["prompts_dir"] is None


def test_from_manifest_without_prompts_dir_yields_none() -> None:
    spec = panel_topology_from_manifest("panel_topology:test", _minimal_topology())
    assert spec.prompts_dir is None


def test_from_manifest_relative_prompts_dir_resolves(tmp_path: Path) -> None:
    prompts_subdir = tmp_path / "prompts"
    prompts_subdir.mkdir()
    spec = panel_topology_from_manifest(
        "panel_topology:test",
        _minimal_topology("prompts"),
        plugin_path=tmp_path,
    )
    assert spec.prompts_dir == prompts_subdir.resolve()
    assert spec.to_dict()["prompts_dir"] == str(prompts_subdir.resolve())


def test_from_manifest_relative_prompts_dir_without_plugin_path_raises() -> None:
    with pytest.raises(ValueError, match="plugin_path is unknown"):
        panel_topology_from_manifest("panel_topology:test", _minimal_topology("prompts"))


def test_from_manifest_missing_prompts_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        panel_topology_from_manifest(
            "panel_topology:test",
            _minimal_topology("prompts_does_not_exist"),
            plugin_path=tmp_path,
        )


def test_from_manifest_absolute_prompts_dir_accepted(tmp_path: Path) -> None:
    custom = tmp_path / "custom_prompts"
    custom.mkdir()
    spec = panel_topology_from_manifest(
        "panel_topology:test",
        _minimal_topology(str(custom)),
    )
    assert spec.prompts_dir == custom


def test_from_manifest_absolute_prompts_dir_missing_raises(tmp_path: Path) -> None:
    nonexistent = tmp_path / "absent"
    with pytest.raises(ValueError, match="does not exist"):
        panel_topology_from_manifest(
            "panel_topology:test",
            _minimal_topology(str(nonexistent)),
        )


def test_from_manifest_non_string_prompts_dir_raises() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        panel_topology_from_manifest("panel_topology:test", _minimal_topology(42))


# ---------------------------------------------------------------------------
# BasePI render_prompt search path
# ---------------------------------------------------------------------------


class _StubPI(BasePI):
    """Minimal concrete PI used to exercise ``render_prompt`` template lookup."""

    role_name = "stub"
    prompt_template_name = "base.jinja2"
    private_kb_dir_name = "stub"

    def fixed_questions(self) -> list[str]:
        return []


def _minimal_render(pi: BasePI) -> str:
    return pi.render_prompt(
        shared_core={},
        private_pack=[],
        private_kb_entries=[],
        target_decisions=[],
        fixed_questions=[],
    )


def test_default_prompts_dir_helper_returns_bundled_dir() -> None:
    bundled = _default_prompts_dir()
    assert bundled.is_dir()
    assert (bundled / "base.jinja2").is_file()


def test_base_pi_without_prompts_dir_uses_bundled_template(tmp_path: Path) -> None:
    """No prompts_dir override → bundled ``base.jinja2`` is rendered."""
    pi = _StubPI(run_dir=tmp_path, workspace=tmp_path, model="dummy")
    assert pi.prompts_dir is None
    out = _minimal_render(pi)
    # The bundled template uppercases role_name in its heading.
    assert "STUB PI" in out


def test_base_pi_with_custom_prompts_dir_uses_override(tmp_path: Path) -> None:
    """A plugin-supplied template wins over the bundled one."""
    custom = tmp_path / "prompts"
    custom.mkdir()
    (custom / "base.jinja2").write_text("CUSTOM-MARKER role={{ role_name }}")
    pi = _StubPI(run_dir=tmp_path, workspace=tmp_path, model="dummy", prompts_dir=custom)
    out = _minimal_render(pi)
    assert "CUSTOM-MARKER" in out
    assert "role=stub" in out


def test_base_pi_partial_override_falls_back_to_bundled(tmp_path: Path) -> None:
    """A custom directory missing ``base.jinja2`` falls back to bundled."""
    custom = tmp_path / "prompts"
    custom.mkdir()
    # Only override round2_cross_review, leaving base.jinja2 to fall back.
    (custom / "round2_cross_review.jinja2").write_text("ROUND2-OVERRIDE")
    pi = _StubPI(run_dir=tmp_path, workspace=tmp_path, model="dummy", prompts_dir=custom)
    out = _minimal_render(pi)
    # Bundled base.jinja2 is recognizable by its uppercase heading.
    assert "STUB PI" in out


# ---------------------------------------------------------------------------
# ChairArbiter render_prompt search path
# ---------------------------------------------------------------------------


def _chair_render(chair: ChairArbiter) -> str:
    return chair.render_prompt(
        shared_core_digest={},
        pi_memos={},
        cross_reviews={},
        confidence_revisions={},
        next_gen_id=1,
        completed_gen_id=0,
        panel_mode="mini",
        shared_core_id="test-id",
    )


def test_chair_arbiter_with_custom_prompts_dir_uses_override(tmp_path: Path) -> None:
    custom = tmp_path / "prompts"
    custom.mkdir()
    (custom / "chair.jinja2").write_text(
        "CHAIR-OVERRIDE peer_budget={{ peer_budget }} mode={{ panel_mode }}"
    )
    chair = ChairArbiter(
        run_dir=tmp_path,
        workspace=tmp_path,
        model="dummy",
        peer_budget=7,
        prompts_dir=custom,
    )
    out = _chair_render(chair)
    assert "CHAIR-OVERRIDE" in out
    assert "peer_budget=7" in out
    assert "mode=mini" in out


def test_chair_arbiter_without_prompts_dir_uses_bundled_template(tmp_path: Path) -> None:
    chair = ChairArbiter(
        run_dir=tmp_path,
        workspace=tmp_path,
        model="dummy",
    )
    assert chair.prompts_dir is None
    out = _chair_render(chair)
    # The bundled chair.jinja2 contains the literal token "peer_contracts" in
    # its schema instructions — a cheap signal that we hit the real template.
    assert "peer_contracts" in out


# ---------------------------------------------------------------------------
# peer_role_rotation override (issue #84)
# ---------------------------------------------------------------------------


def test_panel_topology_spec_default_peer_role_rotation_is_empty() -> None:
    """Default rotation is an empty tuple — the Chair falls back to the

    bundled ``exploit / falsifier / bridge / anti_mainline`` set.
    Topologies that don't opt in see no change in behavior.
    """
    spec = PanelTopologySpec(topology_ref="panel_topology:test", modes={}, rounds=[])
    assert spec.peer_role_rotation == ()
    assert spec.to_dict()["peer_role_rotation"] == []


def test_from_manifest_parses_peer_role_rotation() -> None:
    topology = _minimal_topology()
    topology["topology"]["peer_role_rotation"] = ["specialist_a", "specialist_b"]
    spec = panel_topology_from_manifest("panel_topology:test", topology)
    assert spec.peer_role_rotation == ("specialist_a", "specialist_b")


def test_from_manifest_peer_role_rotation_non_list_raises() -> None:
    topology = _minimal_topology()
    topology["topology"]["peer_role_rotation"] = "exploit"
    with pytest.raises(ValueError, match="peer_role_rotation must be a list"):
        panel_topology_from_manifest("panel_topology:test", topology)


def test_from_manifest_peer_role_rotation_blank_entry_raises() -> None:
    topology = _minimal_topology()
    topology["topology"]["peer_role_rotation"] = ["valid", ""]
    with pytest.raises(ValueError, match="non-empty strings"):
        panel_topology_from_manifest("panel_topology:test", topology)


def test_chair_fallback_agenda_uses_custom_peer_role_rotation() -> None:
    """Issue #84: custom panel topologies can supply their own peer-role

    vocabulary so the deterministic fallback agenda emits role names the
    task project has role files for, instead of the bundled five-role set.
    """
    from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
        chair_arbiter,
    )

    fallback = chair_arbiter._build_deterministic_fallback_agenda(
        pi_memos={},
        cross_reviews={},
        next_gen_id=2,
        completed_gen_id=1,
        panel_mode="mini",
        shared_core_id="core",
        peer_budget=4,
        parse_error="x",
        peer_role_rotation=("specialist_a", "specialist_b"),
    )
    peer_contracts = fallback["peer_contracts"]
    roles = [contract["role"] for contract in peer_contracts.values()]
    # Rotation is applied modulo length, so 4 peers over a 2-role rotation
    # alternate a-b-a-b. None of the bundled role names should appear.
    assert roles == ["specialist_a", "specialist_b", "specialist_a", "specialist_b"]
    for role in roles:
        assert role not in ("exploit", "falsifier", "bridge", "anti_mainline")
    # Unknown role names fall through to the generic ``else`` branch which
    # emits ``fallback_hypotheses`` as the source marker.
    sources = {contract["source"] for contract in peer_contracts.values()}
    assert sources == {"fallback_hypotheses"}
    assert fallback["success_metrics"]["required"] == [
        "at_least_1_contract_for_each_topology_role",
        "at_least_3_cross_peer_hypotheses_or_actions_tested",
        "at_least_1_non_mainline_or_falsification_attempt_completed",
    ]


def test_chair_fallback_agenda_default_rotation_unchanged() -> None:
    """Topologies that don't supply a rotation keep the bundled vocabulary."""
    from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
        chair_arbiter,
    )

    fallback = chair_arbiter._build_deterministic_fallback_agenda(
        pi_memos={},
        cross_reviews={},
        next_gen_id=2,
        completed_gen_id=1,
        panel_mode="mini",
        shared_core_id="core",
        peer_budget=4,
        parse_error="x",
    )
    roles = [contract["role"] for contract in fallback["peer_contracts"].values()]
    assert roles == list(chair_arbiter._PEER_ROLE_ROTATION[:4])
    assert fallback["success_metrics"]["required"] == [
        "at_least_3_cross_peer_hypotheses_tested",
        "at_least_1_bridge_experiment_completed",
        "at_least_1_dissent_resolution_peer",
        "at_least_1_anti_mainline_result_or_negative_finding",
    ]


def test_chair_arbiter_stores_peer_role_rotation(tmp_path: Path) -> None:
    """The Chair stashes the rotation so its two fallback call sites both

    receive the same custom vocabulary.
    """
    chair = ChairArbiter(
        run_dir=tmp_path,
        workspace=tmp_path,
        model="dummy",
        peer_role_rotation=("alpha", "beta"),
    )
    assert chair.peer_role_rotation == ("alpha", "beta")


def test_chair_prompt_schema_uses_custom_peer_role_rotation(tmp_path: Path) -> None:
    chair = ChairArbiter(
        run_dir=tmp_path,
        workspace=tmp_path,
        model="dummy",
        peer_budget=4,
        peer_role_rotation=("specialist_a", "specialist_b"),
    )

    rendered = chair.render_prompt(
        shared_core_digest={},
        pi_memos={},
        cross_reviews={},
        confidence_revisions={},
        next_gen_id=2,
        completed_gen_id=1,
        panel_mode="full",
        shared_core_id="core",
    )

    assert "role: specialist_a" in rendered
    assert "role: specialist_b" in rendered
    assert 'parent_usage: "compare"' in rendered
    assert "at_least_1_contract_for_each_topology_role" in rendered


# ---------------------------------------------------------------------------
# peer_role_descriptions override (issue #85)
# ---------------------------------------------------------------------------


def test_panel_topology_spec_default_peer_role_descriptions_is_empty() -> None:
    """Without an override, ``peer_role_descriptions`` is an empty dict and

    the peer prompt template renders the bundled five-bullet vocabulary.
    """
    spec = PanelTopologySpec(topology_ref="panel_topology:test", modes={}, rounds=[])
    assert spec.peer_role_descriptions == {}
    assert spec.to_dict()["peer_role_descriptions"] == {}


def test_from_manifest_parses_peer_role_descriptions() -> None:
    topology = _minimal_topology()
    topology["topology"]["peer_role_descriptions"] = {
        "specialist_a": "lead the ablation arm",
        "specialist_b": "challenge the leading mechanism",
    }
    spec = panel_topology_from_manifest("panel_topology:test", topology)
    assert spec.peer_role_descriptions == {
        "specialist_a": "lead the ablation arm",
        "specialist_b": "challenge the leading mechanism",
    }


def test_from_manifest_peer_role_descriptions_non_mapping_raises() -> None:
    topology = _minimal_topology()
    topology["topology"]["peer_role_descriptions"] = ["a", "b"]
    with pytest.raises(ValueError, match="must be a mapping"):
        panel_topology_from_manifest("panel_topology:test", topology)


def test_from_manifest_peer_role_descriptions_blank_value_raises() -> None:
    topology = _minimal_topology()
    topology["topology"]["peer_role_descriptions"] = {"specialist_a": ""}
    with pytest.raises(ValueError, match="must be a non-empty string"):
        panel_topology_from_manifest("panel_topology:test", topology)


def test_from_manifest_peer_role_descriptions_blank_key_raises() -> None:
    topology = _minimal_topology()
    topology["topology"]["peer_role_descriptions"] = {"   ": "desc"}
    with pytest.raises(ValueError, match="keys must be non-empty strings"):
        panel_topology_from_manifest("panel_topology:test", topology)
