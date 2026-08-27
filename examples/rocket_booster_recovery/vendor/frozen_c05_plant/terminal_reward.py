"""Pod gen24_pod11 DIG C06 — Jiang speed-knee terminal reward (no lateral bonus).

Mechanism (reward_shaping / ppo_config, intent=repair, semantic family =
terminal reward speed-knee alignment):

    Parent lineage:
      * 5c3d7481 (gen22_pod10) realigned the terminal-reward TILT knee to the
        strict-Jiang 0.05 rad gate and added a tilt-pass-shaped bonus. That
        repaired the rs=0.15 learned tilt-pass collapse but sacrificed the
        learned speed-pass gain because the reward SPEED knee stayed at the
        standard 4.0 m/s.
      * 5cb79771 / 86c8aea3 (gen23_pod9) added the speed knee (2.0 m/s) and
        speed-pass-shaped bonus; the no-lateral arm (lateral bonus DISABLED)
        recovered tilt-pass while preserving speed-pass (strict-Jiang 0.387
        held-out at rs=0.15).

    This pod (C06) uses the SAME no-lateral reward (tilt knee 0.05 + tilt bonus,
    speed knee 2.0 + speed bonus, lateral bonus OFF) and composes it with the
    coast-drag g3 passive terminal fin-drag lever. The knobs below shape the
    PPO TRAINING reward signal only. They do NOT change the evaluator standard
    safe gate (landing_tilt=0.10), the strict-Jiang co-gate (tilt<0.05),
    metrics, data splits, or disk radius.

Environment overrides (ablation hooks, no code change):
    SWORDFISH_REWARD_TILT_KNEE_RAD       -> terminal reward tilt knee (0.05 treat / 0.10 revert)
    SWORDFISH_REWARD_TILT_BONUS_ON       -> tilt-pass-shaped bonus master switch
    SWORDFISH_REWARD_TILT_BONUS_SCALE    -> tilt bonus magnitude at tilt=0
    SWORDFISH_REWARD_SPEED_KNEE_MPS      -> terminal reward speed knee (2.0 treat / 4.0 revert / 3.0 ablation)
    SWORDFISH_REWARD_SPEED_BONUS_ON      -> speed-pass-shaped bonus master switch
    SWORDFISH_REWARD_SPEED_BONUS_SCALE   -> speed bonus magnitude at speed=0
    SWORDFISH_REWARD_LATERAL_BONUS_ON    -> lateral-capture bonus master switch (C06 default OFF)
    SWORDFISH_REWARD_LATERAL_BONUS_SCALE -> lateral bonus magnitude at lateral=0
    SWORDFISH_REWARD_LATERAL_FLOOR_M     -> lateral regularization floor (m)
"""

from __future__ import annotations

import os

# Strict-Jiang evaluator gate constants (source of truth: evaluations/swordfish_eval/run.py).
JIANG_TILT_GATE_RAD = 0.05
JIANG_SPEED_GATE_MPS = 2.0
JIANG_LATERAL_GATE_M = 3.0
JIANG_OMEGA_GATE_RADPS = 0.10

# Standard landing gates (evaluator phase_corrected_safe_rate; do not change).
STANDARD_TILT_KNEE_RAD = 0.10
STANDARD_SPEED_GATE_MPS = 4.0

# Original harness terminal reward constant.
SAFE_BONUS = 250.0

# C06 treat defaults: dual tilt+speed alignment (knees at Jiang gates), NO
# lateral bonus (the harmful lateral-capture bonus from 5cb79771 is removed).
DEFAULTS = {
    "reward_tilt_knee_rad": JIANG_TILT_GATE_RAD,        # 0.05 (treat); revert 0.10 for knee-control
    "reward_tilt_bonus_on": 1.0,                        # tilt-pass-shaped bonus active
    "reward_tilt_bonus_scale": 50.0,                    # tilt bonus magnitude at tilt=0
    "reward_speed_knee_mps": JIANG_SPEED_GATE_MPS,      # 2.0 (treat); revert 4.0 for knee-control
    "reward_speed_bonus_on": 1.0,                       # speed-pass-shaped bonus active
    "reward_speed_bonus_scale": 50.0,                   # speed bonus magnitude at speed=0
    "reward_lateral_bonus_on": 0.0,                     # C06: lateral-capture bonus DISABLED
    "reward_lateral_bonus_scale": 50.0,                 # lateral bonus magnitude at lateral=0 (only if on)
    "reward_lateral_floor_m": JIANG_LATERAL_GATE_M,     # inverse-proportional floor (m)
}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def resolve_reward_cfg() -> dict:
    """Resolve the reward-shaping knobs from environment variables."""
    cfg = dict(DEFAULTS)
    cfg["reward_tilt_knee_rad"] = _env_float("SWORDFISH_REWARD_TILT_KNEE_RAD", cfg["reward_tilt_knee_rad"])
    cfg["reward_tilt_bonus_on"] = _env_float("SWORDFISH_REWARD_TILT_BONUS_ON", cfg["reward_tilt_bonus_on"])
    cfg["reward_tilt_bonus_scale"] = _env_float("SWORDFISH_REWARD_TILT_BONUS_SCALE", cfg["reward_tilt_bonus_scale"])
    cfg["reward_speed_knee_mps"] = _env_float("SWORDFISH_REWARD_SPEED_KNEE_MPS", cfg["reward_speed_knee_mps"])
    cfg["reward_speed_bonus_on"] = _env_float("SWORDFISH_REWARD_SPEED_BONUS_ON", cfg["reward_speed_bonus_on"])
    cfg["reward_speed_bonus_scale"] = _env_float("SWORDFISH_REWARD_SPEED_BONUS_SCALE", cfg["reward_speed_bonus_scale"])
    cfg["reward_lateral_bonus_on"] = _env_float("SWORDFISH_REWARD_LATERAL_BONUS_ON", cfg["reward_lateral_bonus_on"])
    cfg["reward_lateral_bonus_scale"] = _env_float("SWORDFISH_REWARD_LATERAL_BONUS_SCALE", cfg["reward_lateral_bonus_scale"])
    cfg["reward_lateral_floor_m"] = _env_float("SWORDFISH_REWARD_LATERAL_FLOOR_M", cfg["reward_lateral_floor_m"])
    return cfg


def describe(cfg: dict) -> str:
    return (
        "tilt_knee=%.3f rad (tilt_bonus=%s scale %.1f) | "
        "speed_knee=%.2f m/s (speed_bonus=%s scale %.1f) | "
        "lateral_bonus=%s scale %.1f floor %.2f m"
        % (
            cfg["reward_tilt_knee_rad"],
            "ON" if cfg["reward_tilt_bonus_on"] > 0.5 else "OFF",
            cfg["reward_tilt_bonus_scale"],
            cfg["reward_speed_knee_mps"],
            "ON" if cfg["reward_speed_bonus_on"] > 0.5 else "OFF",
            cfg["reward_speed_bonus_scale"],
            "ON" if cfg["reward_lateral_bonus_on"] > 0.5 else "OFF",
            cfg["reward_lateral_bonus_scale"],
            cfg["reward_lateral_floor_m"],
        )
    )
