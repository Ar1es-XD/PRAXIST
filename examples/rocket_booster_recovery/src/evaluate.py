#!/usr/bin/env python3
"""Batched deterministic evaluation of Rocket Booster Recovery on frozen OOD initial states."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .metrics import (
    INITIAL_FUEL_KG,
    INITIAL_MASS_KG,
    MASS_EMPTY_KG,
    SUCCESS_THRESHOLDS,
    first_contact_arrays,
    summarize,
)
from .plant_adapter import file_sha256, load_frozen_plant, provenance
from .rocket_booster_recovery_controller import control_step, init_memory, load_config


ROOT = Path(__file__).resolve().parents[1]
AUDIT_COLUMNS = (
    "active_steps",
    "max_forbidden_action_abs",
    "max_plant_lateral_rcs_nm",
    "rcs_roll_abs_impulse_nms",
    "rcs_roll_signed_impulse_nms",
    "gimbal_total_variation_rad",
    "grid_total_variation_rad",
    "throttle_total_variation",
    "emergency_steps",
    "phase0_steps",
    "phase1_steps",
    "phase2_steps",
    "phase3_steps",
    "gimbal_saturation_steps",
    "grid_saturation_steps",
    "throttle_saturation_steps",
    "rcs_roll_cap_steps",
    "max_body_axis_error_rad",
    "min_brake_margin_m",
    "max_dynamic_pressure_pa",
    "nonfinite_seen",
)


class RolloutResult(NamedTuple):
    terminal_state: jax.Array
    terminal_previous_omega: jax.Array
    done: jax.Array
    max_abs_action: jax.Array
    audit: jax.Array
    first_contact_state: jax.Array
    first_contact_detected: jax.Array
    first_contact_step: jax.Array
    first_contact_leg_sink_speed_mps: jax.Array


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _select_tree(proposed, previous, active: jax.Array):
    return jax.tree.map(
        lambda x, y: jnp.where(active.reshape((active.shape[0],) + (1,) * (x.ndim - 1)), x, y),
        proposed,
        previous,
    )


def build_rollout(plant, plant_cfg, controller_cfg, batch_size: int):
    init_batch = jax.vmap(lambda s: init_memory(s, controller_cfg))
    control_batch = jax.vmap(lambda s, m: control_step(s, m, controller_cfg))
    key = jax.random.PRNGKey(0)
    plant_batch = jax.vmap(lambda s, a: plant.step_one(key, s, a, plant_cfg))
    dt = float(controller_cfg["dt"])
    n_legs = int(plant_cfg.gear_n_legs)
    phases = 2.0 * math.pi * jnp.arange(n_legs) / float(n_legs)
    tips_body = jnp.column_stack(
        (
            jnp.full(n_legs, -float(plant_cfg.gear_contact_height_m)),
            float(plant_cfg.gear_footprint_radius_m) * jnp.cos(phases),
            float(plant_cfg.gear_footprint_radius_m) * jnp.sin(phases),
        )
    )

    def tip_metrics_one(state: jax.Array) -> tuple[jax.Array, jax.Array]:
        rot = plant.quat_to_rot(state[6:10])
        offsets_i = (rot @ tips_body.T).T
        heights = state[0] + offsets_i[:, 0]
        omega_i = rot @ state[10:13]
        tip_velocities = state[3:6][None, :] + jnp.cross(
            jnp.broadcast_to(omega_i, offsets_i.shape), offsets_i
        )
        min_height = jnp.min(heights)
        lowest = heights <= min_height + 1.0e-3
        leg_sink = jnp.max(
            jnp.where(
                lowest,
                jnp.maximum(-tip_velocities[:, 0], 0.0),
                0.0,
            )
        )
        return min_height, leg_sink

    tip_metrics_batch = jax.vmap(tip_metrics_one)

    def interpolate_state(
        state: jax.Array, proposed_state: jax.Array, alpha: jax.Array
    ) -> jax.Array:
        value = state + alpha[:, None] * (proposed_state - state)
        quat = value[:, 6:10]
        quat = quat / jnp.maximum(
            jnp.linalg.norm(quat, axis=1, keepdims=True), 1e-8
        )
        return value.at[:, 6:10].set(quat)

    @jax.jit
    def rollout(initial_state: jax.Array) -> RolloutResult:
        if initial_state.shape != (batch_size, 16):
            raise ValueError(f"compiled batch expects {(batch_size, 16)}, got {initial_state.shape}")
        memory = init_batch(initial_state)
        n = initial_state.shape[0]
        done = jnp.zeros(n, dtype=bool)
        previous_omega = initial_state[:, 10:13]
        active_steps = jnp.zeros(n, dtype=jnp.int32)
        max_abs_action = jnp.zeros((n, 9), dtype=initial_state.dtype)
        max_forbidden = jnp.zeros(n, dtype=initial_state.dtype)
        max_plant_lateral_rcs = jnp.zeros(n, dtype=initial_state.dtype)
        rcs_abs_impulse = jnp.zeros(n, dtype=initial_state.dtype)
        rcs_signed_impulse = jnp.zeros(n, dtype=initial_state.dtype)
        gimbal_tv = jnp.zeros(n, dtype=initial_state.dtype)
        grid_tv = jnp.zeros(n, dtype=initial_state.dtype)
        throttle_tv = jnp.zeros(n, dtype=initial_state.dtype)
        emergency_steps = jnp.zeros(n, dtype=jnp.int32)
        phase_steps = jnp.zeros((n, 4), dtype=jnp.int32)
        gimbal_sat = jnp.zeros(n, dtype=jnp.int32)
        grid_sat = jnp.zeros(n, dtype=jnp.int32)
        throttle_sat = jnp.zeros(n, dtype=jnp.int32)
        rcs_cap = jnp.zeros(n, dtype=jnp.int32)
        max_body_error = jnp.zeros(n, dtype=initial_state.dtype)
        min_brake_margin = jnp.full(n, jnp.inf, dtype=initial_state.dtype)
        max_qdyn = jnp.zeros(n, dtype=initial_state.dtype)
        nonfinite_seen = jnp.zeros(n, dtype=bool)
        first_contact_state = jnp.zeros_like(initial_state)
        first_contact_detected = jnp.zeros(n, dtype=bool)
        first_contact_step = jnp.zeros(n, dtype=jnp.int32)
        first_contact_leg_sink = jnp.zeros(n, dtype=initial_state.dtype)

        carry = (
            initial_state, memory, done, previous_omega, active_steps,
            max_abs_action, max_forbidden, max_plant_lateral_rcs,
            rcs_abs_impulse, rcs_signed_impulse, gimbal_tv, grid_tv,
            throttle_tv, emergency_steps, phase_steps, gimbal_sat, grid_sat,
            throttle_sat, rcs_cap, max_body_error, min_brake_margin, max_qdyn,
            nonfinite_seen, first_contact_state, first_contact_detected,
            first_contact_step, first_contact_leg_sink,
        )

        def body(carry, unused):
            del unused
            (
                state, memory, done, terminal_prev_w, active_steps,
                max_abs_action, max_forbidden, max_plant_lateral_rcs,
                rcs_abs_impulse, rcs_signed_impulse, gimbal_tv, grid_tv,
                throttle_tv, emergency_steps, phase_steps, gimbal_sat, grid_sat,
                throttle_sat, rcs_cap, max_body_error, min_brake_margin, max_qdyn,
                nonfinite_seen, first_contact_state, first_contact_detected,
                first_contact_step, first_contact_leg_sink,
            ) = carry
            active = ~done
            action, proposed_memory, diagnostic = control_batch(state, memory)
            # The plant call is vectorized over all rows for a static GPU graph;
            # outputs from rows already terminated are discarded below.
            _, proposed_state, _, proposed_done, info = plant_batch(state, action)
            finite = jnp.all(jnp.isfinite(proposed_state), axis=1)

            height_now, _ = tip_metrics_batch(state)
            height_next, _ = tip_metrics_batch(proposed_state)
            crossing = (
                active
                & finite
                & (~first_contact_detected)
                & (height_next <= 0.0)
            )

            lo = jnp.zeros(n, dtype=state.dtype)
            hi = jnp.ones(n, dtype=state.dtype)

            def bisect(_, bracket):
                low, high = bracket
                mid = 0.5 * (low + high)
                mid_state = interpolate_state(state, proposed_state, mid)
                mid_height, _ = tip_metrics_batch(mid_state)
                high = jnp.where(mid_height <= 0.0, mid, high)
                low = jnp.where(mid_height > 0.0, mid, low)
                return low, high

            lo, hi = jax.lax.fori_loop(0, 10, bisect, (lo, hi))
            linear_alpha = jnp.clip(
                height_now / jnp.maximum(height_now - height_next, 1e-8),
                0.0,
                1.0,
            )
            alpha = jnp.where(height_now > 0.0, hi, linear_alpha)
            contact_state = interpolate_state(state, proposed_state, alpha)
            _, contact_leg_sink = tip_metrics_batch(contact_state)

            first_contact_state = jnp.where(
                crossing[:, None], contact_state, first_contact_state
            )
            first_contact_detected = first_contact_detected | crossing
            step_number = active_steps + active.astype(jnp.int32)
            first_contact_step = jnp.where(
                crossing, step_number, first_contact_step
            )
            first_contact_leg_sink = jnp.where(
                crossing, contact_leg_sink, first_contact_leg_sink
            )

            plant_terminal = active & ((proposed_done > 0.5) | (~finite))
            newly_done = crossing | plant_terminal
            terminal_prev_w = jnp.where(
                newly_done[:, None], state[:, 10:13], terminal_prev_w
            )
            scored_state = jnp.where(
                crossing[:, None], contact_state, proposed_state
            )
            next_state = jnp.where(active[:, None], scored_state, state)
            next_done = done | newly_done
            next_memory = _select_tree(proposed_memory, memory, active)

            active_f = active.astype(state.dtype)
            active_steps = step_number
            max_abs_action = jnp.maximum(max_abs_action, jnp.abs(action) * active_f[:, None])
            forbidden = jnp.max(jnp.abs(action[:, jnp.array([4, 5, 8])]), axis=1)
            max_forbidden = jnp.maximum(max_forbidden, forbidden * active_f)
            max_plant_lateral_rcs = jnp.maximum(max_plant_lateral_rcs, jnp.abs(info[:, 15]) * active_f)
            rcs_nm = diagnostic[:, 16]
            rcs_abs_impulse = rcs_abs_impulse + jnp.abs(rcs_nm) * dt * active_f
            rcs_signed_impulse = rcs_signed_impulse + rcs_nm * dt * active_f
            gimbal_now = diagnostic[:, 11:13]
            grid_now = diagnostic[:, 13:15]
            throttle_now = diagnostic[:, 15]
            gimbal_tv = gimbal_tv + jnp.sum(jnp.abs(gimbal_now - memory.prev_gimbal_rad), axis=1) * active_f
            grid_tv = grid_tv + jnp.sum(jnp.abs(grid_now - memory.prev_grid_rad), axis=1) * active_f
            throttle_tv = throttle_tv + jnp.abs(throttle_now - memory.prev_throttle) * active_f
            emergency_steps = emergency_steps + ((diagnostic[:, 1] > 0.5) & active).astype(jnp.int32)
            phase_index = jnp.clip(diagnostic[:, 0].astype(jnp.int32), 0, 3)
            phase_steps = phase_steps + jax.nn.one_hot(phase_index, 4, dtype=jnp.int32) * active[:, None]
            gimbal_sat = gimbal_sat + (
                (jnp.max(jnp.abs(gimbal_now), axis=1) >= float(controller_cfg["gimbal_normal_rad"]) - 1e-6) & active
            ).astype(jnp.int32)
            grid_sat = grid_sat + (
                (jnp.max(jnp.abs(grid_now), axis=1) >= float(controller_cfg["grid_normal_rad"]) - 1e-6) & active
            ).astype(jnp.int32)
            throttle_sat = throttle_sat + (
                ((throttle_now <= float(controller_cfg["throttle_normal_min"]) + 1e-6)
                 | (throttle_now >= float(controller_cfg["throttle_normal_max"]) - 1e-6)) & active
            ).astype(jnp.int32)
            cap = jnp.where(
                diagnostic[:, 1] > 0.5,
                float(controller_cfg["rcs_roll_emergency_cap_nm"]),
                float(controller_cfg["rcs_roll_normal_cap_nm"]),
            )
            rcs_cap = rcs_cap + ((jnp.abs(rcs_nm) >= cap - 1.0) & active).astype(jnp.int32)
            max_body_error = jnp.maximum(max_body_error, diagnostic[:, 17] * active_f)
            min_brake_margin = jnp.where(active, jnp.minimum(min_brake_margin, diagnostic[:, 3]), min_brake_margin)
            max_qdyn = jnp.maximum(max_qdyn, diagnostic[:, 4] * active_f)
            nonfinite_seen = nonfinite_seen | (active & (~finite))

            return (
                next_state, next_memory, next_done, terminal_prev_w, active_steps,
                max_abs_action, max_forbidden, max_plant_lateral_rcs,
                rcs_abs_impulse, rcs_signed_impulse, gimbal_tv, grid_tv,
                throttle_tv, emergency_steps, phase_steps, gimbal_sat, grid_sat,
                throttle_sat, rcs_cap, max_body_error, min_brake_margin, max_qdyn,
                nonfinite_seen, first_contact_state, first_contact_detected,
                first_contact_step, first_contact_leg_sink,
            ), None

        final, _ = jax.lax.scan(body, carry, xs=None, length=int(plant_cfg.max_steps))
        (
            state, _, done, terminal_prev_w, active_steps,
            max_abs_action, max_forbidden, max_plant_lateral_rcs,
            rcs_abs_impulse, rcs_signed_impulse, gimbal_tv, grid_tv,
            throttle_tv, emergency_steps, phase_steps, gimbal_sat, grid_sat,
            throttle_sat, rcs_cap, max_body_error, min_brake_margin, max_qdyn,
            nonfinite_seen, first_contact_state, first_contact_detected,
            first_contact_step, first_contact_leg_sink,
        ) = final
        endpoint_state = jnp.where(
            first_contact_detected[:, None], first_contact_state, state
        )
        audit = jnp.column_stack(
            [
                active_steps,
                max_forbidden,
                max_plant_lateral_rcs,
                rcs_abs_impulse,
                rcs_signed_impulse,
                gimbal_tv,
                grid_tv,
                throttle_tv,
                emergency_steps,
                phase_steps,
                gimbal_sat,
                grid_sat,
                throttle_sat,
                rcs_cap,
                max_body_error,
                min_brake_margin,
                max_qdyn,
                nonfinite_seen,
            ]
        )
        return RolloutResult(
            terminal_state=endpoint_state,
            terminal_previous_omega=terminal_prev_w,
            done=done,
            max_abs_action=max_abs_action,
            audit=audit,
            first_contact_state=first_contact_state,
            first_contact_detected=first_contact_detected,
            first_contact_step=first_contact_step,
            first_contact_leg_sink_speed_mps=first_contact_leg_sink,
        )

    return rollout


def _audit_summary(audit: np.ndarray, max_action: np.ndarray) -> dict:
    idx = {name: i for i, name in enumerate(AUDIT_COLUMNS)}
    return {
        "columns": list(AUDIT_COLUMNS),
        "max_abs_normalized_action_by_channel": np.max(max_action, axis=0).tolist(),
        "max_forbidden_action_abs": float(np.max(audit[:, idx["max_forbidden_action_abs"]])),
        "max_plant_lateral_rcs_nm": float(np.max(audit[:, idx["max_plant_lateral_rcs_nm"]])),
        "rcs_roll_abs_impulse_nms": {
            "mean": float(np.mean(audit[:, idx["rcs_roll_abs_impulse_nms"]])),
            "max": float(np.max(audit[:, idx["rcs_roll_abs_impulse_nms"]])),
        },
        "rcs_roll_signed_impulse_nms": {
            "mean": float(np.mean(audit[:, idx["rcs_roll_signed_impulse_nms"]])),
            "min": float(np.min(audit[:, idx["rcs_roll_signed_impulse_nms"]])),
            "max": float(np.max(audit[:, idx["rcs_roll_signed_impulse_nms"]])),
        },
        "nonfinite_trajectories": int(np.count_nonzero(audit[:, idx["nonfinite_seen"]] > 0.5)),
        "active_steps": {
            "mean": float(np.mean(audit[:, idx["active_steps"]])),
            "min": int(np.min(audit[:, idx["active_steps"]])),
            "max": int(np.max(audit[:, idx["active_steps"]])),
        },
        "saturation_step_totals": {
            name: int(np.sum(audit[:, idx[name]]))
            for name in ("gimbal_saturation_steps", "grid_saturation_steps", "throttle_saturation_steps", "rcs_roll_cap_steps")
        },
        "phase_step_totals": {
            f"P{k}": int(np.sum(audit[:, idx[f"phase{k}_steps"]])) for k in range(4)
        },
        "emergency_step_total": int(np.sum(audit[:, idx["emergency_steps"]])),
        "max_body_axis_error_rad": float(np.max(audit[:, idx["max_body_axis_error_rad"]])),
        "minimum_brake_margin_m": float(np.min(audit[:, idx["min_brake_margin_m"]])),
        "maximum_dynamic_pressure_pa": float(np.max(audit[:, idx["max_dynamic_pressure_pa"]])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rocket_booster_recovery_v0.json")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all remaining rows")
    parser.add_argument("--integrator", choices=("rk4", "rk2_full"), default="rk4")
    parser.add_argument("--substeps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    if abs(float(cfg.get("mass_empty_kg", -1.0)) - MASS_EMPTY_KG) > 1e-9:
        raise SystemExit("controller mass_empty_kg must remain 22200 kg")
    if abs(float(cfg.get("initial_fuel_kg", -1.0)) - INITIAL_FUEL_KG) > 1e-9:
        raise SystemExit("controller initial_fuel_kg must remain 7000 kg")
    plant, plant_cfg = load_frozen_plant(args.integrator, args.substeps)
    plant_cfg = plant_cfg._replace(mass_full=INITIAL_MASS_KG, fuel_scale=1.0)
    with np.load(args.dataset, allow_pickle=False) as bank:
        all_state = np.asarray(bank["state"], dtype=np.float32)
        all_source = np.asarray(bank["source_id"], dtype=np.int8) if "source_id" in bank.files else np.zeros(len(all_state), np.int8)
        source_names = np.asarray(bank["source_names"]).astype(str) if "source_names" in bank.files else np.asarray(["all"])
        all_source_row = np.asarray(bank["source_row"], dtype=np.int32) if "source_row" in bank.files else np.arange(len(all_state), dtype=np.int32)
    all_state = all_state.copy()
    all_state[:, 13] = np.float32(INITIAL_MASS_KG)
    end = len(all_state) if args.limit <= 0 else min(len(all_state), args.offset + args.limit)
    state = all_state[args.offset:end]
    source = all_source[args.offset:end]
    source_row = all_source_row[args.offset:end]
    if len(state) == 0:
        raise SystemExit("empty evaluation slice")

    rollout = build_rollout(plant, plant_cfg, cfg, args.batch_size)
    outputs = []
    compile_and_run_times = []
    total_start = time.perf_counter()
    for start in range(0, len(state), args.batch_size):
        valid = min(args.batch_size, len(state) - start)
        block = state[start:start + valid]
        if valid < args.batch_size:
            pad = np.repeat(block[-1:], args.batch_size - valid, axis=0)
            block = np.concatenate([block, pad], axis=0)
        tic = time.perf_counter()
        result = jax.device_get(rollout(jnp.asarray(block)))
        jax.block_until_ready(result.terminal_state)
        compile_and_run_times.append(time.perf_counter() - tic)
        outputs.append(
            tuple(np.asarray(x)[:valid] for x in result)
        )
        print(
            f"block {start // args.batch_size + 1}/{(len(state)+args.batch_size-1)//args.batch_size}: "
            f"rows={valid} elapsed={compile_and_run_times[-1]:.3f}s",
            flush=True,
        )
    elapsed = time.perf_counter() - total_start
    terminal_state = np.concatenate([x[0] for x in outputs])
    previous_omega = np.concatenate([x[1] for x in outputs])
    done = np.concatenate([x[2] for x in outputs])
    max_action = np.concatenate([x[3] for x in outputs])
    audit = np.concatenate([x[4] for x in outputs])
    first_contact_state = np.concatenate([x[5] for x in outputs])
    first_contact_detected = np.concatenate([x[6] for x in outputs])
    first_contact_step = np.concatenate([x[7] for x in outputs])
    first_contact_leg_sink = np.concatenate([x[8] for x in outputs])

    arrays = first_contact_arrays(
        terminal_state,
        first_contact_detected,
        first_contact_leg_sink,
    )
    overall = summarize(arrays)
    by_source = {}
    for source_id, name in enumerate(source_names):
        by_source[str(name)] = summarize(arrays, source == source_id)
    summary = {
        "protocol": "rocket_booster_recovery_first_contact_7000kg_ood_evaluation_v2",
        "controller": {
            "name": "Rocket Booster Recovery v2 first-contact 7000kg baseline",
            "type": "deterministic classical composite controller",
            "neural_network": False,
            "ppo": False,
            "training_updates": 0,
            "checkpoint": None,
            "rcs_pitch_yaw_enabled": False,
            "rcs_roll_only": True,
            "grid_roll_enabled": False,
        },
        "dataset": {
            "absolute_path": str(args.dataset.resolve()),
            "sha256": _sha256(args.dataset),
            "slice_offset": args.offset,
            "slice_rows": len(state),
            "source_names": source_names.tolist(),
        },
        "plant": {
            "integrator": args.integrator,
            "substeps": args.substeps,
            "resolved_env_cfg": dict(plant_cfg._asdict()),
            "provenance": provenance(),
        },
        "success_definition": {
            "name": "landing_success",
            "only_success_standard": True,
            "endpoint": "interpolated_first_landing_leg_contact",
            "post_contact_damping_credit": False,
            "thresholds": SUCCESS_THRESHOLDS,
        },
        "overall": overall,
        "by_source": by_source,
        "runtime_audit": _audit_summary(audit, max_action),
        "execution": {
            "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": elapsed,
            "block_seconds": compile_and_run_times,
            "batch_size": args.batch_size,
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(x) for x in jax.devices()],
            "python": platform.python_version(),
            "pid": os.getpid(),
        },
        "artifact_hashes": {
            "controller_config": file_sha256(args.config),
            "controller_source": file_sha256(ROOT / "src/rocket_booster_recovery_controller.py"),
            "plant_adapter_source": file_sha256(ROOT / "src/plant_adapter.py"),
            "metrics_source": file_sha256(ROOT / "src/metrics.py"),
            "evaluator_source": file_sha256(Path(__file__)),
        },
    }
    result_path = args.out_dir / "terminal_results.npz"
    np.savez_compressed(
        result_path,
        initial_state=state,
        terminal_state=terminal_state,
        terminal_previous_omega=previous_omega,
        first_contact_state=first_contact_state,
        first_contact_step=first_contact_step,
        first_contact_leg_sink_speed_mps=first_contact_leg_sink,
        source_id=source,
        source_row=source_row,
        done=done,
        max_abs_action=max_action,
        audit=audit,
        audit_columns=np.asarray(AUDIT_COLUMNS),
        **{name: np.asarray(value) for name, value in arrays.items()},
    )
    summary["terminal_results"] = {
        "absolute_path": str(result_path.resolve()),
        "sha256": _sha256(result_path),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "resolved_controller_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "summary": str(summary_path.resolve()),
        "trajectories": len(state),
        "landing_success": overall["landing_success_pass"],
        "elapsed_seconds": elapsed,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
