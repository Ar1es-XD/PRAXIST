#!/usr/bin/env python3
"""Replay two archived formal ICs and save representative diagnostic plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .plant_adapter import load_frozen_plant
from .rocket_booster_recovery_controller import DIAGNOSTIC_COLUMNS, control_step, init_memory, load_config, tilt_from_q


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/rocket_booster_recovery_v0_frozen_formal.json")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    plant, pcfg = load_frozen_plant("rk4", 1)
    with np.load(args.formal_results, allow_pickle=False) as d:
        initial = np.asarray(d["initial_state"], np.float32)
        success = np.asarray(d["landing_success_pass"], bool)
        source = np.asarray(d["source_id"], np.int8)
        source_row = np.asarray(d["source_row"], np.int32)
    success_idx = int(np.flatnonzero(success & (source == 0))[0])
    failure_idx = int(np.flatnonzero((~success) & (source == 1))[0])
    indices = np.array([success_idx, failure_idx])
    state0 = jnp.asarray(initial[indices])
    memory0 = jax.vmap(lambda s: init_memory(s, cfg))(state0)
    key = jax.random.PRNGKey(0)
    control = jax.vmap(lambda s, m: control_step(s, m, cfg))
    step = jax.vmap(lambda s, a: plant.step_one(key, s, a, pcfg))

    @jax.jit
    def replay(state, memory):
        done0 = jnp.zeros(2, dtype=bool)
        def body(carry, unused):
            del unused
            state, memory, done = carry
            action, proposed_memory, diagnostic = control(state, memory)
            _, proposed_state, _, proposed_done, info = step(state, action)
            active = ~done
            next_state = jnp.where(active[:, None], proposed_state, state)
            next_memory = jax.tree.map(
                lambda new, old: jnp.where(
                    active.reshape((2,) + (1,)*(new.ndim-1)), new, old
                ),
                proposed_memory,
                memory,
            )
            next_done = done | (active & (proposed_done > 0.5))
            action = jnp.where(active[:, None], action, jnp.zeros_like(action))
            diagnostic = jnp.where(active[:, None], diagnostic, jnp.zeros_like(diagnostic))
            return (next_state, next_memory, next_done), (next_state, action, diagnostic, info, next_done)
        return jax.lax.scan(body, (state, memory, done0), None, length=900)

    _, trace = replay(state0, memory0)
    state_t, action_t, diag_t, info_t, done_t = [np.asarray(x) for x in jax.device_get(trace)]
    # Time-major -> trajectory-major.
    state_t = np.swapaxes(state_t, 0, 1)
    action_t = np.swapaxes(action_t, 0, 1)
    diag_t = np.swapaxes(diag_t, 0, 1)
    info_t = np.swapaxes(info_t, 0, 1)
    done_t = np.swapaxes(done_t, 0, 1)
    np.savez_compressed(
        args.out_dir/"representative_traces.npz",
        formal_index=indices,
        source_id=source[indices],
        source_row=source_row[indices],
        initial_state=initial[indices],
        state=state_t,
        action=action_t,
        diagnostic=diag_t,
        info=info_t,
        done=done_t,
        diagnostic_columns=np.asarray(DIAGNOSTIC_COLUMNS),
        labels=np.asarray(
            ["near_ood_landing_success", "hard_ood_landing_failure"]
        ),
    )

    labels = ["near-OOD landing success", "hard-OOD landing failure"]
    colors = ["#177245", "#b33a3a"]
    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)
    for i, (label, color) in enumerate(zip(labels, colors)):
        terminal_step = int(np.argmax(done_t[i]) + 1) if np.any(done_t[i]) else 900
        sl = slice(0, terminal_step)
        t = np.arange(terminal_step)*float(cfg["dt"])
        s = state_t[i, sl]
        a = action_t[i, sl]
        axes[0].plot(t, s[:, 0], color=color, label=label)
        axes[1].plot(t, np.linalg.norm(s[:, 1:3], axis=1), color=color)
        axes[2].plot(t, np.linalg.norm(s[:, 3:6], axis=1), color=color)
        axes[3].plot(t, info_t[i, sl, 4], color=color)
        axes[4].plot(t, a[:, 0]*float(cfg["gimbal_absolute_rad"]), color=color, linestyle="-", alpha=.9)
        axes[4].plot(t, a[:, 1]*float(cfg["gimbal_absolute_rad"]), color=color, linestyle="--", alpha=.9)
    axes[0].set_ylabel("height [m]")
    axes[1].set_ylabel("lateral radius [m]")
    axes[2].set_ylabel("total speed [m/s]")
    axes[3].set_ylabel("tilt [rad]")
    axes[4].set_ylabel("gimbal y/z [rad]")
    axes[4].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    for ax in axes:
        ax.grid(True, alpha=.3)
    fig.suptitle("Rocket Booster Recovery representative formal OOD traces")
    fig.tight_layout()
    fig.savefig(args.out_dir/"representative_timeseries.png", dpi=170)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for i, (label, color) in enumerate(zip(labels, colors)):
        terminal_step = int(np.argmax(done_t[i]) + 1) if np.any(done_t[i]) else 900
        p = state_t[i, :terminal_step, 0:3]
        ax.plot(p[:, 1], p[:, 2], p[:, 0], color=color, label=label, linewidth=1.6)
        ax.scatter(p[0, 1], p[0, 2], p[0, 0], color=color, marker="o")
        ax.scatter(p[-1, 1], p[-1, 2], p[-1, 0], color=color, marker="x")
    ax.scatter([0], [0], [0], color="black", marker="+", s=80, label="pad")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_zlabel("height x [m]")
    ax.legend(loc="best")
    ax.set_title("Representative 3D trajectories")
    fig.tight_layout()
    fig.savefig(args.out_dir/"representative_trajectories_3d.png", dpi=170)
    plt.close(fig)

    summary = {
        "purpose": "visualization-only replay; not added to formal sample count",
        "formal_indices": indices.tolist(),
        "source_ids": source[indices].tolist(),
        "source_rows": source_row[indices].tolist(),
        "labels": labels,
        "max_forbidden_action_abs": float(np.max(np.abs(action_t[:, :, [4, 5, 8]]))),
        "trace_npz": str((args.out_dir/"representative_traces.npz").resolve()),
        "timeseries_png": str((args.out_dir/"representative_timeseries.png").resolve()),
        "trajectory_png": str((args.out_dir/"representative_trajectories_3d.png").resolve()),
    }
    (args.out_dir/"trace_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
