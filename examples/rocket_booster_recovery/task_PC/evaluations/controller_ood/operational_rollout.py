"""JAX rollout that stops scoring at interpolated first landing-leg contact."""
from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


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


class OperationalRolloutResult(NamedTuple):
    terminal_state: jax.Array
    terminal_previous_omega: jax.Array
    done: jax.Array
    max_abs_action: jax.Array
    audit: jax.Array
    first_contact_state: jax.Array
    first_contact_detected: jax.Array
    first_contact_step: jax.Array
    first_contact_leg_sink_speed_mps: jax.Array


def _select_tree(proposed: Any, previous: Any, active: jax.Array) -> Any:
    return jax.tree.map(
        lambda x, y: jnp.where(
            active.reshape((active.shape[0],) + (1,) * (x.ndim - 1)), x, y
        ),
        proposed,
        previous,
    )


def build_operational_rollout(
    *,
    batch_size: int,
    module: Any,
    controller_cfg: dict[str, Any],
    audit_cfg: dict[str, Any],
    plant: Any,
    plant_cfg: Any,
):
    """Build a static rollout whose landing endpoint precedes damper response."""

    init_batch = jax.vmap(lambda state: module.init_memory(state, controller_cfg))
    control_batch = jax.vmap(
        lambda state, memory: module.control_step(state, memory, controller_cfg)
    )
    key = jax.random.PRNGKey(0)
    plant_batch = jax.vmap(
        lambda state, action: plant.step_one(key, state, action, plant_cfg)
    )
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
        leg_sink = jnp.max(jnp.where(lowest, jnp.maximum(-tip_velocities[:, 0], 0.0), 0.0))
        return min_height, leg_sink

    tip_metrics_batch = jax.vmap(tip_metrics_one)

    def interpolate_state(
        state: jax.Array, proposed_state: jax.Array, alpha: jax.Array
    ) -> jax.Array:
        value = state + alpha[:, None] * (proposed_state - state)
        quat = value[:, 6:10]
        quat = quat / jnp.maximum(jnp.linalg.norm(quat, axis=1, keepdims=True), 1e-8)
        return value.at[:, 6:10].set(quat)

    @jax.jit
    def rollout(initial_state: jax.Array) -> OperationalRolloutResult:
        if initial_state.shape != (batch_size, 16):
            raise ValueError(
                f"compiled batch expects {(batch_size, 16)}, got {initial_state.shape}"
            )
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
            initial_state,
            memory,
            done,
            previous_omega,
            active_steps,
            max_abs_action,
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
            first_contact_state,
            first_contact_detected,
            first_contact_step,
            first_contact_leg_sink,
        )

        def body(carry: tuple[Any, ...], unused: Any):
            del unused
            (
                state,
                memory,
                done,
                terminal_prev_w,
                active_steps,
                max_abs_action,
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
                first_contact_state,
                first_contact_detected,
                first_contact_step,
                first_contact_leg_sink,
            ) = carry

            active = ~done
            action, proposed_memory, diagnostic = control_batch(state, memory)
            _, proposed_state, _, proposed_done, info = plant_batch(state, action)
            finite = jnp.all(jnp.isfinite(proposed_state), axis=1)

            height_now, _ = tip_metrics_batch(state)
            height_next, _ = tip_metrics_batch(proposed_state)
            crossing = active & finite & (~first_contact_detected) & (height_next <= 0.0)

            lo = jnp.zeros(n, dtype=state.dtype)
            hi = jnp.ones(n, dtype=state.dtype)

            def bisect(_: int, bracket: tuple[jax.Array, jax.Array]):
                low, high = bracket
                mid = 0.5 * (low + high)
                mid_state = interpolate_state(state, proposed_state, mid)
                mid_height, _ = tip_metrics_batch(mid_state)
                high = jnp.where(mid_height <= 0.0, mid, high)
                low = jnp.where(mid_height > 0.0, mid, low)
                return low, high

            lo, hi = jax.lax.fori_loop(0, 10, bisect, (lo, hi))
            linear_alpha = jnp.clip(
                height_now / jnp.maximum(height_now - height_next, 1e-8), 0.0, 1.0
            )
            alpha = jnp.where(height_now > 0.0, hi, linear_alpha)
            contact_state = interpolate_state(state, proposed_state, alpha)
            _, contact_leg_sink = tip_metrics_batch(contact_state)

            first_contact_state = jnp.where(
                crossing[:, None], contact_state, first_contact_state
            )
            first_contact_detected = first_contact_detected | crossing
            step_number = active_steps + active.astype(jnp.int32)
            first_contact_step = jnp.where(crossing, step_number, first_contact_step)
            first_contact_leg_sink = jnp.where(
                crossing, contact_leg_sink, first_contact_leg_sink
            )

            plant_terminal = active & ((proposed_done > 0.5) | (~finite))
            newly_done = crossing | plant_terminal
            terminal_prev_w = jnp.where(
                newly_done[:, None], state[:, 10:13], terminal_prev_w
            )
            scored_state = jnp.where(crossing[:, None], contact_state, proposed_state)
            next_state = jnp.where(active[:, None], scored_state, state)
            next_done = done | newly_done
            next_memory = _select_tree(proposed_memory, memory, active)

            active_f = active.astype(state.dtype)
            active_steps = step_number
            max_abs_action = jnp.maximum(
                max_abs_action, jnp.abs(action) * active_f[:, None]
            )
            forbidden = jnp.max(jnp.abs(action[:, jnp.asarray([4, 5, 8])]), axis=1)
            max_forbidden = jnp.maximum(max_forbidden, forbidden * active_f)
            max_plant_lateral_rcs = jnp.maximum(
                max_plant_lateral_rcs, jnp.abs(info[:, 15]) * active_f
            )
            rcs_nm = diagnostic[:, 16]
            rcs_abs_impulse = rcs_abs_impulse + jnp.abs(rcs_nm) * dt * active_f
            rcs_signed_impulse = rcs_signed_impulse + rcs_nm * dt * active_f
            gimbal_now = diagnostic[:, 11:13]
            grid_now = diagnostic[:, 13:15]
            throttle_now = diagnostic[:, 15]
            gimbal_tv = gimbal_tv + jnp.sum(
                jnp.abs(gimbal_now - memory.prev_gimbal_rad), axis=1
            ) * active_f
            grid_tv = grid_tv + jnp.sum(
                jnp.abs(grid_now - memory.prev_grid_rad), axis=1
            ) * active_f
            throttle_tv = throttle_tv + jnp.abs(
                throttle_now - memory.prev_throttle
            ) * active_f
            emergency_steps = emergency_steps + (
                (diagnostic[:, 1] > 0.5) & active
            ).astype(jnp.int32)
            phase_index = jnp.clip(diagnostic[:, 0].astype(jnp.int32), 0, 3)
            phase_steps = phase_steps + jax.nn.one_hot(
                phase_index, 4, dtype=jnp.int32
            ) * active[:, None]
            gimbal_sat = gimbal_sat + (
                (jnp.max(jnp.abs(gimbal_now), axis=1) >= float(audit_cfg["gimbal_normal_rad"]) - 1e-6)
                & active
            ).astype(jnp.int32)
            grid_sat = grid_sat + (
                (jnp.max(jnp.abs(grid_now), axis=1) >= float(audit_cfg["grid_normal_rad"]) - 1e-6)
                & active
            ).astype(jnp.int32)
            throttle_sat = throttle_sat + (
                (
                    (throttle_now <= float(audit_cfg["throttle_normal_min"]) + 1e-6)
                    | (throttle_now >= float(audit_cfg["throttle_normal_max"]) - 1e-6)
                )
                & active
            ).astype(jnp.int32)
            cap = jnp.where(
                diagnostic[:, 1] > 0.5,
                float(audit_cfg["rcs_roll_emergency_cap_nm"]),
                float(audit_cfg["rcs_roll_normal_cap_nm"]),
            )
            rcs_cap = rcs_cap + (
                (jnp.abs(rcs_nm) >= cap - 1.0) & active
            ).astype(jnp.int32)
            max_body_error = jnp.maximum(
                max_body_error, diagnostic[:, 17] * active_f
            )
            min_brake_margin = jnp.where(
                active,
                jnp.minimum(min_brake_margin, diagnostic[:, 3]),
                min_brake_margin,
            )
            max_qdyn = jnp.maximum(max_qdyn, diagnostic[:, 4] * active_f)
            nonfinite_seen = nonfinite_seen | (active & (~finite))

            return (
                next_state,
                next_memory,
                next_done,
                terminal_prev_w,
                active_steps,
                max_abs_action,
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
                first_contact_state,
                first_contact_detected,
                first_contact_step,
                first_contact_leg_sink,
            ), None

        final, _ = jax.lax.scan(body, carry, xs=None, length=int(plant_cfg.max_steps))
        endpoint_state = jnp.where(
            final[24][:, None], final[23], final[0]
        )
        audit = jnp.column_stack(
            [
                final[4],
                final[6],
                final[7],
                final[8],
                final[9],
                final[10],
                final[11],
                final[12],
                final[13],
                final[14],
                final[15],
                final[16],
                final[17],
                final[18],
                final[19],
                final[20],
                final[21],
                final[22],
            ]
        )
        return OperationalRolloutResult(
            terminal_state=endpoint_state,
            terminal_previous_omega=final[3],
            done=final[2],
            max_abs_action=final[5],
            audit=audit,
            first_contact_state=final[23],
            first_contact_detected=final[24],
            first_contact_step=final[25],
            first_contact_leg_sink_speed_mps=final[26],
        )

    return rollout
