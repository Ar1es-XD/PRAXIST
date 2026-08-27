"""Frozen Swordfish C05 plant with a full-physical-action adapter.

The archived harness normally interprets ``step_one`` input as a PPO residual
and composes it with a variant-local guidance prior.  Rocket Booster Recovery produces the
complete nine-channel physical command itself, so this module replaces only
that action-composition boundary with an identity clip.  Force, torque,
contact, mass-flow, termination, and RK4 equations remain the frozen vendor
implementation.

No checkpoint is loaded and no policy/actor function is callable through this
adapter.  RCS pitch/yaw and grid-roll are forced to exact floating-point zero
again at the plant boundary as a defense-in-depth measure.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import jax.numpy as jnp


ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT / "vendor/frozen_c05_plant"
PLANT_PATH = VENDOR_DIR / "ppo_rocket_6dof_finned_jax.py"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_vendor_module() -> ModuleType:
    # Variant-local helper imports are intentionally resolved from the frozen
    # copy, never from the mutable original run directory.
    sys.path.insert(0, str(VENDOR_DIR))
    spec = importlib.util.spec_from_file_location("rocket_booster_recovery_frozen_c05_plant", PLANT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen plant: {PLANT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _physical_action_identity(state, action, cfg):
    """Map Rocket Booster Recovery's complete physical action to the frozen plant.

    Literal multiplication is avoided on locked channels: indexing assignment
    makes the three forbidden channels bit-exact +0.0 even if a caller is
    malformed.  The evaluator separately rejects any pre-boundary violation.
    """
    del state, cfg
    a = jnp.clip(action, -1.0, 1.0)
    a = a.at[4].set(jnp.asarray(0.0, dtype=a.dtype))
    a = a.at[5].set(jnp.asarray(0.0, dtype=a.dtype))
    a = a.at[8].set(jnp.asarray(0.0, dtype=a.dtype))
    return a


def _forbidden_neural_entrypoint(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("neural/PPO entry point is forbidden in the Rocket Booster Recovery task")


def load_frozen_plant(integrator: str = "rk4", substeps: int = 1):
    """Return ``(module, EnvCfg)`` for the frozen 29 t C05 physical stack."""
    plant = _load_vendor_module()

    # This is the sole interface adaptation. JAX-traced force functions perform
    # a global lookup, so both the outer stage and all RK stages receive the
    # same full physical command.
    plant.apply_guidance_residual = _physical_action_identity

    # Definitions may exist in the vendor archive for provenance, but any
    # accidental call is converted into a hard failure.
    for name in (
        "init_params",
        "policy_value",
        "sample_action",
        "ppo_update",
        "train_ppo",
        "load_checkpoint",
    ):
        if hasattr(plant, name):
            setattr(plant, name, _forbidden_neural_entrypoint)

    if integrator not in {"rk4", "rk2_full"}:
        raise ValueError("formal Rocket Booster Recovery plant supports rk4 or rk2_full only")
    if substeps < 1:
        raise ValueError("substeps must be positive")

    cfg = plant.EnvCfg()._replace(
        # Frozen 29 t selected physical vehicle.
        dt=0.1,
        max_steps=900,
        mass_full=25600.0,
        mass_empty=22200.0,
        fuel_scale=2.0,
        thrust_max=845000.0,
        isp=282.0,
        length=40.0,
        radius=1.83,
        lever_x=-15.0,
        gimbal_max=0.0873,
        cd=0.60,
        rho=1.05,
        # Active forward grid-fin plant from the selected variant.
        fins_on=1.0,
        fin_force_on=1.0,
        fin_lift_on=1.0,
        fin_drag_on=1.0,
        fin_damping_on=1.0,
        fin_active_on=1.0,
        fin_area_each=0.75,
        fin_count=4.0,
        fin_station_x=0.5,
        fin_cl_alpha=2.0,
        fin_cd0=0.18,
        fin_cd_alpha=1.20,
        fin_alpha_stall=0.45,
        fin_torque_damping=0.22,
        grid_fin_control_max=0.35,
        grid_fin_control_cl=1.6,
        grid_fin_roll_control_cl=0.55,
        fin_incidence_on=0.0,
        # Plant authority exists, but Rocket Booster Recovery may address only its roll channel.
        rcs_torque_max=180000.0,
        rcs_authority_scale=1.0,
        rcs_schedule_on=1.0,
        rcs_near_com_scale=1.0,
        rcs_far_outer_scale=3.0,
        rcs_boost_radius=700.0,
        rcs_schedule_width=140.0,
        # Selected terminal drag and contact stack.
        terminal_drag_on=1.0,
        terminal_drag_gain=3.0,
        terminal_drag_alt_low=300.0,
        terminal_drag_alt_high=350.0,
        terminal_drag_radial_gate_on=1.0,
        terminal_drag_far_gain=3.0,
        terminal_drag_far_radius=1124.0,
        terminal_drag_radial_width=60.0,
        gear_on=1.0,
        gear_contact_height_m=3.0015,
        gear_footprint_radius_m=6.0,
        gear_n_legs=4,
        gear_spring_k_npm=2.0e4,
        gear_damper_c_nspm=3.0e4,
        gear_friction_mu=0.3,
        gear_contact_restore_scale=0.0,
        gear_bottom_out_guard=1.0,
        gear_bottom_out_force_frac=0.90,
        integrator=integrator,
        integrator_substeps=int(substeps),
        # These are inert under the identity adapter and are zeroed explicitly
        # so the resolved configuration cannot be mistaken for PPO operation.
        guidance_on=0.0,
        residual_scale=0.0,
        residual_gimbal=0.0,
        residual_throttle=0.0,
        residual_rcs=0.0,
        residual_gridfin=0.0,
        residual_rcs_x=0.0,
        residual_rcs_y=0.0,
        residual_rcs_z=0.0,
        far_outer_mask_on=0.0,
        fuel_tail_on=0.0,
    )

    # Module-level constants are resolved when the frozen helper is imported.
    # Fail immediately if the vendored config and explicit EnvCfg diverge.
    if abs(float(plant._FIN_STATION_X) - 0.5) > 1e-12:
        raise RuntimeError(f"frozen grid-fin station mismatch: {plant._FIN_STATION_X}")
    if abs(float(plant._CL_MAX) - 0.35) > 1e-12:
        raise RuntimeError(f"frozen grid-fin limit mismatch: {plant._CL_MAX}")
    return plant, cfg


def provenance() -> dict:
    files = sorted(p for p in VENDOR_DIR.rglob("*") if p.is_file())
    return {
        "frozen_vendor_dir": str(VENDOR_DIR),
        "plant_path": str(PLANT_PATH),
        "plant_sha256": file_sha256(PLANT_PATH),
        "vendor_files": {str(p.relative_to(VENDOR_DIR)): file_sha256(p) for p in files},
        "action_adapter": "full physical action; clip; hard-zero indices 4,5,8",
        "neural_checkpoint_loaded": False,
        "ppo_updates": 0,
    }
