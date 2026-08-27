# Python Virtual Environment And GPU Setup

This example does not distribute `.venv/`. Python virtual environments are not
reliably portable across hosts, and CUDA/JAX binaries must match the target driver and
platform. The root `requirements.txt` is the authoritative reproducible dependency
list.

## Reference Environment

The original baseline score and execution characteristics were measured only in this
GPU-server environment:

- Python 3.11.15.
- JAX and JAXLIB 0.9.2 with a CUDA 12 backend.
- NumPy 2.4.6.
- SciPy 1.17.1.
- Matplotlib 3.11.1.
- PyYAML 6.0.3.
- One server with 8x NVIDIA H100 80GB HBM3 accelerators; one evaluator used one H100.

No personal-computer score or performance record exists. A Linux/NVIDIA PC may reuse
the CUDA dependencies but still requires remeasurement. Apple Silicon requires a
separate CPU or Metal environment and must not treat this CUDA requirements file as a
validated configuration.

Both Praxist tasks configure their runtime as `../.venv`, so create the virtual
environment at the example root, not inside either task directory.

## Create The Environment

```bash
cd "$HOME/PraxistExamples/rocket_booster_recovery"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Exit the environment with:

```bash
deactivate
```

To rebuild while retaining the old environment for rollback:

```bash
mv .venv .venv.backup
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Verify Python Dependencies

```bash
./.venv/bin/python - <<'PY'
import jax
import matplotlib
import numpy
import scipy
import yaml

print("jax", jax.__version__)
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("matplotlib", matplotlib.__version__)
print("pyyaml", yaml.__version__)
PY
```

## Verify The GPU

Confirm that the driver can see the device:

```bash
nvidia-smi
```

Then verify that JAX uses the GPU instead of silently selecting CPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
./.venv/bin/python - <<'PY'
import jax

print("backend:", jax.default_backend())
print("devices:", jax.devices())
assert jax.default_backend() == "gpu"
assert len(jax.devices()) == 1
PY
```

The Praxist central scheduler assigns one GPU to each formal evaluator process. A
normal peer process must not import JAX and occupy every visible device.
`XLA_PYTHON_CLIENT_PREALLOCATE=false` must remain enabled.

## Verify The Baseline Code

The lightweight tests do not create run records:

```bash
ROCKET_BOOSTER_RECOVERY_TASK_DIR=task_GPU_server ./scripts/run_tests.sh
```

Run a real baseline canary:

```bash
ROCKET_BOOSTER_RECOVERY_TASK_DIR=task_GPU_server ./scripts/run_baseline.sh canary
```

The canary writes to `experiments/manual_canary/` under the selected task, which Git
ignores. To keep a one-time check outside the task entirely:

```bash
out_dir="$(mktemp -d)"
ROCKET_BOOSTER_RECOVERY_TASK_DIR=task_GPU_server \
  ./scripts/run_baseline.sh canary "$out_dir"
```

## Praxist Environment Boundary

- `.venv` supplies the Python runtime for the controller, JAX simulator, and
  evaluator.
- The operator environment supplies the `praxist` CLI, agent runtime, model provider,
  and authentication.
- Do not copy API keys, GitHub tokens, ChatGPT or Claude login files, or provider
  configuration into the example.
- Before launch, verify `praxist --version` and resolve the selected task.
- Keep read-only resolve artifacts outside the task:

```bash
resolve_dir="$(mktemp -d)"
praxist resolve "$PWD/task_GPU_server" --run-dir "$resolve_dir"
```

Then launch directly:

```bash
praxist start --task-path "$PWD/task_GPU_server" \
  --cohort 16 --generations 30 --daemonize --json
```

## Troubleshooting

- If `jax.default_backend()` returns `cpu`, verify the NVIDIA driver, confirm that
  `jax[cuda12]==0.9.2` is installed, and check that `CUDA_VISIBLE_DEVICES` is not
  accidentally empty.
- A failed `libcuda.so` or CUDA plugin load is a host driver or dynamic-library issue.
  Do not work around it by uploading another machine's virtual environment.
- If Praxist cannot find the task Python, verify that `.venv/bin/python` exists and is
  executable at the example root. `task.yaml` uses `../.venv/bin/python`.
- If one process reserves excessive GPU memory, verify that
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set.

## PC Baseline Measurement Boundary

`task_PC/` contains the same scientific harness as the server task, but all PC
performance evidence is unmeasured. After adapting dependencies and the backend to
the target PC, produce a real complete result with:

```bash
ROCKET_BOOSTER_RECOVERY_TASK_DIR=task_PC ./scripts/run_tests.sh
ROCKET_BOOSTER_RECOVERY_TASK_DIR=task_PC ./scripts/run_baseline.sh complete
```

Only a complete result that passes static contracts and protocol-integrity checks may
be recorded under `task_PC/assets/baselines/` and used to restore
`task_PC/task.yaml:baselines`. Never copy the 8x H100 values from `task_GPU_server/`
and label them as PC measurements.
