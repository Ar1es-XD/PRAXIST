---
description: Run the complete native Rust Rocket Booster Recovery Praxist example.
---

# Rocket Booster Recovery (Rust)

This complete example implements the Rocket Booster Recovery research problem
as a native Rust project. It is independent of the Python/JAX example, keeps
its Cargo dependencies vendored for offline builds, and includes three Praxist
task profiles:

| Profile | Intended host |
|---|---|
| `task_GPU_server` | High-concurrency Linux server; the evaluator remains CPU-only |
| `task_linux` | Portable x86_64 or aarch64 Linux host |
| `task_macos` | Apple Silicon macOS host with platform-specific baseline qualification |

The packaged copy is read-only. Install a writable project before building,
evaluating, or launching research:

```bash
praxist examples install rocket_booster_recovery_rust
cd ~/PraxistExamples/rocket_booster_recovery_rust
```

The project requires Rust 1.85 or newer. Validate the selected profile from the
writable copy:

```bash
cargo test --release --workspace --locked --offline
praxist resolve "$PWD/task_linux" --run-dir "$(mktemp -d)"
```

Then launch through an agent skill or the direct CLI using that task directory:

```bash
praxist start --task-path "$PWD/task_linux" --daemonize --json
```

The example's [source README](https://github.com/sapientinc/praxist/tree/main/examples/rocket_booster_recovery_rust#readme)
is the sole owner of its scientific protocol, baseline evidence, build audit,
platform constraints, and task-profile details.
