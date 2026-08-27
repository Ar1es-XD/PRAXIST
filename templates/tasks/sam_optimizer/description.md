# SAM Optimizer Variants Research

## Research Scope (Hard Constraints)

**This task has a deliberately narrow scope.** The deliverable is a
**portable PyTorch Optimizer subclass** that improves on vanilla SAM
without sacrificing drop-in compatibility. The constraint is not a
limitation — it IS the research goal:

1. **The variant MUST be a `torch.optim.Optimizer` subclass.** Standard
   interface only (`step()`, `zero_grad()`, `param_groups`,
   `state_dict()`).
2. **NO training-loop modifications.** No custom forward-backward
   hooks, no architecture changes, no LR-scheduler tricks, no
   curriculum / data tricks. The variant must work in any standard
   PyTorch training loop without changes.
3. **Fixed evaluation protocol.** Use the task-local tiered evaluator:
   ResNet-18, cosine LR, batch size 256, seed pool {42,43,44,45,46},
   and the T1/T2/T3 cells declared in `task.yaml:tiered_eval`. T3 is
   the full mature benchmark: CIFAR-10, CIFAR-100, Tiny-ImageNet,
   20 epochs, 5 seeds.

If you find yourself wanting to change anything outside the optimizer
class, your idea is OFF-SCOPE — reformulate so the change can be
expressed entirely as a different `optimizer.step()` implementation.

## Innovation Space

Within the scope above, you have full freedom along **6 design axes**
(declared in `task.yaml: evaluation.diversity_dimensions`):

| Axis | What varies |
|---|---|
| `computational_mechanism` | Specific perturbation/step formula (two-step / lookahead / single-pass / curvature-aware...) |
| `adaptation_strategy` | What's dynamic vs fixed (rho schedule, per-layer rho, gradient-norm-adaptive...) |
| `information_source` | What gradient/parameter info you consume (direction only / + magnitude / + Hessian / + history) |
| `compute_efficiency_profile` | Compute cost vs SGD (2x / 1.2x periodic / 1.5x chunked / 1.0x single-pass...) |
| `inductive_prior` | What you assume about the loss landscape (param magnitude correlates with sharpness / sharpness varies layer-wise / ...) |
| `decomposition_topology` | How perturbation is decomposed (single global / layer-wise / per-tensor / hierarchical) |

When you publish a `result` or `hypothesis` finding, articulate your
implemented variant on these 6 axes via the `design_dimensions` field; PI
`planned_dimensions` remain separate planning context. Aim for
**clean** classification (≥ 3 of 6 axes differ from each frontier
anchor). The system uses this signal observationally — promotion is
by primary metric — but the diversity classification is logged and
visible to later generations and to the PI synthesizer that writes
the next gen's role contracts.

## DIG, QD, And Gems Policy

This task enables DIG only for absolute gen0 in `task.yaml`. Before editing gen0 optimizer code, a peer
must produce a read-only DIG candidate pool, reviews, a quality-diversity
selection trace, and a selected implementation contract. The cohort-level QD
allocator may select a different validated candidate from the same peer's pool
when that improves generation diversity. Later generations use PI proposal-pool
QD under the normal agenda path, without rerunning DIG.

This SAM template keeps a disabled, opt-in Gems reset profile with a
6-generation interval and a compact cap of 4 visible Gems. It starts in
continuous-evolution mode like newly initialized tasks; enable periodic reset
only after an operator request or diagnostic plateau evidence. If enabled, Gems
admission should favor mature SAM candidates with complete tier evidence. For
this task, T1/T2 remain preliminary task-local stages. Gems uses
`selection_policy: mature_evidence_top_k` and `min_mature_eval_units: 15`,
corresponding to the complete T3 seed/dataset-unit count. Its task-local stage
map is `evidence_stage_min_units: {T1: 3, T2: 6, T3: 15}`; Praxist configuration
still names these counts evaluation units.
Do not copy complete-protocol evaluation-unit thresholds from another task.

## Background

**Sharpness-Aware Minimization (SAM)** seeks parameters that lie in flat regions of the loss landscape, improving generalization. The core idea: instead of minimizing loss at a single point, SAM minimizes the worst-case loss within a neighborhood:

```
min_w  max_{||ε||≤ρ}  L(w + ε)
```

This is approximated by a two-step procedure:
1. Compute the adversarial perturbation: `ε = ρ * ∇L(w) / ||∇L(w)||`
2. Update parameters at the perturbed point: `w ← w - lr * ∇L(w + ε)`

This benchmark studies SAM-family optimizers for ResNet-18 image
classification across CIFAR-10, CIFAR-100, and Tiny-ImageNet, with an emphasis
on the accuracy-efficiency-stability tradeoff surfaced by the 2024-2025
literature.

## Known Variants

- **Vanilla SAM** (Foret et al., 2021): Original formulation with fixed ρ
- **ASAM** (Kwon et al., 2021): Adaptive SAM that normalizes perturbation by parameter scale
- **GSAM** (Zhuang et al., 2022): Adds gradient decomposition to steer toward flatter minima
- **LookSAM** (Liu et al., 2022): Reduces overhead by computing the SAM perturbation periodically
- **SAM-ON** (Liu et al., 2023): SAM with online normalization
- **Recent 2024–2025 variants** (Friendly-SAM, CR-SAM, layerwise- and
  phase-adaptive SAM, etc.): see
  `assets/literature/recent_sam_top_venue_review_2024_2025.md` for the curated
  top-venue survey.

## Research Goals

1. **Propose a novel SAM variant** with clear theoretical or empirical motivation
2. **Implement** the variant as a clean PyTorch optimizer, reusing the baseline training harness
3. **Evaluate** with the task-local T1/T2/T3 evaluator; mature T3 evidence
   covers CIFAR-10, CIFAR-100, and Tiny-ImageNet with ResNet-18 across 5 seeds
4. **Compare** against baselines (SGD, Adam, SAM, ASAM)
5. **Analyze** the loss landscape (sharpness proxies, train-test gap) and the compute overhead

A strong submission should simultaneously answer:

- **Is it better?** Improves test accuracy, train-test gap, or a sharpness-related metric.
- **Is it worth the cost?** Training overhead, memory cost, and extra passes are reported.
- **Is it stable enough to use?** Gains are not contingent on extremely fragile hyperparameter tuning.

## Evaluation Protocol

- **Model**: ResNet-18
- **Datasets**: T1 uses CIFAR-100; T2 uses CIFAR-10 and CIFAR-100; T3 uses
  CIFAR-10, CIFAR-100, and Tiny-ImageNet.
- **Training**: cosine LR schedule, batch size 256, 10 epochs at T1 and
  20 epochs at T2/T3.
- **Seeds**: T1/T2 use seeds 42-44; T3 uses seeds 42-46.
- **Primary metric**: `mean_test_accuracy` across completed benchmark datasets;
  per-dataset `test_accuracy_*` metrics remain available for diagnostics.
- **Auxiliary metrics**:
  - Train-test gap (measure of overfitting)
  - Top Hessian eigenvalue (sharpness proxy)
  - Compute overhead ratio (wall-clock time vs SGD)
  - Sensitivity to learning rate / `rho`

## Baseline Results (Pre-computed)

| Optimizer | CIFAR-100 Acc | CIFAR-10 Acc |
|-----------|--------------|--------------|
| SGD       | 75.20 ± 0.3  | 93.80 ± 0.2 |
| Adam      | 73.80 ± 0.4  | 93.20 ± 0.3 |
| SAM       | 77.10 ± 0.3  | 95.10 ± 0.2 |
| ASAM      | 77.50 ± 0.2  | 95.30 ± 0.1 |

## Code Structure

```
tasks/sam_optimizer/
├── task.yaml
├── prompt_task.jinja2
├── evaluations/
│   └── pareto_tiered/run.py       # Canonical public evaluation entrypoint
└── assets/
    ├── harness/
    │   ├── baseline/train.py       # Training script
    │   └── benchmark/run_benchmark.py   # Internal benchmark runner
    ├── baselines/results.jsonl     # Curated baseline metrics
    ├── dataset_metadata/           # Lightweight metadata + resolver
    └── reference_implementations/  # Curated optimizer corpus
```

## How to Run

```bash
# Quick validation (short run)
python assets/harness/baseline/train.py \
  --optimizer sam --dataset cifar100 \
  --epochs 10 --seed 42

# Tiered evaluation through the only public eval entrypoint
python evaluations/pareto_tiered/run.py \
  --variant-path path/to/your_optimizer.py \
  --output-dir experiments/manual_eval \
  --data-dir "$PRAXIST_DATA_DIR"
```

## Repository Note

`baseline/train.py` now provides a faithful ASAM baseline (a dedicated `ASAM` class that scales the perturbation elementwise by `|w|`, per Kwon et al. 2021). Note that ASAM is typically run with a larger `rho` than vanilla SAM (e.g. 0.5–2.0); the default `--rho 0.05` matches SAM and may need to be raised for ASAM comparisons.
