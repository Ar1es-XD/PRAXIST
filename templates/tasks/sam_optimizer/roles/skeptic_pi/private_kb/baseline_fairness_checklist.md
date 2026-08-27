# Baseline fairness checklist

When a variant V claims to "dominate" or "beat" baseline B, verify ALL:

1. **Hyperparameter parity**: was B given the same rho/learning-rate/
   weight-decay sweep as V? An apparent dominance often disappears when
   B is fairly tuned.

2. **Same protocol**: same datasets, same seed pool, same epochs, same
   batch size, same warmup schedule. If one method receives a wider rho sweep,
   the result cannot distinguish a mechanism improvement from a
   hyperparameter discovery.

3. **Confidence intervals**: is the improvement larger than the seed
   variance? With 3-5 seeds and CIFAR-100 std ~0.003-0.005 per seed,
   improvements <0.5pp are within noise.

4. **Cherry-picking**: were intermediate seeds dropped?
   `promotion_eligible` is a structural check but doesn't catch
   selective reporting.

## Reviewer attack patterns

- "Did you compare against SAM/ASAM under the same rho sweep?"
- "Does the law survive architecture change (e.g. ResNet-50, WRN-28-10,
  ViT-S)?"
- "Is this an algorithmic mechanism or just hyperparameter retuning?"
- "What is the seed variance? You report 5-seed mean; show 5-seed std."
- "Why is this Pareto-dominant? Show the dominated baseline at the
  same overhead budget."

## Words to forbid until external validation

- universal
- generally dominant
- architecture-independent
- solved
- obsolete
- breakthrough optimizer

## Words allowed with explicit boundary

- single-protocol scaling law
- current-run Pareto lever
- candidate mechanism (pending external validation)
- under current protocol (ResNet-18 / CIFAR-100/10/tiny-IN / 20 epochs)
