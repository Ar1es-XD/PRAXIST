# SAM / ASAM / GSAM mechanism notes

## SAM (Foret et al. 2021)
- Two-step: ascent step ε = ρ * ∇L / ||∇L||, then descent step at w + ε.
- ρ is the perturbation magnitude.
- Effect: pushes optimizer toward flat minima (low Hessian top eigenvalue).
- Overhead: 2x SGD.

## ASAM (Kwon et al. 2021)
- Adaptive SAM: ε_i = ρ * |w_i| * ∇L_i / ||·||.
- Per-parameter scaling: large weights get larger perturbation.
- Treat perturbation scale and scheduling as protocol parameters. Compare them
  under matched seeds, datasets, epochs, and compute budgets.
- Do not infer an ASAM advantage from an unmatched rho sweep. Measure accuracy,
  generalization gap, sharpness, and overhead through the task evaluator.

## GSAM (Zhuang et al. 2022)
- Adds a gradient decomposition step: project the descent gradient to remove
  the sharpness-increasing component.
- Whether decomposition complements SAM or ASAM is a testable interaction, not
  a task prior. Evaluate each composition against matched single-mechanism
  controls before assigning it parent authority.

## Composition tests

- Compare a composition with both of its component controls under the same
  protocol.
- Separate per-step mechanisms from per-epoch schedules when testing
  interaction effects.
- Require repeated evidence before calling an interaction synergistic,
  redundant, or harmful.
- Record negative interactions as findings instead of turning them into
  permanent prompt assumptions.

## Successful lineage shape
1. Discover a single-axis lever (warmup, rho, GSAM, layer-wise ρ).
2. Test orthogonality: does lever_A × lever_B improve on max(A, B) or just
   add noise?
3. Build scaling law: is the lever monotonic over a range?
4. Test boundary: where does the law break (cifar10 instability, accuracy
   collapse, overhead blowup)?
