# Cross-architecture validation protocols

## Sentinel architecture quick check

For ResNet-18 / CIFAR-100 trained variants, cheap sentinel checks:
- WideResNet-28-10 / CIFAR-100 (1-2 seeds, 20 epochs)
- ResNet-50 / CIFAR-100 (1-2 seeds, 20 epochs)

If the law transfers (relative gap reduction within 30%), it's likely
a mechanism. If it diverges, it's likely a ResNet-18 / 20-epoch artifact.

## Cross-dataset quick check

Tiny-ImageNet is already in the protocol. Adding:
- CIFAR-10 stability check at high rho (the run already saw cifar-10
  variance creep up at rho=0.18+)
- A medium dataset like ImageNet-100 (subset) at small budget

## Long-training boundary

The task's mature T3 protocol is 20 epochs; do not treat longer schedules as
part of the standard benchmark. As a separate external-validity check, a
2x-epoch follow-up can test whether a mechanism survives a longer optimization
horizon, because optimizer rankings may shift when training is extended.

## Publication-level claims

- "scaling law" requires R² ≥ 0.95 fit on ≥4 points AND extrapolation
  to a new point with <1pp error.
- "universal" requires the same mechanism beating baselines on
  ≥3 architectures and ≥2 datasets.
- "obsolete" requires showing the dominated direction also fails on
  external arch / data.
