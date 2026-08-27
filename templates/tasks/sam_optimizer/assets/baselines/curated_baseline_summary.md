# SAM Baseline Summary

Protocol: ResNet-18, CIFAR-10 / CIFAR-100 / Tiny-ImageNet, 20 epochs,
batch size 256, bf16, seeds 42-46. These are curated reference facts for
task guidance and regression checks; raw historical run directories are not
packaged with the task project.

| Optimizer | CIFAR-10 acc | CIFAR-100 acc | Tiny-ImageNet acc | Notes |
|---|---:|---:|---:|---|
| SGD | 0.9128 | 0.7182 | 0.5667 | Momentum SGD, lr=0.1 |
| Adam | 0.9186 | 0.7143 | 0.5645 | lr=1e-3 |
| Vanilla SAM | 0.8930 | 0.7155 | 0.5766 | rho=0.05, SGD base |
| ASAM | 0.9088 | 0.7197 | 0.5720 | rho=0.5, elementwise scale |

Use `results.jsonl` for machine-readable per-dataset means and
`assets/regression_fixtures/baselines/*_multi_benchmark_summary.json` for fixed
multi-benchmark regression fixtures.
