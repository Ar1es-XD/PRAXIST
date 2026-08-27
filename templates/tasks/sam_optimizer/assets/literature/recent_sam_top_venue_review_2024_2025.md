# Recent Sharpness-Aware Minimization Research (2024-2025)

This task-local survey focuses on papers that treat Sharpness-Aware
Minimization (SAM) as a central method or theoretical object. It prioritizes
ICLR, ICML, NeurIPS, UAI, AAAI, CVPR, and ICCV work that informs a
CIFAR/ResNet optimizer benchmark.

The survey is research context, not measured task evidence.

## Research Trends

Recent SAM work has moved from asking whether SAM works to four more precise
questions:

1. Why does SAM work beyond the flat-minima narrative?
2. During which training phases is it most useful?
3. How should perturbation direction, objective, and layer scaling change?
4. How can sharpness-aware behavior be retained with lower overhead?

The most relevant conclusions for this task are:

- later-phase sharpness-aware updates may retain much of the benefit;
- layerwise or scale-aware perturbation can be more meaningful than one global
  radius;
- direction quality and stability matter as much as perturbation magnitude;
- low-overhead approximations deserve explicit wall-clock evaluation.

## 2024 Method and Theory

| Venue | Work | Main contribution | Relevance to this task |
|---|---|---|---|
| ICLR | *Sharpness-Aware Minimization Enhances Feature Quality via Balanced Learning* | Connects SAM to balanced feature learning, not only flatness | Motivates representation and feature-quality analysis |
| ICLR | *TRAM: Bridging Trust Regions and Sharpness Aware Minimization* | Relates trust regions and structured perturbation | Suggests richer perturbation geometry |
| ICLR | *Domain-Inspired Sharpness-Aware Minimization Under Domain Shifts* | Makes perturbation task-structure-aware | Shows that one generic perturbation rule is not always optimal |
| CVPR | *Friendly Sharpness-Aware Minimization* | Separates useful and harmful perturbation components | Provides a practical low-overhead correction direction |
| ICML | *A Universal Class of Sharpness-Aware Minimization Algorithms* | Generalizes the sharpness objective and derives alternatives | Treats the definition of sharpness as a design variable |
| ICML | *Improving SAM Requires Rethinking its Optimization Formulation* | Recasts SAM as a bilevel problem | Motivates objective-level rather than purely heuristic changes |
| ICML | *Lookbehind-SAM: k steps back, 1 step forward* | Improves inner maximization and update variance | Useful comparison for direction quality, but potentially expensive |
| ICML | *Improving Sharpness-Aware Minimization by Lookahead* | Improves optimization stability | Motivates trajectory smoothing |
| ICML | *On the Duality Between Sharpness-Aware Minimization and Adversarial Training* | Connects weight and input perturbation perspectives | Supports robustness-oriented analysis |
| AAAI | *CR-SAM: Curvature Regularized Sharpness-Aware Minimization* | Adds curvature regularization | Shows continued value of curvature-aware design |
| NeurIPS | *Fundamental Convergence Analysis of Sharpness-Aware Minimization* | Gives a general convergence framework | Encourages interpretable update rules |
| NeurIPS | *SAMPa: Sharpness-aware Minimization Parallelized* | Parallelizes the two-gradient process | Establishes efficiency as a primary objective |
| NeurIPS | *Explicit Eigenvalue Regularization Improves Sharpness-Aware Minimization* | Connects effective perturbation with leading curvature directions | Motivates stable direction proxies without requiring a full Hessian |
| NeurIPS | *muP2: Effective Sharpness Aware Minimization Requires Layerwise Perturbation Scaling* | Shows the importance of layerwise scaling in wide models | Strong motivation for simple per-layer normalization |
| NeurIPS | *Implicit Regularization of Sharpness-Aware Minimization for Scale-Invariant Problems* | Explains scale-invariant balancedness effects | Supports scale-aware perturbation allocation |
| NeurIPS | *A Single-Step, Sharpness-Aware Minimization is All You Need to Achieve Efficient and Accurate Sparse Training* | Demonstrates a low-overhead one-step approximation | Supports inexpensive SAM-like updates |

The 2024 literature broadens the explanation beyond flatness, makes efficiency
a first-class concern, and treats perturbation structure, curvature, and layer
scaling as meaningful design axes.

## 2025 Method and Theory

| Venue | Work | Main contribution | Relevance to this task |
|---|---|---|---|
| ICLR | *Sharpness-Aware Minimization Efficiently Selects Flatter Minima Late in Training* | Finds that important benefits may concentrate late in training | Strong motivation for phase-adaptive activation |
| ICLR | *Sharpness-Aware Minimization: General Analysis and Improved Rates* | Unifies analysis of normalized and unnormalized variants | Supports normalization ablations |
| ICML | *Tilted Sharpness-Aware Minimization* | Smooths the extreme min-max objective through exponential tilting | Motivates less brittle objectives |
| ICML | *Focal-SAM: Focal Sharpness-Aware Minimization for Long-Tailed Classification* | Uses class-aware sharpness penalties | Demonstrates data-dependent weighting |
| ICML | *One Arrow, Two Hawks: Sharpness-aware Minimization for Federated Learning via Global Model Trajectory* | Uses global trajectory information | Distinguishes global and local sharpness |
| AAAI | *Asymptotic Unbiased Sample Sampling to Speed Up Sharpness-Aware Minimization* | Uses subsets to reduce overhead while controlling bias | Practical for resource-constrained evaluation |
| UAI | *Critical Influence of Overparameterization on Sharpness-aware Minimization* | Studies dependence on model scale | Warns against assuming transfer across model sizes |
| CVPR | *Beyond Local Sharpness: Communication-Efficient Global Sharpness-aware Minimization for Federated Learning* | Optimizes a global sharpness notion | Reinforces that sharpness level matters |
| ICCV | *Balanced Sharpness-Aware Minimization for Imbalanced Regression* | Balances sharpness in target space | Represents task-structure-aware design |
| NeurIPS | *Momentum-SAM: Sharpness Aware Minimization without Computational Overhead* | Approximates perturbation from momentum | Relevant to low-overhead optimizer design |
| NeurIPS | *Unveiling m-Sharpness Through the Structure of Stochastic Gradient Noise* | Explains micro-batch sharpness through noise structure | Motivates batch/noise-aware designs |

The 2025 work emphasizes when to apply SAM, inexpensive substitutes for a
second gradient, and structure-aware behavior across data, models, and training
phases.

## Task-Relevant Hypotheses

This reference template uses ResNet-18, CIFAR-10/CIFAR-100, multiple seeds, wall-clock
overhead, and a clean optimizer interface. Heavy Hessian or distributed methods
are therefore less suitable than plug-in mechanisms.

High-value hypotheses include:

- **Phase adaptation:** use SGD or weak SAM early, then increase sharpness-aware
  pressure after representations stabilize.
- **Layerwise scaling:** allocate perturbation radius by parameter norm,
  gradient norm, layer size, or fan-in rather than one global radius.
- **Direction correction:** use EMA, projection, momentum, or bounded sampling
  to create a more stable low-overhead ascent direction.

These are initial literature-backed hypotheses, not required implementation
recipes. Peers should still explore alternatives and use task evidence to
accept or reject them.

## Evaluation Implications

Any claimed improvement should report:

- complete protocol accuracy across required datasets and seeds;
- wall-clock and compute overhead relative to SGD and standard SAM;
- phase-schedule ablation;
- layer-scaling ablation when applicable;
- perturbation-correction ablation when applicable;
- stability and invalid-run behavior.

Task reports should not call an implementation ASAM, Momentum-SAM, or another
named method unless the code matches the defining mechanism.

## Reference List

### 2024

- *Sharpness-Aware Minimization Enhances Feature Quality via Balanced Learning*
- *TRAM: Bridging Trust Regions and Sharpness Aware Minimization*
- *Domain-Inspired Sharpness-Aware Minimization Under Domain Shifts*
- *Friendly Sharpness-Aware Minimization*
- *A Universal Class of Sharpness-Aware Minimization Algorithms*
- *Improving SAM Requires Rethinking its Optimization Formulation*
- *Lookbehind-SAM: k steps back, 1 step forward*
- *Improving Sharpness-Aware Minimization by Lookahead*
- *On the Duality Between Sharpness-Aware Minimization and Adversarial Training*
- *CR-SAM: Curvature Regularized Sharpness-Aware Minimization*
- *Fundamental Convergence Analysis of Sharpness-Aware Minimization*
- *SAMPa: Sharpness-aware Minimization Parallelized*
- *Explicit Eigenvalue Regularization Improves Sharpness-Aware Minimization*
- *muP2: Effective Sharpness Aware Minimization Requires Layerwise Perturbation Scaling*
- *Implicit Regularization of Sharpness-Aware Minimization for Scale-Invariant Problems*
- *A Single-Step, Sharpness-Aware Minimization is All You Need to Achieve Efficient and Accurate Sparse Training*

### 2025

- *Sharpness-Aware Minimization Efficiently Selects Flatter Minima Late in Training*
- *Sharpness-Aware Minimization: General Analysis and Improved Rates*
- *Tilted Sharpness-Aware Minimization*
- *Focal-SAM: Focal Sharpness-Aware Minimization for Long-Tailed Classification*
- *One Arrow, Two Hawks: Sharpness-aware Minimization for Federated Learning via Global Model Trajectory*
- *Asymptotic Unbiased Sample Sampling to Speed Up Sharpness-Aware Minimization*
- *Critical Influence of Overparameterization on Sharpness-aware Minimization*
- *Beyond Local Sharpness: Communication-Efficient Global Sharpness-aware Minimization for Federated Learning*
- *Balanced Sharpness-Aware Minimization for Imbalanced Regression*
- *Momentum-SAM: Sharpness Aware Minimization without Computational Overhead*
- *Unveiling m-Sharpness Through the Structure of Stochastic Gradient Noise*
