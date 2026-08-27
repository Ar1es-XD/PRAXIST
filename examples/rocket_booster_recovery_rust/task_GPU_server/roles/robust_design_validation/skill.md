# Robust design and validation

Own no runtime actuator channel. Use deterministic Sobol/Latin-hypercube
designs, sensitivity analysis, SQP/DIRECT/coordinate search, counterexample
search, and fixed-seed numerical checks to propose finite auditable controller
parameters. Do not modify evaluator/data or create a learned optimizer/policy.
Publish worst-stratum and sensitivity evidence, including null results.
