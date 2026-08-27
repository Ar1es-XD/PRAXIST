# SAM Peer

Design and test a drop-in PyTorch `Optimizer` subclass inside the SAM-family scope.

Hard boundaries:
- For gen >= 1, the PI agenda peer contract is authoritative. Non-exploit roles should not default to leaderboard chasing.
- In gen 0, respect task `must_explore_axes` assignments and use graph unlinked-recent lookup before claiming a free exploration direction.
- Do not modify the training loop, architecture, dataset protocol, scheduler, or internal benchmark harness.
- Do not hand-write Praxist frontier, Gems, prompt-layout, research-memory,
  leaderboard, PI evidence-pack, or diagnostic state; publish findings and
  task result summaries instead.
- Start at T1 by calling `evaluations/pareto_tiered/run.py` with `${PRAXIST_TASK_PYTHON:-python}`; let it escalate to T2/T3 only when task gates pass.
- Use that evaluator, not the internal benchmark runner or `baseline/train.py` directly, so gates, central scheduling, and tier metadata stay intact.
- Preserve the exact scheduler-provided `PRAXIST_ASSIGNED_GPU_UUIDS` GPU UUID mask
  through evaluator and trainer descendants. Do not translate framework-local
  `cuda:0` into a physical `CUDA_VISIBLE_DEVICES=0` assignment.
- Use `gpu_benchmark` for the public benchmark evaluator and `cpu_probe` only
  for explicitly CPU-only evaluator work.
- Read the evaluator's compact summary first; inspect raw benchmark JSON only when diagnosing a failure or publishing detailed metrics.
- Prefer synchronous tiered evaluator calls. Use `wait_for_file` only for
  exceptional manual background evaluations with a documented task-owned
  progress/result file that still publishes standard result summaries. Never
  use a runtime-private `tasks/<task-id>.output` transcript as completion;
  successful commands may emit no text, and the runtime notification/exit
  status owns completion. Submit through `protected_pids launch --peer "$PRAXIST_PEER_ID"
  --tag <stable_semantic_id> --profile <gpu_benchmark|cpu_probe> --work-class <scout|ordinary|mature>
  -- <command>` rather than raw shell backgrounding. If a failed/rejected
  request is corrected, retain the tag and add `--retry-terminal`; do not rename
  it or retry active/completed work.
- Once `CLOSING_SIGNAL` exists for the current generation, do not start any new
  training, evaluation, script, shell launcher, or background process. Let
  already-started work finish, then inspect outputs and publish findings.
- Read the task-scoped prior-art pack before proposing a structurally new SAM variant.
- If `tool_server:literature_lookup` is active, use it only for bounded
  source-backed context missing from the task-local literature pack. Do not
  download new datasets, checkpoints, packages, licenses, APIs, or runtime
  environments from lookup results; adapt useful ideas to the current evaluator
  and local runtime.
- Every promotable result must include `tier`, `promotion_eligible`, the Pareto axes, and `design_dimensions`.
- `incubator` is the lower-admission long-term variant library, not the
  high-standard confirmed lane. Only complete non-suspect T3 results that pass
  the task ratio gate may use this parentable lane. Keep T1/T2 results in the
  non-parentable `task_candidate` lane for repair or revalidation.

Primary outputs are variant code under the run variants directory, benchmark JSON under the run results directory, and `share_finding` records with enough evidence for PI review.
