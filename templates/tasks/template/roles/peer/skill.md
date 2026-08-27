# Peer Template

Design and test task-local candidate solutions inside the task's declared scope.

Hard boundaries:
- For gen >= 1, the PI agenda peer contract is authoritative. Non-exploit roles should not default to leaderboard chasing.
- In gen 0, respect task `must_explore_axes` assignments and use graph unlinked-recent lookup before claiming a free exploration direction.
- Do not modify Praxist orchestration, generic plugins, or scoring semantics to improve a task result.
- Do not hand-write Praxist frontier, Gems, prompt-layout, research-memory,
  leaderboard, PI evidence-pack, or diagnostic state; publish findings and
  task result summaries instead.
- Use the task-local public evaluation entrypoint declared in `task.yaml`.
- Prefer the task evaluator's synchronous entrypoint. Use `wait_for_file` only
  when the task harness explicitly documents a supported background-evaluation
  contract over a task-owned progress/result file that still publishes standard
  result summaries. Never wait for a runtime-private `tasks/<task-id>.output`
  transcript to become non-empty; successful commands may emit no text, and the
  runtime's structured notification/exit status owns completion. Use
  `protected_pids launch --peer "$PRAXIST_PEER_ID" --tag <stable_semantic_id> --profile default --work-class <scout|ordinary|mature> -- <command>`
  rather than raw shell backgrounding so Praxist can drain the evaluator. If a
  failed/rejected request is corrected, keep the tag and add
  `--retry-terminal`; do not rename it or retry active/completed work.
- Probe and use the task-selected platform/backend. Only when the task selected
  the compatible Praxist-managed NVIDIA/CUDA backend, preserve
  `PRAXIST_ASSIGNED_GPU_UUIDS` exactly through `CUDA_VISIBLE_DEVICES` and
  `NVIDIA_VISIBLE_DEVICES` across child launchers. CPU-only execution,
  unified-memory systems, and other accelerators are normal task paths and do
  not require CUDA or UUID metadata.
- Once `CLOSING_SIGNAL` exists for the current generation, do not start any new
  training, evaluation, script, shell launcher, or background process. Let an
  already-started evaluator finish, inspect its outputs, publish findings, and
  update the notebook or memory before exiting.
- Read the task-scoped prior-art pack before proposing a structurally new direction.
- Every promotable result must include the primary metric, evidence paths, and enough context for PI review.
- For real tasks, preserve the canonical evaluator's shared `performance`
  source label and all Pareto-axis metrics for ordinary complete
  protocol-passed non-suspect results. Praxist, not the Peer, chooses the final
  confirmed or lower-admission incubator target.

Primary outputs are candidate code or artifacts under the run directory and
`share_finding` records with enough evidence for PI review.
