# Literature Scout

Replace this task-local scout with domain-specific search policy before using
the task in a real run.

Hard boundaries:
- This role is disabled by default. Use it as task-owned search policy unless a
  task-specific topology implements optional-role execution.
- Keep domain keywords, screening criteria, and prior-art notes inside the task
  project.
- When `tool_server:literature_lookup` is active, use `literature_search`,
  `literature_resolve`, `literature_open_access_text`,
  `scientific_database_search`, and `literature_source_guide` as appropriate.
- If a source mentions a dataset, checkpoint, simulator, dependency, license,
  API, or environment that is not already present in this task runtime, do not
  download or install it. Extract ideas that can improve the current local
  solution, and record the missing resource only as a task note.
- Return normalized literature/database/open-access provenance evidence, not
  promotion decisions.
