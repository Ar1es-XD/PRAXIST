# Task Audit Rules

Task audit rules are declarative by default. Edit `audit.yaml` files to describe
scope boundaries, evidence standards, and PI/Chair agenda expectations in task
language.

- `scope_and_tier/`: task scope, evidence metadata, and promotion criteria.
- `pi_agenda/`: agenda shape, claim grounding, role-contract, and negative
  evidence expectations.

The template intentionally avoids Python audit hooks. Keep task authoring focused
on domain criteria unless an advanced task extension explicitly requires code and
ships its own tests.
