# Task Audit Rules

This directory contains task-local audit criteria for the SAM reference template. The
default form is declarative text in `audit.yaml`; task authors should edit the
criteria rather than write Praxist framework code.

- `scope_and_tier/`: scope boundaries, tier metadata expectations, and
  promotion criteria.
- `pi_agenda/`: PI and Chair agenda expectations, including claim grounding and
  negative-evidence preservation.

Do not add Python files here for the default task path. If a task eventually
needs executable audit logic, treat it as an advanced task extension with its
own tests and documentation.
