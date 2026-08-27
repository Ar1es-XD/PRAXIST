# Toy Math Conjecture Search

This resolve-only template describes a tiny non-ML research surface. It exists to
prove that Praxist task startup, plugin resolution, budget ledgers, replay, and
offline integration tests do not depend on a particular research domain.

## Research Surface

Agents would explore simple integer-sequence conjectures if this fixture were
expanded into a real task. Candidate findings should identify:

- the conjecture or counterexample pattern;
- the bounded search range used for checking;
- the deterministic score and any clarity or coverage metrics;
- the evidence needed for a future PI or Chair role to decide whether to keep,
  revise, or discard the direction.

## Fixture Boundary

The checked-in runner uses deterministic fake agents and does not prove real
theorems. The placeholder evaluator exits with an explanatory error. A real
math task should replace the evaluator, role skills, audit rules, and assets
while keeping the same external task-project boundary.

## Default Research Loop Policy

`task.yaml` keeps DIG on only for absolute gen0, gen0 QD on, and periodic Gems reset off because the default
fixture is resolve-only:

- Gen0 DIG/QD should generate and select diverse conjecture/search/proof ideas;
- later PI-synthesis QD is independently available but disabled in this
  one-generation fixture.
- Periodic Gems reset should be enabled only after an operator request or a
  diagnostic pass finds a performance ceiling and recommends a reset cadence.

The checked-in `max_generations: 1` is only for smoke testing.
