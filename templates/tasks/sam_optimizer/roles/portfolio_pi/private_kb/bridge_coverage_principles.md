# Bridge coverage principles

## When NOT to assign a bridge contract

1. The two anchors are already in the same family (e.g. WarmASAM rho=0.12
   bridged to WarmASAM rho=0.15 — that's a grid extension, not a bridge).
2. The bridge has been tested at ≥2 alpha / interpolation points in a
   prior generation. Re-running it produces no new information.
3. One anchor has been retired or marked obsolete. Bridging into a dead
   path wastes a peer.

## When to assign a bridge

1. Two anchors live on different Pareto arms (e.g. accuracy arm vs gap
   arm), and no prior bridge has tested the interaction.
2. Mechanism inheritance is testable (does mechanism_A's effect transfer
   when mechanism_B is added?).
3. The interaction has a quantifiable expected delta (≥0.5pp accuracy or
   ≥10% gap reduction).

Before assigning a bridge, query the coverage matrix. Do not repeat a covered
interaction unless the contract changes a meaningful protocol dimension or is
an explicit replication.

## Anti-mainline budget protection

If forbidden_mechanisms grows monotonically each gen, eventually all
viable mechanisms are forbidden. Two release rules:

1. After a mechanism is on forbidden list for ≥3 gens with no anti-mainline
   peer managing to use it, it can be released as "control mechanism" (not
   exploitable but allowed for ablation).
2. If anti-mainline produces 0 results for 2 consecutive gens, halve the
   forbidden list (drop the lowest-impact entries).
