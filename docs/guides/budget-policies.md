# Budget Policies

Budget is dynamic in Praxist. It is not a fixed tuple copied once into a run and then
blindly enforced for every experiment.

## Budget Requests

Agents and workflow stages may request budget in the units accepted by the
current core validator:

- `tokens`;
- `wall_clock_seconds`;
- `gpu_hours`.

Requests should include scope, reason, estimated cost, expected value, and the
action that will consume the budget.

## Policy Decisions

A BudgetPolicy may:

- auto-grant low-risk requests;
- downscope a request;
- ask a Principal Investigator (PI) or Chair planning agent to review unusually
  large requests;
- deny requests that would damage the run or exceed operator limits.

The default posture is result preservation. A promising experiment should be
allowed to finish when it is inside a reasonable envelope and can produce useful
artifacts, even if exact metering is imperfect.

## Usage Records

Usage records can be exact, estimated, partial, or unknown. Unknown usage must be
recorded explicitly as `usage_unknown` instead of `0`.

Late accounting failure should warn and preserve findings when possible.

## Budget Tests

Budget policy tests should cover grant, deny, downscope, review routing,
usage_unknown, replay visibility, and behavior when a peer produces results before
usage accounting completes.
