# Architecture

Dependency direction is one-way:

`CLI -> Application -> Experiments/Protocol/Training -> Data/Scoring -> Core`

The core domain never imports the CLI, filesystem configuration, experiment executor, or report generation code.

## Protocol responsibilities

The operating-point protocol is split into four responsibilities:

1. `ReferenceThresholdEstimator`: creates the federation reference operating point.
2. `CalibrationReadinessEvaluator`: determines whether independent local benign calibration data can support a replacement threshold with the configured assurance.
3. `ReferenceMismatchEvaluator`: determines whether independent benign evidence demonstrates material mismatch of the reference operating point.
4. `ThresholdDecisionEngine`: combines the typed evidence into exactly one deployment decision.

The decision rules exist only in `protocol/decision.py`. Other layers consume its result and must not duplicate the state machine.

## Class boundaries

Classes represent domain entities or stateful services. Closed identities use enums. Statistical formulas remain pure functions. `dict[str, Any]` is restricted to serialization and external-adapter boundaries.

## Experiment lifecycle

`PENDING -> VALIDATING -> READY -> RUNNING -> VERIFYING -> COMPLETE`

Any execution error transitions to `FAILED`. A failed prerequisite causes dependants to become `BLOCKED`; it never satisfies a dependency.
