# Actor–Critic validation

Validated locally on Windows, Python 3.13, PyTorch 2.14.1+cpu, 2026-10-28 (Asia/Seoul).

- Full regression suite: 48 tests passed. Additional initial-policy parity / unservable-day test passed (49 distinct tests total).
- UI background start and safe day-boundary stop rechecked after selecting neural training as the UI default. Its completion wait was increased from 10 to 60 seconds to allow neural initialization and persistence; nonblocking-start assertions remain.
- Notebook code cells compile; GitHub summary upload remains opt-in.
- Actor and Critic parameters both change; a simple controlled reward fixture increases the probability of the rewarded weight.
- Personal cumulative reward differences sum to the unchanged daily evaluator.
- Fixed weights remain identical throughout each day. Daily optimization occurs once per training date; contextual optimization once per served customer transition.
- Same lambda episodes and visitor opportunities for both variants; no extra incumbent/candidate comparison calls.
- Frozen validation/evaluation leave parameters, optimizer, and RNG states unchanged.
- Exact equality between uninterrupted and checkpoint-resumed results, policy state, Adam state, and RNG state (timing excluded).
- Fixed probes, restored neural policies, line chart export, and absolute-day limits verified.
- Original 21 source-file hashes verified and preserved.

Local end-to-end pilot: 8 synthetic stores, 5 visitors/day, one utility scenario, 2 training + 1 validation + 1 evaluation dates. Finished day 4 with 320 result rows. JSON checkpoints, model snapshots, CSV reports, interactive charts, and PNG charts were generated; evaluation PNG visually inspected. This pilot used the implementation working tree before publication, so its metadata records the prior checkout commit plus the actual source hash. It is not a Colab execution and is not evidence that a neural model has completed 1000 days or improves real store outcomes.

Existing 1000-day random-search runs and their notebooks/checkpoints are not migrated or overwritten. Use a new experiment ID with the Actor–Critic notebook.
