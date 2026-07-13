# Data contract

Explain the benchmark-compatible CSV contract. Prefer one row per experiment, replicate, and time point. Require `experiment_id` and `time_s`; support `replicate`, `temperature_K`, initial-condition columns `<species>0_mol_L`, and observed concentration columns `<species>_mol_L`. Objective columns such as `yield` may be used for CIGP. Describe accepted aliases but require user confirmation for ambiguous mapping or missing units. Never claim that a file has been parsed unless a deterministic parser result is present in context.
