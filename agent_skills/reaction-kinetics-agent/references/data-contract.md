# Benchmark-compatible data contract

Use one row per `experiment_id`, optional `replicate`, and `time_s`.

Required for PC-MCMC:

- `experiment_id`
- `time_s`
- at least one `<species>_mol_L`
- initial state from the zero-time row or `<species>0_mol_L`

Optional condition fields include `replicate`, `temperature_K`, pressure, flow, residence time, catalyst loading, solvent, and other declared controllable variables. Convert explicit aliases such as minutes to seconds or Celsius to kelvin, while retaining the source mapping. Do not infer an unknown unit.

Compile each experiment-replicate group into `t`, `y0_full`, `data_matrix`, `obs_indices`, and optional `temperature`/`conditions`, matching the HBr benchmark. Compile CIGP rows into the selected template's ordered inputs and objective vector, matching the epoxidation benchmark.
