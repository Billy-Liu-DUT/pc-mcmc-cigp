# Scientific guardrails

- Mark unmeasured intermediates and mechanistic steps as candidates.
- Enforce elemental and charge balance before sampling.
- Report pathway inclusion probability as posterior support, not proof.
- Require R-hat, ESS, invalid-evaluation, and predictive-error diagnostics.
- Separate poor convergence from structural mismatch and weak identifiability.
- Do not use endpoint yield alone to claim elementary pathways.
- Carry PC-MCMC parameter posteriors into CIGP as initialization or priors, not fixed truth.
- Mark recommendations outside the observed envelope as extrapolations.
- Return to mechanism discovery when new observations systematically contradict the compiled network.
