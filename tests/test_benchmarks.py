from pc_mcmc_cigp.benchmarks import EpoxidationBenchmark, HBrMechanismBenchmark


def test_hbr_benchmark_smoke_returns_discovery_result():
    result = HBrMechanismBenchmark(random_state=1).run_discovery(n_steps=25, burn_in=5)

    assert result.posterior_inclusion_probabilities.ndim == 1
    assert result.diagnostics["n_samples"] == 20


def test_epoxidation_benchmark_smoke_runs_two_optimization_rounds():
    history = EpoxidationBenchmark(random_state=2).run_optimization(n_initial=3, n_iter=2)

    assert len(history) == 5
    assert history[-1]["best_yield"] >= 0.0


def test_epoxidation_benchmark_supports_explicit_acquisition_strategies():
    for strategy in ["PC_EI", "EI", "GWU", "DH", "UNCERTAINTY", "RANDOM"]:
        history = EpoxidationBenchmark(random_state=2).run_optimization(
            n_initial=3,
            n_iter=1,
            acquisition_name=strategy,
            n_candidates=16,
        )

        assert len(history) == 4
        assert history[-1]["acquisition"] == strategy
        assert "truth" in history[-1]
