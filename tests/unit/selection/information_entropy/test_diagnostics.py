"""Focused scientific closure diagnostics regression tests."""

import pytest

from nepflow.stages.selection.algorithms.information_entropy.diagnostics import (
    DiagnosticDistribution,
    recompute_final_entropy_metrics,
    weighted_quantile,
)


def test_weighted_quantile_is_stable_by_original_row_identity() -> None:
    values = [2.0, 1.0, 1.0, 3.0]
    weights = [0.1, 0.2, 0.4, 0.3]

    assert weighted_quantile(values, 0.5, weights, original_indices=[0, 1, 2, 3]) == 1.0
    assert weighted_quantile(values, 0.99, weights, original_indices=[0, 1, 2, 3]) == 3.0


def test_final_entropy_metrics_recompute_from_p_and_q() -> None:
    probabilities = [0.25, 0.25, 0.25, 0.25]
    final_q = [0.1, 0.2, 0.3, 0.4]
    result = recompute_final_entropy_metrics(
        probabilities,
        final_q,
        beta=1.0,
        selected_count=2,
    )

    assert result["target_mass"] == pytest.approx(1.0)
    assert result["q_mass"] == pytest.approx(1.0)
    assert result["support_mass"] == pytest.approx(3.0)
    assert result["forward_kl"] == pytest.approx(
        result["cross_entropy"] - result["shannon_entropy"]
    )


def test_distribution_manifest_retains_metric_contract() -> None:
    summary = DiagnosticDistribution(
        count=2,
        minimum=1.0,
        median=1.5,
        mean=1.5,
        q95=2.0,
        q99=2.0,
        maximum=2.0,
        unit="whitened Euclidean distance",
        population="all pool rows",
        weighting="equal-candidate-weighted p_i",
        denominator="sum of p_i",
    )

    restored = DiagnosticDistribution.from_manifest(summary.to_manifest())
    assert restored == summary
    assert restored.to_manifest()["quantile_convention"]
