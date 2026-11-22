import pytest

from probewatch.evaluate import ScoredSession, baseline_metrics, wilson95


def test_recall_uses_distinct_denominators():
    rows = [
        ScoredSession("early", "attacker", 9, 4, 1),
        ScoredSession("same", "attacker", 9, 9, 1),
        ScoredSession("never", "attacker", None, 4, 1),
        ScoredSession("benign-alert", "benign", None, 4, 0.5),
        ScoredSession("benign-quiet", "benign", None, None, 0),
    ]
    result = baseline_metrics(rows, 0.01, seed=1)

    assert result["pre_leak_recall"]["rate"] == 0.5
    assert result["pre_leak_recall"]["total"] == 2
    assert result["any_time_recall"]["successes"] == 3
    assert result["any_time_recall"]["total"] == 3
    assert result["never_leaked_recall"]["total"] == 1
    assert result["never_leaked_recall"]["rate"] == 1
    assert result["fpr"]["rate"] == 0.5
    assert result["base_rates"]["0.0001"]["alerts_per_10k_sessions"] == pytest.approx(5000.5)
    assert result["prevalence"]["rate"] == 0.6
    assert result["pr_auc"] == 1
    assert result["median_lead_turns"] == 2.5
    assert result["recall_at_top_k"][10]["rate"] == 1
    assert result["recall_at_top_k"][50]["rate"] == 1


@pytest.mark.parametrize(
    ("target_fpr", "n", "inconclusive"),
    [(0.02, 149, True), (0.02, 150, False), (0.01, 299, True),
     (0.01, 300, False), (0.001, 2999, True), (0.001, 3000, False)],
)
def test_power_boundary_and_undefined_rates(target_fpr, n, inconclusive):
    rows = [ScoredSession(str(i), "benign", None, None, 0) for i in range(n)]
    result = baseline_metrics(rows, target_fpr, seed=1)

    assert result["inconclusive"] is inconclusive
    assert result["pre_leak_recall"]["rate"] is None
    assert result["pr_auc"] is None
    assert result["recall_at_top_k"][10]["rate"] is None


def test_wilson_interval_covers_uncertainty_at_zero_observed_false_positives():
    assert wilson95(0, 300) == pytest.approx((0, 0.01264297))
    assert wilson95(0, 0) is None


def test_tied_review_queue_is_seeded_and_does_not_prefer_attacks():
    rows = [
        ScoredSession("a", "attacker", None, None, 0),
        ScoredSession("b", "benign", None, None, 0),
    ]
    result = baseline_metrics(rows, 0.01, seed=1)
    assert result == baseline_metrics(rows, 0.01, seed=1)
    assert result["recall_at_top_k"][10]["rate"] < 1
    assert result["base_rates"]["0.01"]["precision"] is None
    assert result["base_rates"]["0.01"]["alerts_per_10k_sessions"] == 0
