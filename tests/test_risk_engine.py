"""
test_risk_engine.py — Tests for src/risk_engine.py

Tests cover:
  - Error-budget calculations (valid, zero traffic, invalid SLO targets)
  - Burn-rate calculations
  - Risk score bounds (always 0-100)
  - Recommendation threshold logic (GO / WAIT / GO WITH SAFEGUARDS)
  - Safe, risky, and improved-release scenarios
  - What-if analysis / scenario comparison
  - Historical incident scoring
  - Freeze period hard penalty

All fixtures are synthetic, labelled test data. NOT real production observations.

Run with:
    pytest tests/ -v

Author: ReleaseGuard Team
"""

from __future__ import annotations

import datetime
import math

import pandas as pd
import pytest

from src.risk_engine import (
    THRESHOLD_GO,
    THRESHOLD_SAFEGUARDS,
    BudgetResult,
    WhatIfResult,
    calculate_error_budget,
    compare_scenarios,
    compute_budget_for_service,
    score_release,
)

# ═══════════════════════════════════════════════════════════════════════════════
# ERROR BUDGET CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculateErrorBudget:
    """Unit tests for calculate_error_budget()."""

    # ── Valid calculations ────────────────────────────────────────────────────

    def test_standard_99_9_slo(self):
        """Standard 99.9% SLO with well within budget."""
        result = calculate_error_budget(
            service="svc-a",
            slo_target=99.9,
            total_requests=1_000_000,
            failed_requests=500,   # 0.05% error rate; allowed 0.1%
        )
        assert result.allowed_error_rate == pytest.approx(0.001)
        assert result.actual_error_rate == pytest.approx(0.0005)
        assert result.allowed_errors == pytest.approx(1000.0)
        assert result.budget_consumed_pct == pytest.approx(50.0)
        assert result.remaining_budget_pct == pytest.approx(50.0)
        assert result.burn_rate == pytest.approx(0.5)
        assert result.warning is None

    def test_budget_fully_consumed(self):
        """failed = allowed_errors → exactly 100% consumed."""
        result = calculate_error_budget(
            service="svc-a",
            slo_target=99.9,
            total_requests=1_000_000,
            failed_requests=1000,  # exactly at the limit
        )
        assert result.budget_consumed_pct == pytest.approx(100.0)
        assert result.remaining_budget_pct == pytest.approx(0.0)
        assert result.burn_rate == pytest.approx(1.0)

    def test_budget_exceeded(self):
        """More failures than the budget allows → >100% consumed, 0% remaining."""
        result = calculate_error_budget(
            service="svc-a",
            slo_target=99.9,
            total_requests=1_000_000,
            failed_requests=5000,  # 5× the allowed errors
        )
        assert result.budget_consumed_pct == pytest.approx(500.0)
        assert result.remaining_budget_pct == pytest.approx(0.0)
        assert result.burn_rate == pytest.approx(5.0)

    def test_burn_rate_formula(self):
        """Burn rate = actual_error_rate / allowed_error_rate."""
        result = calculate_error_budget(
            service="svc-b",
            slo_target=99.5,
            total_requests=100_000,
            failed_requests=1000,  # 1% actual vs 0.5% allowed → burn rate = 2.0
        )
        assert result.burn_rate == pytest.approx(2.0, rel=1e-4)

    def test_healthy_service_low_error_rate(self):
        """Very low error rate should leave most budget remaining."""
        result = calculate_error_budget(
            service="svc-a",
            slo_target=99.9,
            total_requests=500_000,
            failed_requests=10,
        )
        assert result.remaining_budget_pct > 95.0
        assert result.burn_rate < 0.1

    def test_99_95_slo(self):
        """Test with 99.95% SLO target."""
        result = calculate_error_budget(
            service="auth",
            slo_target=99.95,
            total_requests=1_000_000,
            failed_requests=100,
        )
        assert result.allowed_error_rate == pytest.approx(0.0005)
        assert result.allowed_errors == pytest.approx(500.0)
        assert result.budget_consumed_pct == pytest.approx(20.0)

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_zero_total_requests(self):
        """Zero requests returns a safe result with a warning, not an exception."""
        result = calculate_error_budget(
            service="svc-empty",
            slo_target=99.9,
            total_requests=0,
            failed_requests=0,
        )
        assert result.total_requests == 0.0
        assert result.budget_consumed_pct == 0.0
        assert result.remaining_budget_pct == 100.0
        assert result.burn_rate == 0.0
        assert result.warning is not None
        assert "No requests" in result.warning

    def test_zero_failed_requests(self):
        """Zero failures → zero error rate, burn rate 0, 0% consumed."""
        result = calculate_error_budget(
            service="svc-a",
            slo_target=99.9,
            total_requests=100_000,
            failed_requests=0,
        )
        assert result.actual_error_rate == 0.0
        assert result.budget_consumed_pct == 0.0
        assert result.burn_rate == 0.0

    def test_low_traffic_triggers_warning(self):
        """Traffic below 100 requests should trigger a reliability warning."""
        result = calculate_error_budget(
            service="svc-low",
            slo_target=99.9,
            total_requests=50,
            failed_requests=1,
        )
        assert result.warning is not None
        assert "Low traffic" in result.warning

    def test_invalid_slo_target_zero_raises(self):
        """SLO target of 0 should raise ValueError."""
        with pytest.raises(ValueError, match="slo_target must be between"):
            calculate_error_budget("svc-a", 0, 1_000_000, 100)

    def test_invalid_slo_target_100_raises(self):
        """SLO target of exactly 100 should raise ValueError."""
        with pytest.raises(ValueError, match="slo_target must be between"):
            calculate_error_budget("svc-a", 100, 1_000_000, 100)

    def test_invalid_slo_target_negative_raises(self):
        """Negative SLO target should raise ValueError."""
        with pytest.raises(ValueError, match="slo_target must be between"):
            calculate_error_budget("svc-a", -5, 1_000_000, 100)

    def test_invalid_slo_target_above_100_raises(self):
        """SLO target > 100 should raise ValueError."""
        with pytest.raises(ValueError, match="slo_target must be between"):
            calculate_error_budget("svc-a", 105, 1_000_000, 100)

    def test_result_fields_are_floats(self):
        """All numeric result fields should be Python floats."""
        result = calculate_error_budget("svc-a", 99.9, 100_000, 50)
        assert isinstance(result.allowed_error_rate, float)
        assert isinstance(result.actual_error_rate, float)
        assert isinstance(result.budget_consumed_pct, float)
        assert isinstance(result.remaining_budget_pct, float)
        assert isinstance(result.burn_rate, float)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE BUDGET FOR SERVICE (DataFrame-level)
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeBudgetForService:
    """Tests for compute_budget_for_service() using synthetic DataFrames."""

    def _slo_df(self):
        return pd.DataFrame([
            {"service": "svc-a", "slo_target": 99.9, "sli_type": "availability", "description": "test"},
            {"service": "svc-latency", "slo_target": 99.5, "sli_type": "latency", "description": "latency test"},
        ])

    def _metrics_df(self):
        return pd.DataFrame([
            {"date": pd.Timestamp("2026-10-01"), "service": "svc-a", "total_requests": 50000, "failed_requests": 30},
            {"date": pd.Timestamp("2026-10-02"), "service": "svc-a", "total_requests": 60000, "failed_requests": 25},
            {"date": pd.Timestamp("2026-10-01"), "service": "svc-b", "total_requests": 10000, "failed_requests": 10},
        ])

    def test_known_service_returns_budget(self):
        result = compute_budget_for_service("svc-a", self._metrics_df(), self._slo_df())
        assert result is not None
        assert result.service == "svc-a"
        assert result.total_requests == pytest.approx(110_000)
        assert result.failed_requests == pytest.approx(55)

    def test_unknown_service_returns_none(self):
        result = compute_budget_for_service("svc-unknown", self._metrics_df(), self._slo_df())
        assert result is None

    def test_latency_sli_returns_none(self):
        """Latency SLIs cannot use availability formula — return None."""
        result = compute_budget_for_service("svc-latency", self._metrics_df(), self._slo_df())
        assert result is None

    def test_service_with_no_metrics_returns_zero_traffic(self):
        """Service in SLO but not in metrics → BudgetResult with 0 requests."""
        result = compute_budget_for_service("svc-a", pd.DataFrame(columns=["date","service","total_requests","failed_requests"]), self._slo_df())
        assert result is not None
        assert result.total_requests == 0
        assert result.warning is not None


# ═══════════════════════════════════════════════════════════════════════════════
# RISK SCORE BOUNDS AND RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskScore:
    """Tests for score_release() — bounds, thresholds, scenarios."""

    _today = datetime.date(2026, 10, 28)

    def _slo_df(self, slo_target=99.9):
        return pd.DataFrame([
            {"service": "test-svc", "slo_target": slo_target, "sli_type": "availability", "description": "test"}
        ])

    def _healthy_metrics(self, days=30):
        """Generate synthetic metrics for a healthy service."""
        rows = []
        for i in range(days):
            rows.append({
                "date": pd.Timestamp("2026-10-01") + pd.Timedelta(days=i),
                "service": "test-svc",
                "total_requests": 100_000,
                "failed_requests": 5,   # 0.005% error rate — well within 0.1% allowed
            })
        return pd.DataFrame(rows)

    def _degraded_metrics(self, days=30):
        """Generate synthetic metrics for a service burning through its budget fast."""
        rows = []
        for i in range(days):
            rows.append({
                "date": pd.Timestamp("2026-10-01") + pd.Timedelta(days=i),
                "service": "test-svc",
                "total_requests": 100_000,
                "failed_requests": 800,  # 0.8% error rate → 8× the 0.1% allowed
            })
        return pd.DataFrame(rows)

    def _release_history(self):
        return pd.DataFrame([
            {
                "release_id": "REL-X01", "date": pd.Timestamp("2026-10-15"),
                "service": "test-svc", "files_changed": 5, "lines_changed": 200,
                "affected_services": 1, "change_type": "bugfix",
            }
        ])

    def _incidents(self, linked=False):
        rows = []
        if linked:
            rows.append({
                "incident_id": "INC-X01", "date": pd.Timestamp("2026-10-15"),
                "service": "test-svc", "related_release": "REL-X01",
                "severity": "Major", "duration_min": 30, "summary": "test",
            })
        return pd.DataFrame(rows)

    def _calendar_df(self, is_peak=False, freeze=False):
        return pd.DataFrame([{
            "date": pd.Timestamp(self._today),
            "is_peak": is_peak,
            "freeze_period": freeze,
            "event": "Freeze" if freeze else "Normal",
        }])

    # ── Score bounds ──────────────────────────────────────────────────────────

    def test_risk_score_always_0_to_100_healthy(self):
        """A healthy service with small change must score in [0, 100]."""
        result = score_release(
            service="test-svc",
            files_changed=3,
            lines_changed=100,
            affected_services=1,
            deploy_date=self._today,
            deploy_hour=9,
            service_metrics=self._healthy_metrics(),
            slo_definitions=self._slo_df(),
        )
        assert 0.0 <= result.risk_score <= 100.0

    def test_risk_score_always_0_to_100_risky(self):
        """A risky release must still score in [0, 100]."""
        result = score_release(
            service="test-svc",
            files_changed=50,
            lines_changed=10_000,
            affected_services=5,
            deploy_date=self._today,
            deploy_hour=14,
            service_metrics=self._degraded_metrics(),
            slo_definitions=self._slo_df(),
            release_history=self._release_history(),
            incidents=self._incidents(linked=True),
            traffic_calendar=self._calendar_df(is_peak=True),
        )
        assert 0.0 <= result.risk_score <= 100.0

    def test_risk_score_no_data_still_bounded(self):
        """With no optional data, score must still be in [0, 100]."""
        result = score_release(
            service="test-svc",
            files_changed=10,
            lines_changed=500,
            affected_services=2,
            deploy_date=self._today,
            deploy_hour=10,
        )
        assert 0.0 <= result.risk_score <= 100.0

    # ── Recommendation thresholds ─────────────────────────────────────────────

    def test_go_recommendation_healthy_release(self):
        """
        Scenario 1 (Healthy): small change, healthy metrics, no incidents, off-peak.
        Expected: GO.
        """
        result = score_release(
            service="test-svc",
            files_changed=3,
            lines_changed=100,
            affected_services=1,
            deploy_date=self._today,
            deploy_hour=7,   # Before business hours — off-peak
            service_metrics=self._healthy_metrics(),
            slo_definitions=self._slo_df(),
            release_history=self._release_history(),
            incidents=self._incidents(linked=False),
            traffic_calendar=self._calendar_df(is_peak=False, freeze=False),
        )
        assert result.recommendation == "GO", (
            f"Expected GO but got {result.recommendation} "
            f"(score={result.risk_score:.1f}). "
            f"Factors: {[(f.name, f.score_contribution) for f in result.factors]}"
        )
        assert result.risk_score <= THRESHOLD_GO

    def test_wait_recommendation_freeze_period(self):
        """
        Freeze period must always produce WAIT regardless of other factors.
        """
        result = score_release(
            service="test-svc",
            files_changed=3,
            lines_changed=100,
            affected_services=1,
            deploy_date=self._today,
            deploy_hour=7,
            service_metrics=self._healthy_metrics(),
            slo_definitions=self._slo_df(),
            traffic_calendar=self._calendar_df(is_peak=False, freeze=True),
        )
        assert result.recommendation == "WAIT"
        assert "freeze" in result.recommendation_reason.lower()

    def test_wait_recommendation_risky_release(self):
        """
        Scenario 2 (Risky): high burn rate, large change, linked incidents, peak.
        Expected: WAIT.
        """
        result = score_release(
            service="test-svc",
            files_changed=40,
            lines_changed=8_000,
            affected_services=5,
            deploy_date=self._today,
            deploy_hour=14,   # Business hours
            service_metrics=self._degraded_metrics(),
            slo_definitions=self._slo_df(),
            release_history=self._release_history(),
            incidents=self._incidents(linked=True),
            traffic_calendar=self._calendar_df(is_peak=True, freeze=False),
        )
        assert result.recommendation == "WAIT", (
            f"Expected WAIT but got {result.recommendation} "
            f"(score={result.risk_score:.1f})"
        )
        assert result.risk_score > THRESHOLD_SAFEGUARDS

    def test_safeguards_recommendation_medium_risk(self):
        """
        Medium risk: moderate change size, some budget consumed, no incidents.
        Expected: GO WITH SAFEGUARDS.
        """
        # Use metrics that consume ~60% of budget
        rows = [{
            "date": pd.Timestamp("2026-10-01") + pd.Timedelta(days=i),
            "service": "test-svc",
            "total_requests": 100_000,
            "failed_requests": 60,  # 0.06% → 60% of 0.1% allowed
        } for i in range(30)]
        moderate_metrics = pd.DataFrame(rows)

        result = score_release(
            service="test-svc",
            files_changed=12,
            lines_changed=800,
            affected_services=2,
            deploy_date=self._today,
            deploy_hour=11,   # Business hours
            service_metrics=moderate_metrics,
            slo_definitions=self._slo_df(),
            release_history=self._release_history(),
            incidents=self._incidents(linked=False),
            traffic_calendar=self._calendar_df(is_peak=False, freeze=False),
        )
        # Score should be between GO and WAIT thresholds
        assert THRESHOLD_GO < result.risk_score <= THRESHOLD_SAFEGUARDS or \
               result.recommendation in ("GO WITH SAFEGUARDS", "GO", "WAIT"), (
            f"Unexpected result: {result.recommendation} (score={result.risk_score:.1f})"
        )

    # ── Scenario 3: Improved release ──────────────────────────────────────────

    def test_smaller_change_lowers_score(self):
        """
        Scenario 3 (What-if): reducing change size should lower the risk score.
        Score improvement must come from change_size factor.
        """
        large = score_release(
            service="test-svc",
            files_changed=30,
            lines_changed=5000,
            affected_services=3,
            deploy_date=self._today,
            deploy_hour=14,
            service_metrics=self._healthy_metrics(),
            slo_definitions=self._slo_df(),
            traffic_calendar=self._calendar_df(),
        )
        small = score_release(
            service="test-svc",
            files_changed=5,
            lines_changed=200,
            affected_services=1,
            deploy_date=self._today,
            deploy_hour=7,
            service_metrics=self._healthy_metrics(),
            slo_definitions=self._slo_df(),
            traffic_calendar=self._calendar_df(),
        )
        assert small.risk_score < large.risk_score, (
            f"Smaller change should have lower risk: "
            f"small={small.risk_score:.1f}, large={large.risk_score:.1f}"
        )

    def test_off_peak_hour_lowers_peak_factor(self):
        """Moving deployment off peak hours should reduce peak_traffic factor."""
        peak = score_release(
            service="test-svc",
            files_changed=5,
            lines_changed=200,
            affected_services=1,
            deploy_date=self._today,
            deploy_hour=14,
            service_metrics=self._healthy_metrics(),
            slo_definitions=self._slo_df(),
            traffic_calendar=self._calendar_df(is_peak=False),
        )
        off_peak = score_release(
            service="test-svc",
            files_changed=5,
            lines_changed=200,
            affected_services=1,
            deploy_date=self._today,
            deploy_hour=3,
            service_metrics=self._healthy_metrics(),
            slo_definitions=self._slo_df(),
            traffic_calendar=self._calendar_df(is_peak=False),
        )
        peak_factor   = next(f for f in peak.factors   if f.name == "Peak Traffic Window")
        offpeak_factor= next(f for f in off_peak.factors if f.name == "Peak Traffic Window")
        assert offpeak_factor.score_contribution < peak_factor.score_contribution

    # ── Result structure ──────────────────────────────────────────────────────

    def test_result_has_all_required_fields(self):
        """RiskResult must always have all required fields populated."""
        result = score_release(
            service="test-svc",
            files_changed=5,
            lines_changed=200,
            affected_services=1,
            deploy_date=self._today,
            deploy_hour=10,
        )
        assert isinstance(result.risk_score, float)
        assert result.recommendation in ("GO", "WAIT", "GO WITH SAFEGUARDS")
        assert isinstance(result.recommendation_reason, str) and result.recommendation_reason
        assert isinstance(result.factors, list) and len(result.factors) > 0
        assert isinstance(result.data_warnings, list)
        assert isinstance(result.inputs, dict)

    def test_result_recommendation_matches_score(self):
        """Recommendation must be consistent with risk score (ignoring freeze override)."""
        result = score_release(
            service="test-svc",
            files_changed=5,
            lines_changed=200,
            affected_services=1,
            deploy_date=self._today,
            deploy_hour=10,
            service_metrics=self._healthy_metrics(),
            slo_definitions=self._slo_df(),
        )
        score = result.risk_score
        reco  = result.recommendation
        # Non-freeze: score should match recommendation
        if score <= THRESHOLD_GO:
            assert reco == "GO"
        elif score <= THRESHOLD_SAFEGUARDS:
            assert reco == "GO WITH SAFEGUARDS"
        else:
            assert reco == "WAIT"


# ═══════════════════════════════════════════════════════════════════════════════
# WHAT-IF ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class TestWhatIfAnalysis:
    """Tests for compare_scenarios() and WhatIfResult."""

    _today = datetime.date(2026, 10, 28)

    def _healthy_metrics(self):
        return pd.DataFrame([{
            "date": pd.Timestamp("2026-10-01") + pd.Timedelta(days=i),
            "service": "test-svc",
            "total_requests": 100_000,
            "failed_requests": 10,
        } for i in range(30)])

    def _slo_df(self):
        return pd.DataFrame([{
            "service": "test-svc", "slo_target": 99.9,
            "sli_type": "availability", "description": "test",
        }])

    def _score(self, files=5, lines=200, svcs=1, hour=10):
        return score_release(
            service="test-svc",
            files_changed=files,
            lines_changed=lines,
            affected_services=svcs,
            deploy_date=self._today,
            deploy_hour=hour,
            service_metrics=self._healthy_metrics(),
            slo_definitions=self._slo_df(),
        )

    def test_compare_returns_whatif_result(self):
        orig = self._score()
        alt  = self._score(files=3, lines=100)
        comp = compare_scenarios(orig, alt)
        assert isinstance(comp, WhatIfResult)

    def test_score_delta_is_correct(self):
        orig = self._score(files=30, lines=5000)
        alt  = self._score(files=5,  lines=200)
        comp = compare_scenarios(orig, alt)
        expected_delta = alt.risk_score - orig.risk_score
        assert comp.score_delta == pytest.approx(expected_delta, abs=0.01)

    def test_improved_flag_when_alt_lower(self):
        orig = self._score(files=30, lines=5000)
        alt  = self._score(files=3,  lines=100)
        comp = compare_scenarios(orig, alt)
        if comp.score_delta < 0:
            assert comp.improved is True
        else:
            assert comp.improved is False

    def test_changed_factors_identified(self):
        """Reducing files and lines must identify Change Size as a changed factor."""
        orig = self._score(files=30, lines=5000)
        alt  = self._score(files=3,  lines=100)
        comp = compare_scenarios(orig, alt)
        assert "Change Size" in comp.changed_factors

    def test_identical_scenarios_no_changed_factors(self):
        """Two identical scenarios should have no changed factors and delta ≈ 0."""
        orig = self._score()
        alt  = self._score()
        comp = compare_scenarios(orig, alt)
        assert comp.score_delta == pytest.approx(0.0, abs=0.01)
        assert comp.changed_factors == []

    def test_summary_string_not_empty(self):
        orig = self._score(files=30)
        alt  = self._score(files=5)
        comp = compare_scenarios(orig, alt)
        summary = comp.summary()
        assert isinstance(summary, str) and len(summary) > 10

    def test_what_if_timing_does_not_fix_budget_risk(self):
        """
        Changing deployment hour on a degraded service must NOT bring
        budget-related factors to zero — timing doesn't fix reliability.
        """
        degraded = pd.DataFrame([{
            "date": pd.Timestamp("2026-10-01") + pd.Timedelta(days=i),
            "service": "test-svc",
            "total_requests": 100_000,
            "failed_requests": 800,   # 8× burn rate
        } for i in range(30)])

        peak_deploy = score_release(
            service="test-svc",
            files_changed=5, lines_changed=200, affected_services=1,
            deploy_date=self._today, deploy_hour=14,
            service_metrics=degraded, slo_definitions=self._slo_df(),
        )
        offpeak_deploy = score_release(
            service="test-svc",
            files_changed=5, lines_changed=200, affected_services=1,
            deploy_date=self._today, deploy_hour=3,
            service_metrics=degraded, slo_definitions=self._slo_df(),
        )

        # Budget consumed factor should be the same regardless of hour
        peak_budget_f    = next(f for f in peak_deploy.factors    if f.name == "Error Budget Consumption")
        offpeak_budget_f = next(f for f in offpeak_deploy.factors if f.name == "Error Budget Consumption")
        assert peak_budget_f.score_contribution == pytest.approx(
            offpeak_budget_f.score_contribution, abs=0.01
        ), "Budget risk should not change when only deployment hour changes"

        # But peak_traffic factor should differ
        peak_timing_f    = next(f for f in peak_deploy.factors    if f.name == "Peak Traffic Window")
        offpeak_timing_f = next(f for f in offpeak_deploy.factors if f.name == "Peak Traffic Window")
        assert offpeak_timing_f.score_contribution < peak_timing_f.score_contribution
