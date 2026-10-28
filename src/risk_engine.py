"""
risk_engine.py — ReleaseGuard Release Risk Scoring Engine

Implements:
  1. SLO / error-budget calculations (availability-based only).
  2. A transparent, heuristic-weighted release risk scorer.
  3. A what-if comparator.

All weights and thresholds are explicit constants at the top of this module
so they can be reviewed, tuned, and tested without touching logic.

⚠  The heuristic risk score is NOT a calibrated failure probability.
   It is a transparent, explainable decision-support signal.

Dataset note: all underlying data is synthetic hackathon data, not real
production observations.

Author: ReleaseGuard Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

# ── Configurable weights and thresholds ───────────────────────────────────────
# Each weight is a maximum contribution to the 0-100 risk score.
# Adjust these values to match your organisation's risk appetite.

WEIGHTS: dict[str, float] = {
    "budget_consumed":    25.0,   # Error-budget consumption %
    "burn_rate":          20.0,   # Error-budget burn rate
    "historical_incidents": 20.0, # Past incidents associated with releases on this service
    "change_size":        15.0,   # Files + lines changed
    "affected_services":  10.0,   # Number of services touched
    "peak_traffic":        5.0,   # Deploying during peak traffic
    "freeze_period":      15.0,   # Deploying during a freeze (hard penalty)
    "recent_instability":  10.0,  # Service instability in last 7 days
}

# Decision thresholds (score out of 100)
THRESHOLD_GO            = 35   # score ≤ this → GO
THRESHOLD_SAFEGUARDS    = 65   # score ≤ this → GO WITH SAFEGUARDS (else WAIT)

# Budget thresholds
BUDGET_WARN  = 50.0    # % consumed — moderate concern
BUDGET_HIGH  = 80.0    # % consumed — high concern

# Burn-rate thresholds
BURN_WARN    = 2.0     # moderately elevated
BURN_HIGH    = 5.0     # seriously elevated

# Peak-traffic deployment hour window (24h clock, inclusive)
PEAK_HOUR_START = 9
PEAK_HOUR_END   = 18

# Change-size bands (files changed)
CHANGE_FILES_MEDIUM = 10
CHANGE_FILES_LARGE  = 20

# Lines-changed bands
CHANGE_LINES_MEDIUM = 500
CHANGE_LINES_LARGE  = 2000

# Recent instability window (days)
INSTABILITY_WINDOW_DAYS = 7

# Historical incident lookback (days)
INCIDENT_LOOKBACK_DAYS = 30


# ── SLO / error-budget helpers ────────────────────────────────────────────────


@dataclass
class BudgetResult:
    """Structured result of an error-budget calculation."""

    service: str
    slo_target: float               # e.g. 99.9 (percent)
    window_days: int

    total_requests: float
    failed_requests: float

    allowed_error_rate: float       # = 1 - slo_target/100
    actual_error_rate: float        # = failed / total

    allowed_errors: float           # = total * allowed_error_rate
    budget_consumed_pct: float      # (failed / allowed_errors) * 100, clamped ≥ 0
    remaining_budget_pct: float     # max(0, 100 - consumed)
    burn_rate: float                # actual_error_rate / allowed_error_rate

    warning: Optional[str] = None   # Set when calculation has caveats


def calculate_error_budget(
    service: str,
    slo_target: float,
    total_requests: float,
    failed_requests: float,
    window_days: int = 30,
) -> BudgetResult:
    """Calculate error-budget metrics for an availability-based SLO.

    Parameters
    ----------
    service:
        Service name (used in result labelling only).
    slo_target:
        SLO target as a percentage, e.g. 99.9 (NOT 0.999).
    total_requests:
        Total requests in the measurement window.
    failed_requests:
        Failed requests in the measurement window.
    window_days:
        Number of days the window covers (informational).

    Returns
    -------
    BudgetResult
        All intermediate values plus warnings where applicable.

    Raises
    ------
    ValueError
        Only for completely invalid arguments (slo_target outside 0–100).
        Missing/zero traffic is handled gracefully, not raised.
    """
    if not (0 < slo_target < 100):
        raise ValueError(
            f"slo_target must be between 0 and 100 exclusive; got {slo_target}"
        )

    warning: Optional[str] = None

    # Guard zero traffic
    if total_requests <= 0:
        warning = (
            "No requests recorded in this window. "
            "Error-budget calculations require at least one request. "
            "Score is based on zero traffic — treat with caution."
        )
        return BudgetResult(
            service=service,
            slo_target=slo_target,
            window_days=window_days,
            total_requests=0.0,
            failed_requests=0.0,
            allowed_error_rate=1 - slo_target / 100,
            actual_error_rate=0.0,
            allowed_errors=0.0,
            budget_consumed_pct=0.0,
            remaining_budget_pct=100.0,
            burn_rate=0.0,
            warning=warning,
        )

    # Clamp failed_requests
    failed_requests = max(0.0, float(failed_requests))
    total_requests  = float(total_requests)

    allowed_error_rate  = 1.0 - (slo_target / 100.0)
    actual_error_rate   = failed_requests / total_requests
    allowed_errors      = total_requests * allowed_error_rate

    if allowed_errors <= 0:
        # Theoretically possible only if slo_target == 100 (blocked by guard above)
        budget_consumed_pct = 100.0 if failed_requests > 0 else 0.0
        burn_rate = float("inf") if failed_requests > 0 else 0.0
        warning = "allowed_errors is effectively zero; budget calculation is degenerate."
    else:
        budget_consumed_pct = (failed_requests / allowed_errors) * 100.0
        burn_rate = actual_error_rate / allowed_error_rate

    remaining_budget_pct = max(0.0, 100.0 - budget_consumed_pct)

    if total_requests < 100:
        warn_msg = (
            f"Low traffic: only {total_requests:.0f} requests in window. "
            "Budget percentages are unreliable at low request volumes."
        )
        warning = (warning + " | " + warn_msg) if warning else warn_msg

    return BudgetResult(
        service=service,
        slo_target=slo_target,
        window_days=window_days,
        total_requests=total_requests,
        failed_requests=failed_requests,
        allowed_error_rate=allowed_error_rate,
        actual_error_rate=actual_error_rate,
        allowed_errors=allowed_errors,
        budget_consumed_pct=budget_consumed_pct,
        remaining_budget_pct=remaining_budget_pct,
        burn_rate=burn_rate,
        warning=warning,
    )


def compute_budget_for_service(
    service: str,
    service_metrics: pd.DataFrame,
    slo_definitions: pd.DataFrame,
    window_days: int = 30,
) -> Optional[BudgetResult]:
    """Compute error budget for a service from DataFrames.

    Returns None if the service is not found in SLO definitions.
    Aggregates daily metrics over the most recent `window_days` days.
    """
    slo_row = slo_definitions[slo_definitions["service"] == service]
    if slo_row.empty:
        return None

    slo_target = float(slo_row.iloc[0]["slo_target"])
    sli_type   = slo_row.iloc[0].get("sli_type", "availability")

    if sli_type != "availability":
        # Cannot apply availability formula to latency SLOs without thresholds
        return None

    svc_metrics = service_metrics[service_metrics["service"] == service].copy()
    svc_metrics = svc_metrics.sort_values("date")

    if not svc_metrics.empty and window_days > 0:
        cutoff = svc_metrics["date"].max() - pd.Timedelta(days=window_days - 1)
        svc_metrics = svc_metrics[svc_metrics["date"] >= cutoff]

    total   = svc_metrics["total_requests"].sum()
    failed  = svc_metrics["failed_requests"].sum()
    actual_window = svc_metrics["date"].nunique()

    return calculate_error_budget(
        service=service,
        slo_target=slo_target,
        total_requests=total,
        failed_requests=failed,
        window_days=actual_window,
    )


# ── Risk engine ───────────────────────────────────────────────────────────────


@dataclass
class RiskFactor:
    """One contributing factor in the risk score."""

    name: str
    score_contribution: float  # Points added to risk score (0–weight_max)
    weight_max: float          # Maximum possible contribution
    evidence: str              # Human-readable evidence
    raw_value: Any = None      # The underlying measured value


@dataclass
class RiskResult:
    """Full output of the risk engine for one proposed release."""

    # Core result
    risk_score: float                      # 0–100
    recommendation: str                    # "GO" | "WAIT" | "GO WITH SAFEGUARDS"
    recommendation_reason: str

    # Breakdown
    factors: list[RiskFactor] = field(default_factory=list)

    # Safeguard suggestions
    suggested_safeguards: list[str] = field(default_factory=list)

    # Data quality
    data_warnings: list[str] = field(default_factory=list)

    # Budget (may be None if not calculable)
    budget: Optional[BudgetResult] = None

    # Input snapshot (for what-if display)
    inputs: dict[str, Any] = field(default_factory=dict)


def _score_budget_consumed(budget: Optional[BudgetResult]) -> RiskFactor:
    """Score contribution from error-budget consumption."""
    w = WEIGHTS["budget_consumed"]
    if budget is None:
        return RiskFactor(
            name="Error Budget Consumption",
            score_contribution=w * 0.5,  # Unknown → moderate risk assumption
            weight_max=w,
            evidence="No budget data available. Assuming moderate risk (50% of weight).",
        )
    consumed = budget.budget_consumed_pct
    if consumed >= 100:
        fraction = 1.0
        evidence = f"Budget fully consumed ({consumed:.1f}%). Service is at or past its SLO limit."
    elif consumed >= BUDGET_HIGH:
        fraction = 0.8
        evidence = f"Budget {consumed:.1f}% consumed (≥{BUDGET_HIGH}% threshold). High risk."
    elif consumed >= BUDGET_WARN:
        fraction = 0.5
        evidence = f"Budget {consumed:.1f}% consumed (≥{BUDGET_WARN}% threshold). Moderate risk."
    else:
        fraction = consumed / 100.0 * 0.3  # Low consumption → low score
        evidence = f"Budget {consumed:.1f}% consumed. Healthy."

    return RiskFactor(
        name="Error Budget Consumption",
        score_contribution=w * fraction,
        weight_max=w,
        evidence=evidence,
        raw_value=consumed,
    )


def _score_burn_rate(budget: Optional[BudgetResult]) -> RiskFactor:
    """Score contribution from error-budget burn rate."""
    w = WEIGHTS["burn_rate"]
    if budget is None:
        return RiskFactor(
            name="Error Budget Burn Rate",
            score_contribution=w * 0.5,
            weight_max=w,
            evidence="No budget data available. Assuming moderate risk.",
        )
    burn = budget.burn_rate
    if math.isinf(burn) or burn >= BURN_HIGH:
        fraction = 1.0
        evidence = f"Burn rate {burn:.2f}× (≥{BURN_HIGH}×). Budget burning very fast."
    elif burn >= BURN_WARN:
        fraction = 0.6
        evidence = f"Burn rate {burn:.2f}× (≥{BURN_WARN}×). Elevated consumption."
    elif burn >= 1.0:
        fraction = 0.3
        evidence = f"Burn rate {burn:.2f}× (>1× means budget is being consumed)."
    else:
        fraction = 0.0
        evidence = f"Burn rate {burn:.2f}× — currently better than SLO target."

    return RiskFactor(
        name="Error Budget Burn Rate",
        score_contribution=w * fraction,
        weight_max=w,
        evidence=evidence,
        raw_value=burn,
    )


def _score_historical_incidents(
    service: str,
    release_history: Optional[pd.DataFrame],
    incidents: Optional[pd.DataFrame],
    lookback_days: int = INCIDENT_LOOKBACK_DAYS,
) -> RiskFactor:
    """Score based on past incidents associated with releases on this service."""
    w = WEIGHTS["historical_incidents"]

    if release_history is None or incidents is None:
        return RiskFactor(
            name="Historical Release Incidents",
            score_contribution=w * 0.3,
            weight_max=w,
            evidence="Incident or release history unavailable. Moderate risk assumed.",
        )

    # Guard: empty incidents DataFrame (no rows or missing required column)
    if incidents.empty or "related_release" not in incidents.columns:
        return RiskFactor(
            name="Historical Release Incidents",
            score_contribution=0.0,
            weight_max=w,
            evidence="No incident records available to link to releases.",
            raw_value=0,
        )

    # Releases for this service in the lookback window
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
    svc_releases = release_history[
        (release_history["service"] == service)
        & (release_history["date"] >= cutoff)
    ]

    if svc_releases.empty:
        return RiskFactor(
            name="Historical Release Incidents",
            score_contribution=0.0,
            weight_max=w,
            evidence=f"No releases for '{service}' in the last {lookback_days} days.",
            raw_value=0,
        )

    release_ids = set(svc_releases["release_id"].dropna())
    linked_incidents = incidents[incidents["related_release"].isin(release_ids)]

    total_releases  = len(svc_releases)
    incident_count  = len(linked_incidents)
    incident_rate   = incident_count / total_releases if total_releases > 0 else 0.0

    # Count critical vs major
    critical_count = len(linked_incidents[linked_incidents["severity"].str.lower() == "critical"])

    if incident_count == 0:
        fraction = 0.0
        evidence = (
            f"No incidents associated with the last {total_releases} release(s) "
            f"on '{service}' in the past {lookback_days} days."
        )
    elif critical_count > 0:
        fraction = min(1.0, 0.6 + 0.2 * critical_count)
        evidence = (
            f"{incident_count} incident(s) associated with {total_releases} recent release(s) "
            f"on '{service}' ({incident_rate:.0%} rate), including {critical_count} Critical. "
            f"Note: association ≠ causation."
        )
    else:
        fraction = min(1.0, 0.3 + 0.15 * incident_count)
        evidence = (
            f"{incident_count} incident(s) associated with {total_releases} recent release(s) "
            f"on '{service}' ({incident_rate:.0%} rate). Note: association ≠ causation."
        )

    return RiskFactor(
        name="Historical Release Incidents",
        score_contribution=w * fraction,
        weight_max=w,
        evidence=evidence,
        raw_value=incident_count,
    )


def _score_change_size(files_changed: int, lines_changed: int) -> RiskFactor:
    """Score based on size of the proposed change."""
    w = WEIGHTS["change_size"]

    # Score files
    if files_changed >= CHANGE_FILES_LARGE:
        file_fraction = 1.0
    elif files_changed >= CHANGE_FILES_MEDIUM:
        file_fraction = 0.6
    else:
        file_fraction = files_changed / CHANGE_FILES_MEDIUM * 0.4

    # Score lines
    if lines_changed >= CHANGE_LINES_LARGE:
        line_fraction = 1.0
    elif lines_changed >= CHANGE_LINES_MEDIUM:
        line_fraction = 0.6
    else:
        line_fraction = lines_changed / CHANGE_LINES_MEDIUM * 0.4

    fraction = max(file_fraction, line_fraction)

    if fraction >= 0.8:
        size_label = "Large"
    elif fraction >= 0.5:
        size_label = "Medium"
    else:
        size_label = "Small"

    evidence = (
        f"{size_label} change: {files_changed} file(s) changed, "
        f"{lines_changed:,} line(s) changed."
    )

    return RiskFactor(
        name="Change Size",
        score_contribution=w * fraction,
        weight_max=w,
        evidence=evidence,
        raw_value={"files": files_changed, "lines": lines_changed},
    )


def _score_affected_services(affected_services: int) -> RiskFactor:
    """Score based on how many services are affected."""
    w = WEIGHTS["affected_services"]

    if affected_services >= 4:
        fraction = 1.0
        evidence = f"{affected_services} services affected — high blast radius."
    elif affected_services >= 2:
        fraction = 0.5
        evidence = f"{affected_services} services affected — moderate blast radius."
    else:
        fraction = 0.0
        evidence = f"{affected_services} service(s) affected — contained change."

    return RiskFactor(
        name="Affected Services",
        score_contribution=w * fraction,
        weight_max=w,
        evidence=evidence,
        raw_value=affected_services,
    )


def _score_peak_traffic(
    deploy_date: date,
    deploy_hour: int,
    traffic_calendar: Optional[pd.DataFrame],
) -> RiskFactor:
    """Score based on whether deployment occurs during peak traffic."""
    w = WEIGHTS["peak_traffic"]

    is_calendar_peak = False
    is_business_hours = PEAK_HOUR_START <= deploy_hour <= PEAK_HOUR_END
    calendar_note = ""

    if traffic_calendar is not None:
        cal_row = traffic_calendar[
            traffic_calendar["date"].dt.date == deploy_date
        ]
        if not cal_row.empty:
            is_calendar_peak = bool(cal_row.iloc[0]["is_peak"])
            event = cal_row.iloc[0].get("event", "")
            calendar_note = f" Calendar event: '{event}'."

    is_peak = is_calendar_peak or is_business_hours

    if is_calendar_peak and is_business_hours:
        fraction = 1.0
        evidence = f"Deploying at {deploy_hour:02d}:00 during a calendar peak AND business hours.{calendar_note}"
    elif is_calendar_peak:
        fraction = 0.8
        evidence = f"Deploying at {deploy_hour:02d}:00 during a calendar-marked peak period.{calendar_note}"
    elif is_business_hours:
        fraction = 0.5
        evidence = (
            f"Deploying at {deploy_hour:02d}:00 during business hours "
            f"({PEAK_HOUR_START:02d}:00–{PEAK_HOUR_END:02d}:00)."
        )
    else:
        fraction = 0.0
        evidence = (
            f"Deploying at {deploy_hour:02d}:00 — outside business hours and no calendar peak.{calendar_note}"
        )

    return RiskFactor(
        name="Peak Traffic Window",
        score_contribution=w * fraction,
        weight_max=w,
        evidence=evidence,
        raw_value={"hour": deploy_hour, "is_peak": is_peak},
    )


def _score_freeze_period(
    deploy_date: date,
    traffic_calendar: Optional[pd.DataFrame],
) -> RiskFactor:
    """Score (hard penalty) for deploying during a freeze period."""
    w = WEIGHTS["freeze_period"]

    if traffic_calendar is None:
        return RiskFactor(
            name="Freeze Period",
            score_contribution=0.0,
            weight_max=w,
            evidence="No traffic calendar loaded; freeze period unknown.",
        )

    cal_row = traffic_calendar[traffic_calendar["date"].dt.date == deploy_date]

    if cal_row.empty:
        return RiskFactor(
            name="Freeze Period",
            score_contribution=0.0,
            weight_max=w,
            evidence=f"Date {deploy_date} not in traffic calendar; freeze status unknown.",
        )

    is_freeze = bool(cal_row.iloc[0]["freeze_period"])
    event = cal_row.iloc[0].get("event", "")

    if is_freeze:
        return RiskFactor(
            name="Freeze Period",
            score_contribution=w,  # Full penalty
            weight_max=w,
            evidence=f"⚠ Deployment date {deploy_date} falls within a freeze period ('{event}'). Full penalty applied.",
            raw_value=True,
        )
    else:
        return RiskFactor(
            name="Freeze Period",
            score_contribution=0.0,
            weight_max=w,
            evidence=f"Date {deploy_date} is not a freeze period.",
            raw_value=False,
        )


def _score_recent_instability(
    service: str,
    service_metrics: Optional[pd.DataFrame],
    slo_definitions: Optional[pd.DataFrame],
    window: int = INSTABILITY_WINDOW_DAYS,
) -> RiskFactor:
    """Score based on error rate volatility and elevated failures in recent days."""
    w = WEIGHTS["recent_instability"]

    if service_metrics is None:
        return RiskFactor(
            name="Recent Service Instability",
            score_contribution=w * 0.3,
            weight_max=w,
            evidence="No metrics available. Moderate risk assumed.",
        )

    svc = service_metrics[service_metrics["service"] == service].copy()
    if svc.empty:
        return RiskFactor(
            name="Recent Service Instability",
            score_contribution=w * 0.3,
            weight_max=w,
            evidence=f"No metrics found for service '{service}'.",
        )

    svc = svc.sort_values("date")
    recent = svc.tail(window)

    recent = recent[recent["total_requests"] > 0].copy()
    if recent.empty:
        return RiskFactor(
            name="Recent Service Instability",
            score_contribution=0.0,
            weight_max=w,
            evidence="No traffic in recent window.",
        )

    recent["error_rate"] = recent["failed_requests"] / recent["total_requests"]
    mean_error = recent["error_rate"].mean()
    std_error  = recent["error_rate"].std(ddof=1) if len(recent) > 1 else 0.0

    # Get allowed error rate for context
    allowed_rate: Optional[float] = None
    if slo_definitions is not None:
        slo_row = slo_definitions[slo_definitions["service"] == service]
        if not slo_row.empty:
            target = float(slo_row.iloc[0]["slo_target"])
            allowed_rate = 1.0 - target / 100.0

    days_above_allowed = 0
    if allowed_rate is not None and allowed_rate > 0:
        days_above_allowed = int((recent["error_rate"] > allowed_rate).sum())

    # Score: more days exceeding SLO → higher instability score
    if days_above_allowed >= window // 2:
        fraction = 1.0
        evidence = (
            f"Service exceeded SLO error rate on {days_above_allowed}/{len(recent)} days "
            f"in the last {window} days. Mean error rate: {mean_error:.4%}."
        )
    elif days_above_allowed > 0:
        fraction = 0.5 + 0.1 * days_above_allowed
        fraction = min(fraction, 0.9)
        evidence = (
            f"Service exceeded SLO error rate on {days_above_allowed}/{len(recent)} day(s) "
            f"in the last {window} days. Mean error rate: {mean_error:.4%}."
        )
    elif std_error > mean_error * 0.5 and mean_error > 0:
        # High volatility even if within SLO
        fraction = 0.3
        evidence = (
            f"Error rate shows high variability (std={std_error:.4%}) in last {window} days, "
            f"mean={mean_error:.4%}. Service may be unstable."
        )
    else:
        fraction = max(0.0, mean_error / (allowed_rate or 0.001) * 0.2)
        fraction = min(fraction, 0.2)
        evidence = (
            f"Service appears stable in last {window} days. "
            f"Mean error rate: {mean_error:.4%}."
        )

    return RiskFactor(
        name="Recent Service Instability",
        score_contribution=w * fraction,
        weight_max=w,
        evidence=evidence,
        raw_value={
            "mean_error_rate": mean_error,
            "days_above_allowed": days_above_allowed,
        },
    )


def _build_safeguards(factors: list[RiskFactor], recommendation: str) -> list[str]:
    """Generate safeguard suggestions based on which factors are elevated."""
    safeguards = []
    factor_map = {f.name: f for f in factors}

    def get_contrib(name: str) -> float:
        f = factor_map.get(name)
        if f is None:
            return 0.0
        return f.score_contribution / f.weight_max if f.weight_max > 0 else 0.0

    if get_contrib("Error Budget Consumption") > 0.4:
        safeguards.append("Enable feature flags so this release can be rolled back instantly.")
        safeguards.append(
            "Increase SLO alerting sensitivity during the deployment window (reduce alert threshold)."
        )

    if get_contrib("Error Budget Burn Rate") > 0.4:
        safeguards.append(
            "Perform a canary release (route 1–5% of traffic first) and observe burn rate before promoting."
        )

    if get_contrib("Historical Release Incidents") > 0.4:
        safeguards.append(
            "Run pre-deployment smoke tests and load tests mirroring past incident conditions."
        )

    if get_contrib("Change Size") > 0.5:
        safeguards.append(
            "Consider splitting this release into smaller, independently deployable units."
        )

    if get_contrib("Affected Services") > 0.4:
        safeguards.append(
            "Coordinate with on-call engineers for all affected services during the deployment window."
        )

    if get_contrib("Peak Traffic Window") > 0.3:
        safeguards.append(
            "Consider moving the deployment to an off-peak window (e.g. before 09:00 or after 18:00)."
        )

    if get_contrib("Freeze Period") > 0:
        safeguards.append(
            "⚠ This deployment violates the freeze policy. Obtain explicit approval "
            "from the change review board before proceeding."
        )

    if get_contrib("Recent Service Instability") > 0.4:
        safeguards.append(
            "Stabilise the service (investigate and resolve current elevated error rate) "
            "before adding new risk from a deployment."
        )
        safeguards.append(
            "Set up PagerDuty or equivalent alerts to fire within 5 minutes of error spike."
        )

    if not safeguards and recommendation == "GO WITH SAFEGUARDS":
        safeguards.append("Monitor error rate closely for 30 minutes after deployment.")
        safeguards.append("Have a rollback plan ready before proceeding.")

    return safeguards


def _choose_recommendation(score: float, factors: list[RiskFactor]) -> tuple[str, str]:
    """Select GO / WAIT / GO WITH SAFEGUARDS and write a reason."""
    freeze = next(
        (f for f in factors if f.name == "Freeze Period" and f.score_contribution > 0), None
    )

    if freeze is not None:
        return (
            "WAIT",
            "Deployment is during a freeze period. Override requires explicit change-board approval.",
        )

    if score <= THRESHOLD_GO:
        return (
            "GO",
            f"Risk score {score:.0f} ≤ {THRESHOLD_GO} (GO threshold). "
            f"No critical risk factors detected.",
        )
    elif score <= THRESHOLD_SAFEGUARDS:
        return (
            "GO WITH SAFEGUARDS",
            f"Risk score {score:.0f} is between {THRESHOLD_GO} and {THRESHOLD_SAFEGUARDS}. "
            f"Proceed only with the safeguards listed.",
        )
    else:
        return (
            "WAIT",
            f"Risk score {score:.0f} > {THRESHOLD_SAFEGUARDS} (WAIT threshold). "
            f"Too many risk factors are elevated. Address them before deploying.",
        )


def score_release(
    service: str,
    files_changed: int,
    lines_changed: int,
    affected_services: int,
    deploy_date: date,
    deploy_hour: int,
    service_metrics: Optional[pd.DataFrame] = None,
    slo_definitions: Optional[pd.DataFrame] = None,
    release_history: Optional[pd.DataFrame] = None,
    incidents: Optional[pd.DataFrame] = None,
    traffic_calendar: Optional[pd.DataFrame] = None,
) -> RiskResult:
    """Compute a heuristic risk score for a proposed release.

    All parameters are validated; missing data degrades gracefully with warnings.
    The score is a heuristic on 0–100 and is NOT a calibrated failure probability.

    Parameters
    ----------
    service:              Target service name.
    files_changed:        Number of files in this change.
    lines_changed:        Number of lines changed (additions + deletions).
    affected_services:    Number of services that will be affected.
    deploy_date:          Proposed deployment date.
    deploy_hour:          Proposed deployment hour (0–23, 24-hour clock).
    service_metrics:      Optional metrics DataFrame from data_loader.
    slo_definitions:      Optional SLO DataFrame from data_loader.
    release_history:      Optional release history DataFrame from data_loader.
    incidents:            Optional incidents DataFrame from data_loader.
    traffic_calendar:     Optional calendar DataFrame from data_loader.
    """
    data_warnings: list[str] = []

    # ── Compute error budget ───────────────────────────────────────────────────
    budget: Optional[BudgetResult] = None
    if service_metrics is not None and slo_definitions is not None:
        budget = compute_budget_for_service(service, service_metrics, slo_definitions)
        if budget is None:
            data_warnings.append(
                f"No SLO definition found for '{service}'. "
                "Budget-based risk factors will use a moderate-risk default."
            )
        elif budget.warning:
            data_warnings.append(f"Budget warning for '{service}': {budget.warning}")
    else:
        data_warnings.append(
            "Service metrics or SLO definitions not loaded. "
            "Budget-based risk factors will use moderate-risk defaults."
        )

    # ── Score each factor ─────────────────────────────────────────────────────
    factors = [
        _score_budget_consumed(budget),
        _score_burn_rate(budget),
        _score_historical_incidents(service, release_history, incidents),
        _score_change_size(files_changed, lines_changed),
        _score_affected_services(affected_services),
        _score_peak_traffic(deploy_date, deploy_hour, traffic_calendar),
        _score_freeze_period(deploy_date, traffic_calendar),
        _score_recent_instability(service, service_metrics, slo_definitions),
    ]

    # ── Aggregate score ────────────────────────────────────────────────────────
    total_weight = sum(f.weight_max for f in factors)
    raw_score    = sum(f.score_contribution for f in factors)
    # Normalise to 0-100 relative to total possible weight
    risk_score   = min(100.0, max(0.0, (raw_score / total_weight) * 100.0))

    recommendation, reason = _choose_recommendation(risk_score, factors)
    safeguards = _build_safeguards(factors, recommendation)

    return RiskResult(
        risk_score=risk_score,
        recommendation=recommendation,
        recommendation_reason=reason,
        factors=factors,
        suggested_safeguards=safeguards,
        data_warnings=data_warnings,
        budget=budget,
        inputs={
            "service": service,
            "files_changed": files_changed,
            "lines_changed": lines_changed,
            "affected_services": affected_services,
            "deploy_date": str(deploy_date),
            "deploy_hour": deploy_hour,
        },
    )


# ── What-if analysis ──────────────────────────────────────────────────────────


@dataclass
class WhatIfResult:
    """Comparison between an original and an alternative release scenario."""

    original: RiskResult
    alternative: RiskResult

    score_delta: float          # alternative.risk_score - original.risk_score
    changed_factors: list[str]  # Factor names whose contribution changed

    @property
    def improved(self) -> bool:
        return self.score_delta < 0

    def summary(self) -> str:
        direction = "↓ improved" if self.improved else "↑ worsened"
        return (
            f"Risk score {direction} by {abs(self.score_delta):.1f} points "
            f"({self.original.risk_score:.1f} → {self.alternative.risk_score:.1f}). "
            f"Recommendation: {self.original.recommendation} → {self.alternative.recommendation}."
        )


def compare_scenarios(original: RiskResult, alternative: RiskResult) -> WhatIfResult:
    """Compare two RiskResult objects and identify changed factors."""
    orig_map = {f.name: f.score_contribution for f in original.factors}
    alt_map  = {f.name: f.score_contribution for f in alternative.factors}

    changed = [
        name
        for name in set(orig_map) | set(alt_map)
        if abs(orig_map.get(name, 0) - alt_map.get(name, 0)) > 0.01
    ]

    return WhatIfResult(
        original=original,
        alternative=alternative,
        score_delta=alternative.risk_score - original.risk_score,
        changed_factors=changed,
    )
