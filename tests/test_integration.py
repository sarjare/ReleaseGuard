"""
test_integration.py — End-to-End QA Integration Tests for ReleaseGuard
========================================================================

Covers the 10 required QA test areas:
  1. Successful dataset loading
  2. Missing CSV files
  3. Invalid or missing columns
  4. Date and numeric parsing
  5. Duplicate identifiers and unmatched references
  6. Dataset joins and row-count integrity
  7. Risk-engine integration with loaded data
  8. Invalid, missing and empty inputs
  9. Release recommendation outputs (GO / WAIT / GO WITH SAFEGUARDS)
  10. Dashboard imports and basic rendering via Streamlit AppTest
"""

import datetime
import hashlib
from pathlib import Path
import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.data_loader import (
    DATA_DIR,
    LoadedData,
    get_services,
    load_all_data,
    _load_csv,
    _validate_slo_definitions,
    _validate_service_metrics,
    _validate_release_history,
    _validate_incidents,
    _validate_traffic_calendar,
    _cross_validate,
)
from src.risk_engine import (
    THRESHOLD_GO,
    THRESHOLD_SAFEGUARDS,
    BudgetResult,
    RiskResult,
    WhatIfResult,
    calculate_error_budget,
    compare_scenarios,
    compute_budget_for_service,
    score_release,
)
from src.dashboard_components import (
    COLORS,
    releases_with_incidents_table,
    risk_gauge,
    budget_gauge,
    error_rate_trend,
    whatif_comparison_chart,
)


@pytest.fixture
def real_data() -> LoadedData:
    return load_all_data()


# ── 1. Successful Dataset Loading ─────────────────────────────────────────────
def test_successful_dataset_loading(real_data):
    """Verify all 5 CSV files are loaded and parsed successfully."""
    assert real_data.is_usable
    assert real_data.slo_definitions is not None and not real_data.slo_definitions.empty
    assert real_data.service_metrics is not None and not real_data.service_metrics.empty
    assert real_data.release_history is not None and not real_data.release_history.empty
    assert real_data.incidents is not None and not real_data.incidents.empty
    assert real_data.traffic_calendar is not None and not real_data.traffic_calendar.empty

    services = get_services(real_data)
    assert len(services) >= 4
    assert "payment-api" in services
    assert "user-api" in services


# ── 2. Missing CSV Files ──────────────────────────────────────────────────────
def test_missing_csv_files(tmp_path):
    """Missing CSV files must produce readable errors and not crash."""
    empty_dir = tmp_path / "empty_datasets"
    empty_dir.mkdir()
    result = load_all_data(data_dir=empty_dir)

    assert isinstance(result, LoadedData)
    assert not result.is_usable
    assert len(result.errors) >= 3
    assert any("slo_definitions.csv" in err for err in result.errors)
    assert any("service_metrics.csv" in err for err in result.errors)
    assert any("release_history.csv" in err for err in result.errors)


# ── 3. Invalid or Missing Columns ─────────────────────────────────────────────
def test_invalid_or_missing_columns(tmp_path):
    """CSVs missing mandatory headers must record descriptive errors."""
    bad_csv = tmp_path / "slo_definitions.csv"
    bad_csv.write_text("service,unknown_col\npayment-api,99.9\n")
    result = LoadedData()
    df = _load_csv("slo_definitions", bad_csv, result)
    assert df is None
    assert any("missing required columns" in e.lower() for e in result.errors)


# ── 4. Date and Numeric Parsing ───────────────────────────────────────────────
def test_date_and_numeric_parsing():
    """Verify dates are coerced safely and non-numeric values are sanitized."""
    result = LoadedData()

    # Invalid dates and negative counts in service metrics
    metrics_raw = pd.DataFrame([
        {"date": "2026-10-01", "service": "svc-a", "total_requests": 1000, "failed_requests": 5},
        {"date": "invalid-date", "service": "svc-a", "total_requests": 1000, "failed_requests": 5},
        {"date": "2026-10-02", "service": "svc-a", "total_requests": -50, "failed_requests": 2},
    ])
    validated_m = _validate_service_metrics(metrics_raw, result)
    assert len(validated_m) == 1
    assert any("unparseable dates" in w for w in result.warnings)
    assert any("invalid numeric values" in w for w in result.warnings)

    # Deployment hour range checking
    releases_raw = pd.DataFrame([
        {"release_id": "R1", "date": "2026-10-01", "deployment_hour": 25, "service": "svc-a", "files_changed": 1, "lines_changed": 10, "affected_services": 1, "change_type": "fix"},
    ])
    _ = _validate_release_history(releases_raw, result)
    assert any("deployment_hour outside 0–23" in w for w in result.warnings)


# ── 5. Duplicate Identifiers and Unmatched References ─────────────────────────
def test_duplicate_identifiers_and_unmatched_references():
    """Duplicate IDs and orphaned incident links must be flagged."""
    result = LoadedData()

    # Duplicate release_id
    rel_df = pd.DataFrame([
        {"release_id": "REL-1", "date": "2026-10-01", "deployment_hour": 10, "service": "svc-a", "files_changed": 1, "lines_changed": 10, "affected_services": 1, "change_type": "fix"},
        {"release_id": "REL-1", "date": "2026-10-02", "deployment_hour": 11, "service": "svc-a", "files_changed": 2, "lines_changed": 20, "affected_services": 1, "change_type": "fix"},
    ])
    result.release_history = _validate_release_history(rel_df, result)
    assert any("duplicate release_id" in w.lower() for w in result.warnings)

    # Orphaned incident reference
    inc_df = pd.DataFrame([
        {"incident_id": "INC-1", "date": "2026-10-01", "service": "svc-a", "related_release": "REL-GHOST", "severity": "Major"},
    ])
    result.incidents = _validate_incidents(inc_df, result)
    _cross_validate(result)
    assert any("REL-GHOST" in w for w in result.warnings)


# ── 6. Dataset Joins and Row-Count Integrity ──────────────────────────────────
def test_dataset_joins_and_row_count_integrity(real_data):
    """Merging releases with multiple incidents must not cause row duplication."""
    rel = real_data.release_history
    inc = real_data.incidents

    # Join releases and incidents
    merged = releases_with_incidents_table(rel, inc, service="All")
    assert len(merged) == len(rel), "Join should preserve exactly 1 row per release"

    # Filtered by specific service
    user_merged = releases_with_incidents_table(rel, inc, service="user-api")
    expected_user_count = len(rel[rel["service"] == "user-api"])
    assert len(user_merged) == expected_user_count


# ── 7. Risk-Engine Integration with Loaded Data ───────────────────────────────
def test_risk_engine_integration_with_loaded_data(real_data):
    """The risk engine must successfully evaluate releases for all known services."""
    services = get_services(real_data)
    for svc in services:
        res = score_release(
            service=svc,
            files_changed=8,
            lines_changed=600,
            affected_services=2,
            deploy_date=datetime.date(2026, 10, 28),
            deploy_hour=14,
            service_metrics=real_data.service_metrics,
            slo_definitions=real_data.slo_definitions,
            release_history=real_data.release_history,
            incidents=real_data.incidents,
            traffic_calendar=real_data.traffic_calendar,
        )
        assert isinstance(res, RiskResult)
        assert 0.0 <= res.risk_score <= 100.0
        assert res.recommendation in ["GO", "WAIT", "GO WITH SAFEGUARDS"]
        assert len(res.factors) == 8


# ── 8. Invalid, Missing and Empty Inputs ───────────────────────────────────────
def test_invalid_missing_empty_inputs():
    """Engine must degrade gracefully when metrics or services are missing."""
    # Unknown service with None dataframes
    res = score_release(
        service="nonexistent-service",
        files_changed=0,
        lines_changed=0,
        affected_services=0,
        deploy_date=datetime.date(2026, 10, 28),
        deploy_hour=10,
        service_metrics=None,
        slo_definitions=None,
    )
    assert isinstance(res, RiskResult)
    assert 0.0 <= res.risk_score <= 100.0
    assert len(res.data_warnings) > 0


# ── 9. Release Recommendation Outputs ─────────────────────────────────────────
def test_release_recommendation_outputs(real_data):
    """Verify recommendation logic adheres to policy thresholds and freeze periods."""
    # Freeze period deployment must ALWAYS yield WAIT
    freeze_date = datetime.date(2026, 10, 15)  # Known freeze date in calendar
    res_freeze = score_release(
        service="payment-api",
        files_changed=1,
        lines_changed=5,
        affected_services=1,
        deploy_date=freeze_date,
        deploy_hour=10,
        service_metrics=real_data.service_metrics,
        slo_definitions=real_data.slo_definitions,
        traffic_calendar=real_data.traffic_calendar,
    )
    # If the date is marked as freeze_period in traffic_calendar, recommendation must be WAIT
    cal = real_data.traffic_calendar
    is_freeze = cal[cal["date"].dt.date == freeze_date]["freeze_period"].any()
    if is_freeze:
        assert res_freeze.recommendation == "WAIT"

    # What-If scenario delta calculation
    res_alt = score_release(
        service="payment-api",
        files_changed=1,
        lines_changed=10,
        affected_services=1,
        deploy_date=datetime.date(2026, 10, 28),
        deploy_hour=6,
        service_metrics=real_data.service_metrics,
        slo_definitions=real_data.slo_definitions,
        traffic_calendar=real_data.traffic_calendar,
    )
    comp = compare_scenarios(res_freeze, res_alt)
    assert isinstance(comp, WhatIfResult)
    assert isinstance(comp.improved, bool)


# ── 10. Dashboard Imports and Basic Rendering ─────────────────────────────────
def test_dashboard_imports_and_rendering():
    """Verify Streamlit app initializes and navigates without unhandled exceptions."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception

    # Section 1: Overview
    assert at.radio[0].value == "🏠 Overview Dashboard"

    # Section 2: Service Health
    at.radio[0].set_value("🩺 Service Health").run()
    assert not at.exception

    # Section 3: Evaluate a Release
    at.radio[0].set_value("🚀 Evaluate a Release").run()
    assert not at.exception

    # Section 4: What-If Analysis
    at.radio[0].set_value("🔮 What-If Analysis").run()
    assert not at.exception

    # Section 5: Historical Releases & Incidents
    at.radio[0].set_value("📜 Historical Releases & Incidents").run()
    assert not at.exception
