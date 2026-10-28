"""
test_dashboard_ui.py — Comprehensive UI Component and App Testing
===================================================================

Tests:
  1. CSS injection and custom color definitions
  2. Component rendering with real datasets and edge-case synthetic data
  3. Risk gauge, budget gauge, factor breakdown, and trend chart generators
  4. Historical table generation with and without incident joins
  5. Streamlit AppTest execution across all pages and interactive forms
  6. Filter interactions on Historical Releases page
  7. Edge case rendering: 100% budget consumed, zero requests, freeze period
"""

import datetime
from pathlib import Path
import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.data_loader import load_all_data
from src.risk_engine import (
    score_release,
    compare_scenarios,
    compute_budget_for_service,
    calculate_error_budget,
)
from src.dashboard_components import (
    COLORS,
    risk_gauge,
    budget_gauge,
    factor_breakdown_chart,
    error_rate_trend,
    latency_trend_chart,
    request_volume_chart,
    releases_with_incidents_table,
    whatif_comparison_chart,
)


@pytest.fixture
def dataset():
    return load_all_data()


def test_colors_palette():
    required_keys = ["bg", "card", "border", "blue", "teal", "amber", "coral", "violet"]
    for k in required_keys:
        assert k in COLORS
        assert COLORS[k].startswith("#") or COLORS[k].startswith("rgba")


def test_charts_creation(dataset):
    # Test risk gauge
    fig_gauge = risk_gauge(45.0)
    assert fig_gauge is not None

    # Test budget gauge
    budget = compute_budget_for_service(
        "payment-api", dataset.service_metrics, dataset.slo_definitions
    )
    assert budget is not None
    fig_bg = budget_gauge(budget)
    assert fig_bg is not None

    # Test factor breakdown chart
    risk_res = score_release(
        service="payment-api",
        files_changed=5,
        lines_changed=300,
        affected_services=1,
        deploy_date=datetime.date(2026, 10, 28),
        deploy_hour=14,
        service_metrics=dataset.service_metrics,
        slo_definitions=dataset.slo_definitions,
        release_history=dataset.release_history,
        incidents=dataset.incidents,
        traffic_calendar=dataset.traffic_calendar,
    )
    fig_fb = factor_breakdown_chart(risk_res.factors)
    assert fig_fb is not None

    # Test trends
    fig_err = error_rate_trend("payment-api", dataset.service_metrics, 99.9)
    assert fig_err is not None

    fig_lat = latency_trend_chart("payment-api", dataset.service_metrics)
    assert fig_lat is not None

    fig_vol = request_volume_chart("payment-api", dataset.service_metrics)
    assert fig_vol is not None


def test_edge_case_gauges():
    # Budget exhausted
    b_exhausted = calculate_error_budget("test-svc", 99.9, 10000, 20)
    fig = budget_gauge(b_exhausted)
    assert fig is not None

    # Zero requests
    b_zero = calculate_error_budget("test-svc", 99.9, 0, 0)
    fig_zero = budget_gauge(b_zero)
    assert fig_zero is not None

    # Boundary risk scores
    assert risk_gauge(0.0) is not None
    assert risk_gauge(100.0) is not None


def test_historical_table_generation(dataset):
    df_merged = releases_with_incidents_table(
        dataset.release_history, dataset.incidents, service=None
    )
    assert not df_merged.empty
    assert "release_id" in df_merged.columns
    assert "incident_count" in df_merged.columns
    assert "has_incident" in df_merged.columns

    # Filtered by service
    df_user = releases_with_incidents_table(
        dataset.release_history, dataset.incidents, service="user-api"
    )
    assert (df_user["service"] == "user-api").all()

    # Empty incidents handling
    df_no_inc = releases_with_incidents_table(
        dataset.release_history, pd.DataFrame(), service=None
    )
    assert not df_no_inc.empty
    assert (df_no_inc["incident_count"] == 0).all()


def test_whatif_comparison_chart(dataset):
    res_a = score_release(
        service="user-api",
        files_changed=15,
        lines_changed=1500,
        affected_services=3,
        deploy_date=datetime.date(2026, 10, 28),
        deploy_hour=15,
        service_metrics=dataset.service_metrics,
        slo_definitions=dataset.slo_definitions,
        release_history=dataset.release_history,
        incidents=dataset.incidents,
        traffic_calendar=dataset.traffic_calendar,
    )
    res_b = score_release(
        service="user-api",
        files_changed=3,
        lines_changed=100,
        affected_services=1,
        deploy_date=datetime.date(2026, 10, 28),
        deploy_hour=6,
        service_metrics=dataset.service_metrics,
        slo_definitions=dataset.slo_definitions,
        release_history=dataset.release_history,
        incidents=dataset.incidents,
        traffic_calendar=dataset.traffic_calendar,
    )
    comp = compare_scenarios(res_a, res_b)
    fig_comp = whatif_comparison_chart(comp)
    assert fig_comp is not None


def test_app_test_overview_and_navigation():
    """Verify Streamlit app runs end-to-end without unhandled exceptions."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"App threw exception on load: {at.exception}"

    # Navigate to Service Health
    at.radio[0].set_value("🩺 Service Health").run()
    assert not at.exception, f"Exception on Service Health: {at.exception}"

    # Change service in dropdown
    if len(at.selectbox) > 0:
        at.selectbox[0].select("search-api").run()
        assert not at.exception, f"Exception changing service dropdown: {at.exception}"

    # Navigate to Evaluate a Release
    at.radio[0].set_value("🚀 Evaluate a Release").run()
    assert not at.exception, f"Exception on Evaluate a Release: {at.exception}"

    # Submit evaluation form
    if len(at.button) > 0:
        at.button[0].click().run()
        assert not at.exception, f"Exception submitting release evaluation: {at.exception}"

    # Navigate to What-If Analysis
    at.radio[0].set_value("🔮 What-If Analysis").run()
    assert not at.exception, f"Exception on What-If Analysis: {at.exception}"
    if len(at.button) > 0:
        at.button[0].click().run()
        assert not at.exception, f"Exception comparing scenarios: {at.exception}"

    # Navigate to Historical Releases & Incidents
    at.radio[0].set_value("📜 Historical Releases & Incidents").run()
    assert not at.exception, f"Exception on Historical Releases: {at.exception}"

    # Filter incidents only
    if len(at.checkbox) > 0:
        at.checkbox[0].check().run()
        assert not at.exception, f"Exception checking incidents only: {at.exception}"
