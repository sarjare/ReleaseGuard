"""
app.py — ReleaseGuard: AI Ops Release Safety Advisor
=====================================================

Professional SRE & DevOps Control Room Dashboard for AI Ops Release Governance.
Evaluates deployment risk using SLO error budgets, historical incidents,
change size blast radius, and traffic calendar windows.

Sections:
  1. 🏠 Overview Dashboard
  2. 🩺 Service Health
  3. 🚀 Evaluate a Release & Recommendation
  4. 🔮 What-If Analysis
  5. 📜 Historical Releases & Incidents

Design System:
  - Dark navy operations center theme with blue, violet, and teal accents.
  - Reusable components from src.dashboard_components.
  - Transparent integration with src.risk_engine and src.data_loader.
  - Explicit synthetic data labelling throughout the UI.

Author: ReleaseGuard Team
"""

from __future__ import annotations

import datetime
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="ReleaseGuard — AI Ops Release Safety Advisor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports from Project Modules ──────────────────────────────────────────────
from src.dashboard_components import (
    COLORS,
    budget_gauge,
    control_room_header,
    data_warnings_box,
    error_rate_trend,
    explanation_panel,
    factor_breakdown_chart,
    factor_evidence_table,
    inject_css,
    latency_trend_chart,
    metric_card,
    ml_prediction_card,
    recommendation_banner,
    releases_with_incidents_table,
    request_volume_chart,
    risk_gauge,
    safeguards_list,
    service_topology_chart,
    synthetic_notice,
    traffic_window_badge,
    whatif_comparison_chart,
    whatif_delta_card,
)
from src.data_loader import LoadedData, get_services, load_all_data
from src.explainability import build_deterministic_explanation, generate_ai_summary
from src.ml_predictor import estimate_release_probability
from src.risk_engine import (
    THRESHOLD_GO,
    THRESHOLD_SAFEGUARDS,
    WEIGHTS,
    BudgetResult,
    RiskResult,
    compare_scenarios,
    compute_budget_for_service,
    score_release,
)

# ── CSS Injection ─────────────────────────────────────────────────────────────
inject_css()

# ── Data Loading & Session Initialization ─────────────────────────────────────
@st.cache_data(ttl=300)
def get_data() -> LoadedData:
    return load_all_data()

data = get_data()

if "evaluations_run" not in st.session_state:
    st.session_state["evaluations_run"] = 0

# ── Sidebar Navigation & Control Panel ────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 16px 12px 20px; background: linear-gradient(180deg, rgba(168,85,247,0.12) 0%, rgba(56,189,248,0.06) 50%, transparent 100%); border-radius: 14px; margin-bottom: 14px; border: 1px solid rgba(168,85,247,0.24); box-shadow: 0 0 24px rgba(168,85,247,0.14);">
          <div style="font-size:2.6rem; line-height:1; filter: drop-shadow(0 0 16px rgba(56,189,248,0.5));">🛡️</div>
          <div style="font-size:1.35rem; font-weight:800; color:{COLORS['text']}; letter-spacing:0.03em; margin-top:6px;">
            ReleaseGuard
          </div>
          <div style="display:inline-block; font-size:0.68rem; font-family:'JetBrains Mono',monospace; color:{COLORS['cyan']}; background:rgba(34,211,238,0.12); padding:3px 12px; border-radius:14px; margin-top:6px; border:1px solid rgba(34,211,238,0.35); letter-spacing:0.08em;">
            COSMIC SRE CONTROL ROOM
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_selection = st.radio(
        "Navigation",
        [
            "🏠 Overview Dashboard",
            "🩺 Service Health",
            "🚀 Evaluate a Release",
            "🔮 What-If Analysis",
            "📜 Historical Releases & Incidents",
        ],
        label_visibility="collapsed",
    )

    st.markdown(f"<hr style='border-color:rgba(255,255,255,0.08); margin:16px 0;'/>", unsafe_allow_html=True)

    # Dataset Status
    if data.is_usable:
        st.markdown(
            f"""
            <div style="background:rgba(45,212,191,0.08); backdrop-filter:blur(10px); border:1px solid rgba(45,212,191,0.35); border-radius:10px; padding:10px 14px; font-size:0.82rem; color:{COLORS['teal']}; display:flex; align-items:center; gap:10px;">
              <span style="font-size:1.1rem;">✅</span>
              <div>
                <strong>Datasets Verified</strong><br/>
                <span style="font-size:0.75rem; color:{COLORS['text_muted']};">5/5 CSV sources validated</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background:rgba(239,68,68,0.08); backdrop-filter:blur(10px); border:1px solid rgba(239,68,68,0.35); border-radius:10px; padding:10px 14px; font-size:0.82rem; color:{COLORS['coral']}; display:flex; align-items:center; gap:10px;">
              <span style="font-size:1.1rem;">❌</span>
              <div>
                <strong>Dataset Error</strong><br/>
                <span style="font-size:0.75rem; color:{COLORS['text_muted']};">Critical CSV missing</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if data.errors:
        with st.expander("❌ Fatal Data Errors", expanded=True):
            for err in data.errors:
                st.error(err)

    if data.warnings:
        with st.expander(f"⚠️ Data Notices ({len(data.warnings)})"):
            for warn in data.warnings:
                st.caption(f"• {warn}")

    st.markdown(f"<hr style='border-color:{COLORS['border']}; margin:16px 0;'/>", unsafe_allow_html=True)

    # Risk Engine Configuration (read-only reference)
    with st.expander("⚙️ Risk Engine Policy", expanded=False):
        st.markdown(
            f"""
            <div style="font-size:0.75rem; color:{COLORS['text_muted']};">
              <strong>Scoring Thresholds:</strong><br/>
              • 0 – {THRESHOLD_GO}: <span style="color:{COLORS['teal']}; font-weight:600;">GO</span><br/>
              • {THRESHOLD_GO+1} – {THRESHOLD_SAFEGUARDS}: <span style="color:{COLORS['amber']}; font-weight:600;">GO WITH SAFEGUARDS</span><br/>
              • > {THRESHOLD_SAFEGUARDS}: <span style="color:{COLORS['coral']}; font-weight:600;">WAIT</span>
            </div>
            <div style="margin-top:10px; font-size:0.75rem; color:{COLORS['text_muted']};">
              <strong>Factor Weights (Max Pts):</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for factor_key, weight_val in WEIGHTS.items():
            st.caption(f"• `{factor_key}`: **{weight_val:.0f} pts**")

    # Compliance notice in sidebar
    synthetic_notice()


# ── Guard: Ensure Minimum Datasets Loaded ──────────────────────────────────────
if not data.is_usable:
    st.error(
        "### ⚠️ Data Loading Halted\n"
        "ReleaseGuard cannot initialize because required CSV files in `data/` could not be loaded.\n"
        "Please check the sidebar errors."
    )
    st.stop()

services_list = get_services(data)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if nav_selection == "🏠 Overview Dashboard":
    control_room_header("Overview Dashboard")

    # Calculate system-wide KPIs
    total_services = len(services_list)
    budgets: dict[str, BudgetResult] = {}
    healthy_services_count = 0
    warning_services_count = 0
    critical_services_count = 0

    for svc in services_list:
        if data.service_metrics is not None and data.slo_definitions is not None:
            b = compute_budget_for_service(svc, data.service_metrics, data.slo_definitions)
            if b is not None:
                budgets[svc] = b
                if b.budget_consumed_pct >= 80 or b.burn_rate >= 2.0:
                    critical_services_count += 1
                elif b.budget_consumed_pct >= 50 or b.burn_rate >= 1.0:
                    warning_services_count += 1
                else:
                    healthy_services_count += 1

    total_incidents = len(data.incidents) if data.incidents is not None else 0
    total_releases_recorded = len(data.release_history) if data.release_history is not None else 0
    total_evaluated = total_releases_recorded + st.session_state["evaluations_run"]

    # 4 Primary KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    metric_card(
        label="Services Monitored",
        value=total_services,
        sub="Availability SLOs tracked",
        accent="blue",
        col=col1,
    )
    metric_card(
        label="Healthy Services",
        value=f"{healthy_services_count} / {total_services}",
        sub=f"{critical_services_count} critical, {warning_services_count} warning",
        accent="teal" if critical_services_count == 0 else "amber",
        col=col2,
    )
    metric_card(
        label="Recent Incidents",
        value=total_incidents,
        sub="Linked to historical releases",
        accent="coral" if total_incidents > 0 else "teal",
        col=col3,
    )
    metric_card(
        label="Releases Evaluated",
        value=total_evaluated,
        sub=f"{total_releases_recorded} in dataset + {st.session_state['evaluations_run']} session",
        accent="violet",
        col=col4,
    )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ── Signature Cosmic Service Topology Hero Section ────────────────────────
    st.markdown(
        f"""
        <div class="topology-container">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:10px;">
            <div>
              <div style="font-size:1.24rem; font-weight:700; color:{COLORS['text']}; display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.5rem;">🌌</span>
                <span>Autonomous Service Mesh & Release Topology</span>
              </div>
              <div style="font-size:0.84rem; color:{COLORS['text_muted']}; margin-top:3px;">
                Luminous service dependency filaments, real-time SLO error budget consumption, and blast radius mapping
              </div>
            </div>
            <div style="display:flex; align-items:center; gap:14px; font-size:0.75rem; font-family:'JetBrains Mono',monospace;">
              <span style="display:inline-flex; align-items:center; gap:5px;"><span style="color:{COLORS['teal']};">●</span> Healthy (&lt;50% Budget)</span>
              <span style="display:inline-flex; align-items:center; gap:5px;"><span style="color:{COLORS['amber']};">●</span> Warning (50-80%)</span>
              <span style="display:inline-flex; align-items:center; gap:5px;"><span style="color:{COLORS['coral']};">●</span> Critical (&gt;80% / Incidents)</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(service_topology_chart(data, height=420), use_container_width=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # ── Service Health Status Grid ────────────────────────────────────────────
    st.markdown('<div class="section-header">🩺 Service SLO & Error Budget Fleet Status</div>', unsafe_allow_html=True)

    if budgets:
        rows = []
        for svc, b in budgets.items():
            slo_row = data.slo_definitions[data.slo_definitions["service"] == svc]
            target_val = float(slo_row.iloc[0]["slo_target"]) if not slo_row.empty else 99.9

            if b.budget_consumed_pct >= 100:
                health_badge = "🔴 Exhausted"
            elif b.budget_consumed_pct >= 80 or b.burn_rate >= 2.0:
                health_badge = "🟠 Critical"
            elif b.budget_consumed_pct >= 50 or b.burn_rate >= 1.0:
                health_badge = "🟡 Warning"
            else:
                health_badge = "🟢 Healthy"

            rows.append({
                "Service": svc,
                "Health Status": health_badge,
                "SLO Target (%)": f"{target_val:.2f}%",
                "Actual Error Rate": f"{b.actual_error_rate*100:.4f}%",
                "Budget Consumed (%)": f"{b.budget_consumed_pct:.1f}%",
                "Remaining Budget (%)": f"{b.remaining_budget_pct:.1f}%",
                "Burn Rate": f"{b.burn_rate:.2f}×",
                "Requests (Window)": f"{b.total_requests:,.0f}",
                "Failed (Window)": f"{b.failed_requests:,.0f}",
            })

        summary_df = pd.DataFrame(rows)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Error budget metrics unavailable.")

    # ── Multi-Service Reliability Sparklines ───────────────────────────────────
    st.markdown('<div class="section-header">📈 Multi-Service Error Rate Timelines (Last 30 Days)</div>', unsafe_allow_html=True)

    if data.service_metrics is not None:
        spark_cols = st.columns(min(3, len(services_list)))
        for idx, svc in enumerate(services_list):
            slo_row = (
                data.slo_definitions[data.slo_definitions["service"] == svc]
                if data.slo_definitions is not None
                else pd.DataFrame()
            )
            slo_t = float(slo_row.iloc[0]["slo_target"]) if not slo_row.empty else None
            with spark_cols[idx % 3]:
                fig_trend = error_rate_trend(svc, data.service_metrics, slo_t)
                fig_trend.update_layout(
                    height=220,
                    title=dict(text=f"<b>{svc}</b>", font=dict(size=12, color=COLORS["text"])),
                    margin=dict(l=25, r=15, t=35, b=25),
                )
                st.plotly_chart(fig_trend, use_container_width=True)

    synthetic_notice()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SERVICE HEALTH
# ═══════════════════════════════════════════════════════════════════════════════
elif nav_selection == "🩺 Service Health":
    control_room_header("Service Health & Error Budget Telemetry")

    col_selector, col_context = st.columns([1, 2])
    with col_selector:
        selected_svc = st.selectbox("Select Monitored Service", services_list, key="sh_svc")

    slo_row = (
        data.slo_definitions[data.slo_definitions["service"] == selected_svc]
        if data.slo_definitions is not None
        else pd.DataFrame()
    )

    slo_target = 99.9
    sli_type = "availability"
    description = ""
    if not slo_row.empty:
        slo_target = float(slo_row.iloc[0]["slo_target"])
        sli_type = str(slo_row.iloc[0].get("sli_type", "availability"))
        description = str(slo_row.iloc[0].get("description", ""))

    with col_context:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg, rgba(19,26,39,0.85) 0%, rgba(23,33,54,0.7) 100%); backdrop-filter:blur(14px); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:14px 20px; margin-top:24px; box-shadow:0 4px 18px rgba(0,0,0,0.25);">
              <span style="color:{COLORS['blue']}; font-size:1.05rem; font-weight:700;">{selected_svc}</span> &nbsp;|&nbsp;
              <strong>SLO Target:</strong> <code style="color:{COLORS['teal']}; font-weight:700;">{slo_target}%</code> &nbsp;|&nbsp;
              <strong>SLI:</strong> <code>{sli_type}</code> &nbsp;|&nbsp;
              <span style="color:{COLORS['text_muted']}; font-size:0.85rem;">{description}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Compute Budget
    budget = None
    if data.service_metrics is not None and data.slo_definitions is not None:
        budget = compute_budget_for_service(
            selected_svc, data.service_metrics, data.slo_definitions
        )

    if budget is None:
        st.error(f"Cannot compute error budget for '{selected_svc}'. Please check dataset warnings.")
    else:
        if budget.warning:
            data_warnings_box([budget.warning])

        # KPI Metrics Row
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        metric_card("SLO Target", f"{budget.slo_target:.2f}%", sub="Availability standard", accent="blue", col=col_m1)
        metric_card(
            "Budget Consumed",
            f"{budget.budget_consumed_pct:.1f}%",
            sub=f"Remaining: {budget.remaining_budget_pct:.1f}%",
            accent="coral" if budget.budget_consumed_pct >= 80 else ("amber" if budget.budget_consumed_pct >= 50 else "teal"),
            col=col_m2,
        )
        metric_card(
            "Burn Rate",
            f"{budget.burn_rate:.2f}×",
            sub="Allowed vs observed error rate",
            accent="coral" if budget.burn_rate >= 2.0 else ("amber" if budget.burn_rate >= 1.0 else "teal"),
            col=col_m3,
        )
        metric_card(
            "Observed Error Rate",
            f"{budget.actual_error_rate*100:.4f}%",
            sub=f"{budget.failed_requests:,.0f} / {budget.total_requests:,.0f} requests",
            accent="violet",
            col=col_m4,
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Visual Reliability Panels: Donut Gauge + Error Rate Trend
        col_gauge, col_trend = st.columns([1, 2])
        with col_gauge:
            st.plotly_chart(budget_gauge(budget), use_container_width=True)
        with col_trend:
            if data.service_metrics is not None:
                st.plotly_chart(
                    error_rate_trend(selected_svc, data.service_metrics, budget.slo_target),
                    use_container_width=True,
                )

        # Traffic & Latency Visualizations
        col_vol, col_lat = st.columns(2)
        with col_vol:
            if data.service_metrics is not None:
                st.plotly_chart(
                    request_volume_chart(selected_svc, data.service_metrics),
                    use_container_width=True,
                )
        with col_lat:
            if data.service_metrics is not None:
                lat_fig = latency_trend_chart(selected_svc, data.service_metrics)
                if lat_fig:
                    st.plotly_chart(lat_fig, use_container_width=True)
                else:
                    st.info("Latency metrics not available for this service.")

        # Service Mesh Topology View
        with st.expander(f"🌌 View {selected_svc} in Cosmic Service Mesh Topology"):
            st.plotly_chart(service_topology_chart(data, selected_service=selected_svc, height=340), use_container_width=True)

        # Detailed Budget Breakdown Table
        with st.expander("📊 SRE Error Budget Ledger Breakdown"):
            st.markdown(
                f"""
                | Metric Dimension | Value | SRE Interpretation |
                | :--- | :--- | :--- |
                | **SLO Target** | `{budget.slo_target}%` | Agreed availability baseline |
                | **Allowed Error Rate** | `{budget.allowed_error_rate:.6f}` ({budget.allowed_error_rate*100:.4f}%) | 1 - (SLO Target ÷ 100) |
                | **Actual Error Rate** | `{budget.actual_error_rate:.6f}` ({budget.actual_error_rate*100:.4f}%) | Failed Requests ÷ Total Requests |
                | **Total Requests** | `{budget.total_requests:,.0f}` | Measured volume in observation window |
                | **Failed Requests** | `{budget.failed_requests:,.0f}` | Recorded unhandled errors / 5xx |
                | **Allowed Error Budget** | `{budget.allowed_errors:,.1f} errors` | Total volume × Allowed error rate |
                | **Budget Consumed** | `{budget.budget_consumed_pct:.2f}%` | (Failed errors ÷ Allowed errors) × 100 |
                | **Remaining Budget** | `{budget.remaining_budget_pct:.2f}%` | Available headroom before SLO breach |
                | **Burn Rate Factor** | `{budget.burn_rate:.3f}×` | <1× stable, 1–2× elevated, >2× high risk |
                """
            )

    # Historical Incidents for Selected Service
    st.markdown(f'<div class="section-header">📜 Recorded Incidents for {selected_svc}</div>', unsafe_allow_html=True)
    if data.incidents is not None:
        svc_incidents = data.incidents[data.incidents["service"] == selected_svc].copy()
        if svc_incidents.empty:
            st.success(f"✅ No incidents associated with '{selected_svc}' in the dataset.")
        else:
            svc_incidents = svc_incidents.sort_values("date", ascending=False)
            st.dataframe(svc_incidents, use_container_width=True, hide_index=True)
    else:
        st.info("Incidents dataset unavailable.")

    synthetic_notice()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EVALUATE A RELEASE & RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════
elif nav_selection == "🚀 Evaluate a Release":
    control_room_header("Evaluate a Release & SRE Recommendation")

    st.markdown(
        "Enter proposed release parameters to calculate an explainable risk score and "
        "receive an operational deployment recommendation (GO / WAIT / GO WITH SAFEGUARDS)."
    )

    with st.form("release_eval_form"):
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            eval_service = st.selectbox("Target Service", services_list, key="form_eval_svc")
            files_changed = st.number_input(
                "Files Changed",
                min_value=1,
                max_value=1000,
                value=6,
                step=1,
                help="Total number of files modified in this pull request / artifact.",
            )
            lines_changed = st.number_input(
                "Lines Changed (Insertions + Deletions)",
                min_value=1,
                max_value=100_000,
                value=450,
                step=50,
                help="Total lines of code touched across all modified files.",
            )
            change_type = st.selectbox(
                "Change Classification",
                ["bugfix", "feature", "refactor", "dependency_update", "hotfix", "config", "other"],
                help="Operational category of the change.",
            )

        with col_f2:
            affected_services = st.number_input(
                "Affected Services (Blast Radius)",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                help="Number of interconnected services directly or transitively affected.",
            )
            deploy_date = st.date_input(
                "Proposed Deployment Date",
                value=datetime.date.today(),
                help="Date on which this deployment is scheduled to run.",
            )
            deploy_hour = st.slider(
                "Proposed Deployment Hour (24-Hour UTC)",
                min_value=0,
                max_value=23,
                value=14,
                format="%02d:00",
                help="Target hour for execution. Business hours (09:00 - 18:00) increase traffic risk.",
            )

        # Contextual traffic window preview
        st.markdown("<div style='margin-top: 8px;'><strong>Traffic Window Status:</strong></div>", unsafe_allow_html=True)
        traffic_window_badge(deploy_date, int(deploy_hour), data.traffic_calendar)

        st.markdown("<hr style='margin: 16px 0; border-color: rgba(255,255,255,0.08);'/>", unsafe_allow_html=True)
        eval_submitted = st.form_submit_button("🛡️ Run SRE Risk Evaluation", use_container_width=True, type="primary")

    if eval_submitted:
        with st.spinner("Executing risk engine calculations..."):
            risk_result = score_release(
                service=eval_service,
                files_changed=int(files_changed),
                lines_changed=int(lines_changed),
                affected_services=int(affected_services),
                deploy_date=deploy_date,
                deploy_hour=int(deploy_hour),
                service_metrics=data.service_metrics,
                slo_definitions=data.slo_definitions,
                release_history=data.release_history,
                incidents=data.incidents,
                traffic_calendar=data.traffic_calendar,
            )

            ml_prob, ml_status, ml_warning, ml_metrics, ml_features = estimate_release_probability(
                service=eval_service,
                files_changed=int(files_changed),
                lines_changed=int(lines_changed),
                affected_services=int(affected_services),
                deploy_date=deploy_date,
                deploy_hour=int(deploy_hour),
                service_metrics=data.service_metrics,
                slo_definitions=data.slo_definitions,
                release_history=data.release_history,
                incidents=data.incidents,
            )

            st.session_state["evaluations_run"] += 1
            st.session_state["last_result"] = risk_result
            st.session_state["last_ml_eval"] = {
                "probability": ml_prob,
                "status": ml_status,
                "warning": ml_warning,
                "metrics": ml_metrics,
                "features": ml_features,
            }

            # Build deterministic explanation and optionally run AI summarizer
            explanation = build_deterministic_explanation(
                result=risk_result,
                service=eval_service,
                ml_eval=st.session_state["last_ml_eval"],
            )
            explanation = generate_ai_summary(explanation)
            st.session_state["last_explanation"] = explanation

            st.session_state["last_inputs"] = {
                "service": eval_service,
                "files_changed": int(files_changed),
                "lines_changed": int(lines_changed),
                "affected_services": int(affected_services),
                "deploy_date": deploy_date,
                "deploy_hour": int(deploy_hour),
                "change_type": change_type,
            }

    # Render results if available in session
    if "last_result" in st.session_state and st.session_state["last_result"] is not None:
        result: RiskResult = st.session_state["last_result"]

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📋 Release Evaluation Recommendation</div>', unsafe_allow_html=True)

        if result.data_warnings:
            data_warnings_box(result.data_warnings)

        # Recommendation Banner
        recommendation_banner(result)

        # Hybrid AI & Deterministic Decision Explainability Panel
        if st.session_state.pop("retry_gemini_summary_requested", False) and "last_explanation" in st.session_state:
            with st.spinner("Connecting to Google Gemini to regenerate operational summary..."):
                st.session_state["last_explanation"] = generate_ai_summary(
                    st.session_state["last_explanation"]
                )

        if "last_explanation" not in st.session_state or st.session_state["last_explanation"] is None:
            st.session_state["last_explanation"] = generate_ai_summary(
                build_deterministic_explanation(
                    result=result,
                    service=st.session_state.get("last_inputs", {}).get("service", eval_service),
                    ml_eval=st.session_state.get("last_ml_eval"),
                )
            )
        explanation_panel(st.session_state["last_explanation"])

        # Machine Learning Predictive Intelligence Layer
        if "last_ml_eval" in st.session_state and st.session_state["last_ml_eval"]:
            ml_info = st.session_state["last_ml_eval"]
            ml_prediction_card(
                probability=ml_info.get("probability"),
                status=ml_info.get("status", "unknown"),
                warning=ml_info.get("warning", ""),
                metrics=ml_info.get("metrics"),
            )

        # Target Service Topology & Blast Radius Filament Map
        st.markdown(
            f"""
            <div style="font-size:0.88rem; text-transform:uppercase; letter-spacing:0.06em; font-weight:700; color:{COLORS['blue']}; margin-top:20px; margin-bottom:8px; display:flex; align-items:center; gap:8px;">
              <span>🌌</span> <span>Target Service Topology & Blast Radius Filament Map ({eval_service})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(service_topology_chart(data, selected_service=eval_service, height=360), use_container_width=True)

        # Risk Gauge + Factor Breakdown Chart
        col_g, col_f = st.columns([1, 2])
        with col_g:
            st.plotly_chart(risk_gauge(result.risk_score), use_container_width=True)
        with col_f:
            st.plotly_chart(factor_breakdown_chart(result.factors), use_container_width=True)

        # Factor Evidence Accordions
        factor_evidence_table(result.factors)

        # Prescribed Safeguards Checklist
        safeguards_list(result.suggested_safeguards)

        # Error Budget Details if present
        if result.budget is not None:
            with st.expander("📊 Associated Error Budget Snapshot"):
                b = result.budget
                st.markdown(
                    f"""
                    • **Service:** `{b.service}` | **SLO Target:** `{b.slo_target}%`  
                    • **Budget Consumed:** `{b.budget_consumed_pct:.1f}%` | **Burn Rate:** `{b.burn_rate:.2f}×`  
                    • **Failed Requests / Total:** `{b.failed_requests:,.0f} / {b.total_requests:,.0f}`  
                    • **Remaining Budget:** `{b.remaining_budget_pct:.1f}%`
                    """
                )

        # Next-step CTA
        st.info("💡 **Next Step:** You can simulate mitigation options on the **What-If Analysis** page.")

    synthetic_notice()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WHAT-IF ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav_selection == "🔮 What-If Analysis":
    control_room_header("What-If Scenario Simulation")

    st.markdown(
        "Simulate alternate deployment strategies by adjusting change size, blast radius, "
        "and deployment timing. The risk engine recalculates the score in real time and "
        "quantifies the exact difference between the original and proposed scenarios."
    )

    # Seed initial values from last evaluation or defaults
    last_inputs = st.session_state.get("last_inputs", {})
    def_svc = last_inputs.get("service", services_list[0])
    if def_svc not in services_list:
        def_svc = services_list[0]

    def_files = int(last_inputs.get("files_changed", 14))
    def_lines = int(last_inputs.get("lines_changed", 1200))
    def_aff = int(last_inputs.get("affected_services", 2))
    def_date = last_inputs.get("deploy_date", datetime.date.today())
    def_hour = int(last_inputs.get("deploy_hour", 14))

    col_orig, col_alt = st.columns(2)

    with col_orig:
        st.markdown('<div class="section-header">Scenario A: Original Plan</div>', unsafe_allow_html=True)
        svc_a = st.selectbox("Service", services_list, index=services_list.index(def_svc), key="wia_svc_a")
        files_a = st.number_input("Files Changed", 1, 1000, def_files, key="wia_files_a")
        lines_a = st.number_input("Lines Changed", 1, 100_000, def_lines, step=50, key="wia_lines_a")
        aff_a = st.number_input("Affected Services", 1, 20, def_aff, key="wia_aff_a")
        date_a = st.date_input("Deploy Date", def_date, key="wia_date_a")
        hour_a = st.slider("Deploy Hour", 0, 23, def_hour, format="%02d:00", key="wia_hour_a")
        traffic_window_badge(date_a, hour_a, data.traffic_calendar)

    with col_alt:
        st.markdown('<div class="section-header">Scenario B: Proposed Mitigation</div>', unsafe_allow_html=True)
        svc_b = st.selectbox("Service", services_list, index=services_list.index(def_svc), key="wib_svc_b")
        # Default scenario B with mitigated parameters (halved size, off-peak timing)
        files_b = st.number_input("Files Changed", 1, 1000, max(1, def_files // 2), key="wib_files_b", help="E.g., split into smaller PR")
        lines_b = st.number_input("Lines Changed", 1, 100_000, max(1, def_lines // 2), step=50, key="wib_lines_b")
        aff_b = st.number_input("Affected Services", 1, 20, max(1, def_aff - 1), key="wib_aff_b", help="Contain blast radius")
        date_b = st.date_input("Deploy Date", def_date, key="wib_date_b")
        hour_b = st.slider("Deploy Hour", 0, 23, 7, format="%02d:00", key="wib_hour_b", help="Off-peak deployment (e.g. 07:00 UTC)")
        traffic_window_badge(date_b, hour_b, data.traffic_calendar)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Automatic evaluation on button or parameter change
    if st.button("⚡ Calculate Scenario Comparison", use_container_width=True, type="primary"):
        with st.spinner("Evaluating both scenarios..."):
            res_a = score_release(
                service=svc_a,
                files_changed=int(files_a),
                lines_changed=int(lines_a),
                affected_services=int(aff_a),
                deploy_date=date_a,
                deploy_hour=int(hour_a),
                service_metrics=data.service_metrics,
                slo_definitions=data.slo_definitions,
                release_history=data.release_history,
                incidents=data.incidents,
                traffic_calendar=data.traffic_calendar,
            )
            res_b = score_release(
                service=svc_b,
                files_changed=int(files_b),
                lines_changed=int(lines_b),
                affected_services=int(aff_b),
                deploy_date=date_b,
                deploy_hour=int(hour_b),
                service_metrics=data.service_metrics,
                slo_definitions=data.slo_definitions,
                release_history=data.release_history,
                incidents=data.incidents,
                traffic_calendar=data.traffic_calendar,
            )
            st.session_state["whatif_res_a"] = res_a
            st.session_state["whatif_res_b"] = res_b

    # Render What-If Comparison if available
    if "whatif_res_a" in st.session_state and "whatif_res_b" in st.session_state:
        res_a: RiskResult = st.session_state["whatif_res_a"]
        res_b: RiskResult = st.session_state["whatif_res_b"]
        comparison = compare_scenarios(res_a, res_b)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Delta Summary Card
        whatif_delta_card(comparison)

        # Side-by-side risk gauges & recommendation banners
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown(f"<div style='font-size:0.95rem; font-weight:600; color:{COLORS['text']};'>Scenario A (Original)</div>", unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(res_a.risk_score, "Scenario A"), use_container_width=True)
            recommendation_banner(res_a)
        with c_right:
            st.markdown(f"<div style='font-size:0.95rem; font-weight:600; color:{COLORS['text']};'>Scenario B (Mitigated)</div>", unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(res_b.risk_score, "Scenario B"), use_container_width=True)
            recommendation_banner(res_b)

        # Grouped Factor Comparison Chart
        st.plotly_chart(whatif_comparison_chart(comparison), use_container_width=True)

        # Detailed Factor Delta Breakdown
        if comparison.changed_factors:
            st.markdown('<div class="section-header">🔍 Key Contributing Factors That Changed</div>', unsafe_allow_html=True)
            orig_map = {f.name: f for f in res_a.factors}
            alt_map = {f.name: f for f in res_b.factors}

            diff_rows = []
            for fname in comparison.changed_factors:
                o_f = orig_map.get(fname)
                a_f = alt_map.get(fname)
                if o_f and a_f:
                    f_delta = a_f.score_contribution - o_f.score_contribution
                    diff_rows.append({
                        "Factor": fname,
                        "Scenario A Pts": f"{o_f.score_contribution:.1f} / {o_f.weight_max:.0f}",
                        "Scenario B Pts": f"{a_f.score_contribution:.1f} / {a_f.weight_max:.0f}",
                        "Point Delta": f"{f_delta:+.1f} pts",
                        "Impact Direction": "Improved 📉" if f_delta < 0 else "Worsened 📈",
                        "Mitigation Rationale": a_f.evidence,
                    })

            st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)

        st.caption(
            "ℹ️ **SRE Governance Note:** Adjusting change size or deployment hour reduces operational "
            "and traffic risk. Underlying service reliability (exhausted error budgets or elevated burn rate) "
            "reflects active system health and cannot be solved solely by changing deployment schedules."
        )

    synthetic_notice()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HISTORICAL RELEASES & INCIDENTS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav_selection == "📜 Historical Releases & Incidents":
    control_room_header("Historical Releases & Post-Mortem Incident Audit")

    st.markdown(
        "Audit previous releases alongside associated incidents to discover failure patterns, "
        "correlation with blast radius, and peak traffic vulnerabilities."
    )

    if data.release_history is None:
        st.error("Release history dataset could not be loaded.")
        st.stop()

    inc_df = data.incidents if data.incidents is not None else pd.DataFrame(
        columns=["incident_id", "date", "service", "related_release", "severity", "summary"]
    )

    # Filter Controls
    col_filter_svc, col_filter_inc, col_filter_sev = st.columns(3)
    with col_filter_svc:
        svc_options = ["All"] + services_list
        filter_service = st.selectbox("Filter Service", svc_options, key="hist_filter_svc")

    with col_filter_inc:
        filter_incidents_only = st.checkbox("Show only releases with linked incidents", value=False)

    with col_filter_sev:
        available_severities = ["All"]
        if not inc_df.empty and "severity" in inc_df.columns:
            available_severities += sorted(inc_df["severity"].dropna().unique().tolist())
        filter_severity = st.selectbox("Filter Severity", available_severities, key="hist_filter_sev")

    # Table Generation
    hist_table = releases_with_incidents_table(
        releases=data.release_history,
        incidents=inc_df,
        service=None if filter_service == "All" else filter_service,
    )

    if filter_incidents_only:
        hist_table = hist_table[hist_table["has_incident"]]

    if filter_severity != "All":
        hist_table = hist_table[hist_table["severity"].str.contains(filter_severity, case=False, na=False)]

    display_cols = [
        "release_id",
        "date",
        "service",
        "deployment_hour",
        "change_type",
        "files_changed",
        "lines_changed",
        "affected_services",
        "incident_count",
        "severity",
        "incident_summary",
    ]
    display_cols = [c for c in display_cols if c in hist_table.columns]

    st.dataframe(
        hist_table[display_cols],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Showing {len(hist_table)} releases matching active filters. "
        "Association between releases and incident tickets is correlational and manual; "
        "it does not automatically prove causation."
    )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Visual Analytics: Incident Frequency & Deployment Hour Distribution
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown('<div class="section-header">📊 Releases vs Linked Incidents by Service</div>', unsafe_allow_html=True)
        if not inc_df.empty and data.release_history is not None:
            rel_by_svc = data.release_history.groupby("service").size().reset_index(name="releases")
            inc_by_svc = (
                inc_df.groupby("service")["incident_id"]
                .count()
                .reset_index(name="incidents")
            )
            svc_summary = rel_by_svc.merge(inc_by_svc, on="service", how="left").fillna(0)

            fig_svc_bar = px.bar(
                svc_summary,
                x="service",
                y=["releases", "incidents"],
                barmode="group",
                color_discrete_map={"releases": COLORS["blue"], "incidents": COLORS["coral"]},
                labels={"value": "Total Count", "variable": "Metric"},
            )
            fig_svc_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["text"]),
                height=300,
                xaxis=dict(title="", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.06)"),
                legend=dict(orientation="h", y=1.1, x=0.2),
            )
            st.plotly_chart(fig_svc_bar, use_container_width=True)

    with col_chart2:
        st.markdown('<div class="section-header">⏰ Deployment Hour Distribution</div>', unsafe_allow_html=True)
        if data.release_history is not None:
            rel_df = data.release_history.copy()
            if not inc_df.empty:
                rel_df = rel_df.merge(
                    inc_df[["related_release"]]
                    .assign(linked_incident="Incident Associated")
                    .rename(columns={"related_release": "release_id"}),
                    on="release_id",
                    how="left",
                )
                rel_df["linked_incident"] = rel_df["linked_incident"].fillna("Incident-Free")
            else:
                rel_df["linked_incident"] = "Incident-Free"

            fig_hour = px.histogram(
                rel_df,
                x="deployment_hour",
                color="linked_incident",
                nbins=24,
                color_discrete_map={"Incident Associated": COLORS["coral"], "Incident-Free": COLORS["teal"]},
                labels={"deployment_hour": "Deployment Hour (UTC)", "count": "Releases"},
            )
            fig_hour.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["text"]),
                height=300,
                xaxis=dict(title="Hour of Day (0–23)", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Releases", gridcolor="rgba(255,255,255,0.06)"),
                legend=dict(orientation="h", y=1.1, x=0.2),
            )
            st.plotly_chart(fig_hour, use_container_width=True)

    synthetic_notice()
