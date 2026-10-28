"""
app.py — ReleaseGuard: AI Ops Release Safety Advisor
=====================================================

Main Streamlit application entry point.

Run with:
    streamlit run app.py

Sections:
  A. Overview dashboard
  B. Service health
  C. Evaluate a new release
  D. Release recommendation
  E. Historical release analysis
  F. What-if analysis

All data is synthetic hackathon data — NOT real production observations.

Author: ReleaseGuard Team
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="ReleaseGuard — AI Ops Release Safety Advisor",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.dashboard_components import (
    budget_gauge,
    data_warnings_box,
    error_rate_trend,
    factor_breakdown_chart,
    factor_evidence_table,
    inject_css,
    metric_card,
    COLORS,
    recommendation_banner,
    releases_with_incidents_table,
    request_volume_chart,
    risk_gauge,
    safeguards_list,
    synthetic_notice,
    whatif_comparison_chart,
)
from src.data_loader import LoadedData, load_all_data, get_services
from src.risk_engine import (
    BudgetResult,
    THRESHOLD_GO,
    THRESHOLD_SAFEGUARDS,
    WEIGHTS,
    compare_scenarios,
    compute_budget_for_service,
    score_release,
)

# ── CSS ────────────────────────────────────────────────────────────────────────
inject_css()

# ── Data loading (cached) ──────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def get_data() -> LoadedData:
    return load_all_data()


data = get_data()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 16px 0 8px;">
          <div style="font-size:2rem;">🛡</div>
          <div style="font-size:1.2rem; font-weight:700; color:{COLORS['blue']};">ReleaseGuard</div>
          <div style="font-size:0.78rem; color:{COLORS['subtext']};">AI Ops Release Safety Advisor</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "🔍 Service Health",
            "🚀 Evaluate Release",
            "📊 Historical Analysis",
            "🔮 What-If Analysis",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Data status indicator
    if data.is_usable:
        st.success("✓ All datasets loaded")
    else:
        st.error("✗ Data loading failed")

    if data.errors:
        with st.expander("❌ Data Errors", expanded=True):
            for e in data.errors:
                st.error(e)

    if data.warnings:
        with st.expander(f"⚠ {len(data.warnings)} Warning(s)"):
            for w in data.warnings:
                st.warning(w)

    synthetic_notice()

    # Config panel
    with st.expander("⚙ Risk Weights (read-only)"):
        for name, weight in WEIGHTS.items():
            st.caption(f"{name}: **{weight}** pts")
        st.caption(f"GO threshold: ≤ {THRESHOLD_GO}")
        st.caption(f"SAFEGUARDS threshold: ≤ {THRESHOLD_SAFEGUARDS}")
        st.caption(f"WAIT threshold: > {THRESHOLD_SAFEGUARDS}")


# ── Guard: data must be usable for most pages ──────────────────────────────────

if not data.is_usable:
    st.error(
        "**ReleaseGuard cannot start because critical datasets failed to load.**\n\n"
        "Please check the errors in the sidebar and ensure the following files are present "
        "in the `data/` directory:\n"
        "- `slo_definitions.csv`\n"
        "- `service_metrics.csv`\n"
        "- `release_history.csv`\n"
        "- `incidents.csv` (optional)\n"
        "- `traffic_calendar.csv` (optional)\n"
    )
    st.stop()

services_list = get_services(data)


# ═══════════════════════════════════════════════════════════════════════════════
# A. OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown(
        f"<h1 style='color:{COLORS['blue']}; margin-bottom:4px;'>🛡 ReleaseGuard</h1>"
        f"<div style='color:{COLORS['subtext']}; margin-bottom:24px; font-size:1rem;'>"
        "AI Ops Release Safety Advisor — <em>Is this release safe to ship right now?</em></div>",
        unsafe_allow_html=True,
    )

    # ── Top-level metrics ─────────────────────────────────────────────────────
    num_services  = len(services_list)
    num_incidents = len(data.incidents) if data.incidents is not None else 0
    num_releases  = len(data.release_history) if data.release_history is not None else 0

    # Compute budget for each service, identify needing attention
    attention_services = []
    budgets: dict[str, BudgetResult] = {}
    for svc in services_list:
        if data.service_metrics is not None and data.slo_definitions is not None:
            b = compute_budget_for_service(svc, data.service_metrics, data.slo_definitions)
            if b is not None:
                budgets[svc] = b
                if b.budget_consumed_pct >= 80 or b.burn_rate >= 2.0:
                    attention_services.append(svc)

    col1, col2, col3, col4 = st.columns(4)
    metric_card("Services Monitored",  num_services,  sub="All availability SLOs", col=col1)
    metric_card("Needing Attention",   len(attention_services),
                sub="≥80% budget consumed or burn rate ≥2×", col=col2)
    metric_card("Recorded Incidents",  num_incidents, sub="Linked to releases", col=col3)
    metric_card("Historical Releases", num_releases,  sub="In dataset", col=col4)

    st.markdown("---")

    # ── Service health summary table ──────────────────────────────────────────
    st.markdown(
        f"<div class='section-header'>Service Health Summary</div>",
        unsafe_allow_html=True,
    )

    if budgets:
        rows = []
        for svc, b in budgets.items():
            slo_row = data.slo_definitions[data.slo_definitions["service"] == svc]
            slo_target = float(slo_row.iloc[0]["slo_target"]) if not slo_row.empty else None

            if b.budget_consumed_pct >= 100:
                status = "🔴 Exhausted"
            elif b.budget_consumed_pct >= 80:
                status = "🟠 Critical"
            elif b.budget_consumed_pct >= 50:
                status = "🟡 Warning"
            else:
                status = "🟢 Healthy"

            rows.append({
                "Service":            svc,
                "SLO Target (%)":     slo_target,
                "Total Requests":     f"{b.total_requests:,.0f}",
                "Failed Requests":    f"{b.failed_requests:,.0f}",
                "Error Rate (%)":     f"{b.actual_error_rate*100:.4f}%",
                "Budget Consumed (%)": f"{b.budget_consumed_pct:.1f}%",
                "Remaining Budget (%)": f"{b.remaining_budget_pct:.1f}%",
                "Burn Rate":          f"{b.burn_rate:.2f}×",
                "Status":             status,
            })

        summary_df = pd.DataFrame(rows)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Error-budget calculations unavailable (SLO definitions or metrics missing).")

    synthetic_notice()

    # ── Error rate sparklines ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<div class='section-header'>Error Rate Trends (Last 30 Days)</div>",
        unsafe_allow_html=True,
    )

    if data.service_metrics is not None:
        cols = st.columns(min(3, len(services_list)))
        for i, svc in enumerate(services_list):
            slo_row = data.slo_definitions[data.slo_definitions["service"] == svc] \
                if data.slo_definitions is not None else pd.DataFrame()
            slo_t = float(slo_row.iloc[0]["slo_target"]) if not slo_row.empty else None
            with cols[i % 3]:
                fig = error_rate_trend(svc, data.service_metrics, slo_t)
                fig.update_layout(height=200, title=dict(text=svc, font=dict(size=11)),
                                  margin=dict(l=20, r=10, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# B. SERVICE HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Service Health":
    st.markdown(
        f"<h2 style='color:{COLORS['blue']};'>🔍 Service Health</h2>",
        unsafe_allow_html=True,
    )

    selected_svc = st.selectbox("Select Service", services_list, key="health_svc")

    slo_row = (
        data.slo_definitions[data.slo_definitions["service"] == selected_svc]
        if data.slo_definitions is not None
        else pd.DataFrame()
    )

    if slo_row.empty:
        st.warning(f"No SLO definition found for '{selected_svc}'.")
    else:
        slo_target = float(slo_row.iloc[0]["slo_target"])
        sli_type   = slo_row.iloc[0].get("sli_type", "availability")
        description = slo_row.iloc[0].get("description", "")

        st.markdown(
            f"**SLO:** {slo_target}% availability &nbsp;|&nbsp; "
            f"**SLI type:** {sli_type} &nbsp;|&nbsp; "
            f"**Description:** {description}",
        )

    # Budget calculation
    budget = None
    if data.service_metrics is not None and data.slo_definitions is not None:
        budget = compute_budget_for_service(
            selected_svc, data.service_metrics, data.slo_definitions
        )

    if budget is None:
        st.error(f"Cannot compute error budget for '{selected_svc}'. Check data warnings.")
    else:
        if budget.warning:
            data_warnings_box([budget.warning])

        # Metric row
        col1, col2, col3, col4 = st.columns(4)
        metric_card("Total Requests",     f"{budget.total_requests:,.0f}",
                    sub=f"{budget.window_days} days", col=col1)
        metric_card("Failed Requests",    f"{budget.failed_requests:,.0f}",
                    sub=f"Error rate: {budget.actual_error_rate*100:.4f}%", col=col2)
        metric_card("Budget Consumed",    f"{budget.budget_consumed_pct:.1f}%",
                    sub=f"Remaining: {budget.remaining_budget_pct:.1f}%", col=col3)
        metric_card("Burn Rate",          f"{budget.burn_rate:.2f}×",
                    sub="Actual ÷ Allowed error rate", col=col4)

        # Charts
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.plotly_chart(budget_gauge(budget), use_container_width=True)

        with col_b:
            if data.service_metrics is not None:
                slo_t = budget.slo_target if not slo_row.empty else None
                st.plotly_chart(
                    error_rate_trend(selected_svc, data.service_metrics, slo_t),
                    use_container_width=True,
                )

        if data.service_metrics is not None:
            st.plotly_chart(
                request_volume_chart(selected_svc, data.service_metrics),
                use_container_width=True,
            )

    # Incident history for service
    st.markdown(
        f"<div class='section-header'>Recent Incidents for {selected_svc}</div>",
        unsafe_allow_html=True,
    )
    if data.incidents is not None:
        svc_incidents = data.incidents[data.incidents["service"] == selected_svc].copy()
        svc_incidents = svc_incidents.sort_values("date", ascending=False)
        if svc_incidents.empty:
            st.success(f"No incidents recorded for '{selected_svc}' in the dataset.")
        else:
            st.dataframe(svc_incidents, use_container_width=True, hide_index=True)
            st.caption(
                "Note: incidents listed here are associated with releases by the 'related_release' "
                "field. Association does not imply causation."
            )
    else:
        st.info("Incidents dataset not loaded.")

    synthetic_notice()


# ═══════════════════════════════════════════════════════════════════════════════
# C + D. EVALUATE A NEW RELEASE → RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🚀 Evaluate Release":
    st.markdown(
        f"<h2 style='color:{COLORS['blue']};'>🚀 Evaluate a New Release</h2>",
        unsafe_allow_html=True,
    )

    with st.form("release_form"):
        col1, col2 = st.columns(2)

        with col1:
            svc = st.selectbox("Target Service", services_list, key="eval_svc")

            files_changed = st.number_input(
                "Files Changed",
                min_value=1,
                max_value=1000,
                value=6,
                step=1,
                help="Number of files modified in this release.",
            )
            lines_changed = st.number_input(
                "Lines Changed (additions + deletions)",
                min_value=1,
                max_value=100_000,
                value=500,
                step=50,
                help="Total lines added and removed.",
            )

        with col2:
            affected_svcs = st.number_input(
                "Affected Services",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                help="Number of services this release touches (including the primary).",
            )
            deploy_date = st.date_input(
                "Proposed Deployment Date",
                value=datetime.date.today(),
                help="Date you plan to deploy.",
            )
            deploy_hour = st.slider(
                "Proposed Deployment Hour (24h)",
                min_value=0,
                max_value=23,
                value=14,
                format="%d:00",
                help="Hour of deployment in 24-hour format.",
            )

        change_type = st.selectbox(
            "Change Type",
            ["bugfix", "feature", "refactor", "dependency_update", "hotfix", "config", "other"],
            help="Type of change (informational only; not used in risk scoring).",
        )

        st.markdown("---")
        submitted = st.form_submit_button("🔍 Evaluate Risk", use_container_width=True)

    # ── Show traffic calendar context ─────────────────────────────────────────
    if data.traffic_calendar is not None:
        cal_row = data.traffic_calendar[
            data.traffic_calendar["date"].dt.date == deploy_date
        ]
        if not cal_row.empty:
            row = cal_row.iloc[0]
            flags = []
            if row["is_peak"]:
                flags.append("📈 Peak Traffic Day")
            if row["freeze_period"]:
                flags.append("🚫 Freeze Period")
            if flags:
                st.warning(f"Calendar: {' | '.join(flags)} — Event: {row.get('event','')}")
            else:
                st.info(f"Calendar: Normal day — Event: {row.get('event','')}")
        else:
            st.caption(f"Date {deploy_date} not found in traffic calendar.")

    if submitted:
        with st.spinner("Calculating risk score..."):
            result = score_release(
                service=svc,
                files_changed=int(files_changed),
                lines_changed=int(lines_changed),
                affected_services=int(affected_svcs),
                deploy_date=deploy_date,
                deploy_hour=int(deploy_hour),
                service_metrics=data.service_metrics,
                slo_definitions=data.slo_definitions,
                release_history=data.release_history,
                incidents=data.incidents,
                traffic_calendar=data.traffic_calendar,
            )

        # Store for What-If page
        st.session_state["last_result"] = result
        st.session_state["last_inputs"] = {
            "svc": svc,
            "files_changed": int(files_changed),
            "lines_changed": int(lines_changed),
            "affected_svcs": int(affected_svcs),
            "deploy_date": deploy_date,
            "deploy_hour": int(deploy_hour),
        }

        # ── D. Release Recommendation ─────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            f"<h3 style='color:{COLORS['text']};'>📋 Release Recommendation</h3>",
            unsafe_allow_html=True,
        )

        data_warnings_box(result.data_warnings)
        recommendation_banner(result)

        col_gauge, col_chart = st.columns([1, 2])
        with col_gauge:
            st.plotly_chart(risk_gauge(result.risk_score), use_container_width=True)
        with col_chart:
            st.plotly_chart(
                factor_breakdown_chart(result.factors), use_container_width=True
            )

        factor_evidence_table(result.factors)

        safeguards_list(result.suggested_safeguards)

        # Show budget details
        if result.budget is not None:
            with st.expander("📊 Error Budget Details"):
                b = result.budget
                st.markdown(f"""
| Metric | Value |
|--------|-------|
| SLO Target | {b.slo_target}% |
| Allowed Error Rate | {b.allowed_error_rate:.6f} ({b.allowed_error_rate*100:.4f}%) |
| Actual Error Rate | {b.actual_error_rate:.6f} ({b.actual_error_rate*100:.4f}%) |
| Total Requests (window) | {b.total_requests:,.0f} |
| Failed Requests (window) | {b.failed_requests:,.0f} |
| Allowed Errors (window) | {b.allowed_errors:.1f} |
| Budget Consumed | {b.budget_consumed_pct:.1f}% |
| Budget Remaining | {b.remaining_budget_pct:.1f}% |
| Burn Rate | {b.burn_rate:.3f}× |
""")
                st.caption(
                    "Budget figures are aggregated over the full available metric window. "
                    "Aggregating daily observations may obscure intra-day patterns."
                )

        synthetic_notice()

        st.success(
            "✓ Evaluation complete. Navigate to **What-If Analysis** to explore alternatives."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# E. HISTORICAL RELEASE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Historical Analysis":
    st.markdown(
        f"<h2 style='color:{COLORS['blue']};'>📊 Historical Release Analysis</h2>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Association between a release and a following incident does NOT prove causation. "
        "These records show only that an incident was manually linked to a release."
    )

    if data.release_history is None:
        st.error("Release history dataset not loaded.")
        st.stop()

    inc_df = data.incidents if data.incidents is not None else pd.DataFrame(
        columns=["incident_id", "related_release", "severity"]
    )

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        all_svcs = ["All"] + services_list
        hist_svc = st.selectbox("Filter by Service", all_svcs, key="hist_svc")
    with col2:
        show_incidents_only = st.checkbox("Show only releases with associated incidents")

    filter_svc = None if hist_svc == "All" else hist_svc

    merged = releases_with_incidents_table(
        data.release_history, inc_df, service=filter_svc
    )
    if show_incidents_only:
        merged = merged[merged["has_incident"]]

    # Display table
    display_cols = [
        "release_id", "date", "service", "deployment_hour",
        "files_changed", "lines_changed", "affected_services",
        "change_type", "incident_count", "severity",
    ]
    display_cols = [c for c in display_cols if c in merged.columns]

    st.dataframe(
        merged[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    # Incidents per service bar chart
    st.markdown(
        f"<div class='section-header'>Incident-Associated Releases by Service</div>",
        unsafe_allow_html=True,
    )

    if not inc_df.empty and data.release_history is not None:
        svc_inc = (
            data.release_history
            .merge(
                inc_df[["related_release", "severity"]].rename(
                    columns={"related_release": "release_id"}
                ),
                on="release_id",
                how="left",
            )
        )
        svc_inc_count = (
            svc_inc.groupby("service")["severity"]
            .count()
            .reset_index()
            .rename(columns={"severity": "incidents"})
        )
        total_by_svc = (
            data.release_history.groupby("service")
            .size()
            .reset_index(name="releases")
        )
        svc_inc_count = svc_inc_count.merge(total_by_svc, on="service")
        svc_inc_count["incident_rate"] = svc_inc_count["incidents"] / svc_inc_count["releases"]

        fig = px.bar(
            svc_inc_count,
            x="service",
            y=["releases", "incidents"],
            barmode="group",
            color_discrete_map={"releases": COLORS["blue"], "incidents": COLORS["coral"]},
            labels={"value": "Count", "variable": ""},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
            height=320,
            title="Releases vs Incident-Associated Releases per Service",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Deployment hour distribution
    st.markdown(
        f"<div class='section-header'>Deployment Hour Distribution</div>",
        unsafe_allow_html=True,
    )

    if data.release_history is not None:
        hr_df = data.release_history.copy()
        if not inc_df.empty:
            hr_df = hr_df.merge(
                inc_df[["related_release"]].assign(had_incident=True)
                .rename(columns={"related_release": "release_id"}),
                on="release_id",
                how="left",
            )
            hr_df["had_incident"] = hr_df["had_incident"].fillna(False)
        else:
            hr_df["had_incident"] = False

        fig2 = px.histogram(
            hr_df,
            x="deployment_hour",
            color="had_incident",
            nbins=24,
            color_discrete_map={True: COLORS["coral"], False: COLORS["teal"]},
            labels={"deployment_hour": "Hour of Day", "count": "Releases",
                    "had_incident": "Had Associated Incident"},
            title="Deployment Hours (releases with and without associated incidents)",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Correlation between deployment hour and incidents is observational only. "
            "This dataset is synthetic and should not be used to draw causal conclusions."
        )

    synthetic_notice()


# ═══════════════════════════════════════════════════════════════════════════════
# F. WHAT-IF ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 What-If Analysis":
    st.markdown(
        f"<h2 style='color:{COLORS['blue']};'>🔮 What-If Analysis</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "Adjust release parameters to see how the risk score changes. "
        "The same risk engine is used for both scenarios. "
        "Note: changing deployment timing does NOT reduce underlying service reliability risk."
    )

    # If we have a previous evaluation, seed the form with those values
    last = st.session_state.get("last_inputs", {})
    last_result = st.session_state.get("last_result", None)

    if last_result is not None:
        st.info(
            f"Original evaluation loaded: **{last.get('svc','?')}** — "
            f"Score {last_result.risk_score:.0f} / 100, "
            f"Recommendation: **{last_result.recommendation}**"
        )

    col_orig, col_alt = st.columns(2)

    with col_orig:
        st.markdown(
            f"<div class='section-header'>Scenario A (Original)</div>",
            unsafe_allow_html=True,
        )
        svc_a = st.selectbox(
            "Service", services_list,
            index=services_list.index(last.get("svc", services_list[0]))
                if last.get("svc") in services_list else 0,
            key="wia_svc",
        )
        files_a   = st.number_input("Files Changed", 1, 1000, int(last.get("files_changed", 6)), key="wia_files")
        lines_a   = st.number_input("Lines Changed", 1, 100000, int(last.get("lines_changed", 500)), step=50, key="wia_lines")
        svcs_a    = st.number_input("Affected Services", 1, 20, int(last.get("affected_svcs", 1)), key="wia_svcs")
        date_a    = st.date_input("Deploy Date", last.get("deploy_date", datetime.date.today()), key="wia_date")
        hour_a    = st.slider("Deploy Hour", 0, 23, int(last.get("deploy_hour", 14)), key="wia_hour", format="%d:00")

    with col_alt:
        st.markdown(
            f"<div class='section-header'>Scenario B (Alternative)</div>",
            unsafe_allow_html=True,
        )
        svc_b    = st.selectbox("Service", services_list,
                                index=services_list.index(last.get("svc", services_list[0]))
                                    if last.get("svc") in services_list else 0,
                                key="wib_svc")
        files_b  = st.number_input("Files Changed", 1, 1000, max(1, int(last.get("files_changed", 6)) // 2), key="wib_files")
        lines_b  = st.number_input("Lines Changed", 1, 100000, max(1, int(last.get("lines_changed", 500)) // 2), step=50, key="wib_lines")
        svcs_b   = st.number_input("Affected Services", 1, 20, int(last.get("affected_svcs", 1)), key="wib_svcs")
        date_b   = st.date_input("Deploy Date", last.get("deploy_date", datetime.date.today()), key="wib_date")
        hour_b   = st.slider("Deploy Hour", 0, 23, 7, key="wib_hour", format="%d:00",
                             help="Try off-peak hours (before 09:00 or after 18:00).")

    if st.button("⚡ Compare Scenarios", use_container_width=True, type="primary"):
        with st.spinner("Calculating..."):
            result_a = score_release(
                service=svc_a,
                files_changed=int(files_a),
                lines_changed=int(lines_a),
                affected_services=int(svcs_a),
                deploy_date=date_a,
                deploy_hour=int(hour_a),
                service_metrics=data.service_metrics,
                slo_definitions=data.slo_definitions,
                release_history=data.release_history,
                incidents=data.incidents,
                traffic_calendar=data.traffic_calendar,
            )
            result_b = score_release(
                service=svc_b,
                files_changed=int(files_b),
                lines_changed=int(lines_b),
                affected_services=int(svcs_b),
                deploy_date=date_b,
                deploy_hour=int(hour_b),
                service_metrics=data.service_metrics,
                slo_definitions=data.slo_definitions,
                release_history=data.release_history,
                incidents=data.incidents,
                traffic_calendar=data.traffic_calendar,
            )

        comparison = compare_scenarios(result_a, result_b)

        # Summary banner
        delta = comparison.score_delta
        if delta < -1:
            st.success(f"✅ Scenario B is **safer**: score improved by {abs(delta):.1f} pts")
        elif delta > 1:
            st.warning(f"⚠ Scenario B is **riskier**: score increased by {delta:.1f} pts")
        else:
            st.info("Scenarios have nearly identical risk scores.")

        # Score comparison
        col_a2, col_b2 = st.columns(2)
        with col_a2:
            st.plotly_chart(risk_gauge(result_a.risk_score, "Scenario A"), use_container_width=True)
            recommendation_banner(result_a)
        with col_b2:
            st.plotly_chart(risk_gauge(result_b.risk_score, "Scenario B"), use_container_width=True)
            recommendation_banner(result_b)

        # Factor comparison chart
        st.plotly_chart(
            whatif_comparison_chart(comparison), use_container_width=True
        )

        # Changed factors detail
        if comparison.changed_factors:
            st.markdown(
                f"<div class='section-header'>Factors That Changed</div>",
                unsafe_allow_html=True,
            )
            orig_map = {f.name: f for f in result_a.factors}
            alt_map  = {f.name: f for f in result_b.factors}
            for fname in comparison.changed_factors:
                o = orig_map.get(fname)
                a = alt_map.get(fname)
                if o and a:
                    delta_f = a.score_contribution - o.score_contribution
                    arrow = "↓" if delta_f < 0 else "↑"
                    color = COLORS["teal"] if delta_f < 0 else COLORS["coral"]
                    st.markdown(
                        f"<span style='color:{color};'>{arrow} **{fname}**: "
                        f"{o.score_contribution:.1f} → {a.score_contribution:.1f} pts</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"B: {a.evidence}")

        # Safeguard caveat
        st.markdown(
            "> **Limitation:** Changing deployment timing reduces only timing-related risk. "
            "If the service's error budget is exhausted or burn rate is high, those risks "
            "remain regardless of when you deploy.",
        )

        synthetic_notice()
