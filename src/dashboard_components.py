"""
dashboard_components.py — Reusable Streamlit UI Components for ReleaseGuard

Provides styled cards, charts, status indicators, and table renderers
used across all dashboard sections.

Design system:
  - Dark navy background (#0d1117)
  - Blue (#58a6ff), violet (#bc8cff), teal (#56d364), coral (#f78166) accents
  - Rounded glassmorphism cards
  - Consistent spacing and typography

All chart data comes from validated DataFrames; no values are invented here.

Author: ReleaseGuard Team
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.risk_engine import BudgetResult, RiskFactor, RiskResult, WhatIfResult

# ── Colour palette ─────────────────────────────────────────────────────────────
COLORS = {
    "bg":       "#0d1117",
    "card":     "#161b22",
    "border":   "#30363d",
    "blue":     "#58a6ff",
    "violet":   "#bc8cff",
    "teal":     "#56d364",
    "coral":    "#f78166",
    "amber":    "#e3b341",
    "text":     "#e6edf3",
    "subtext":  "#8b949e",
    "go":       "#1a7f37",
    "go_text":  "#3fb950",
    "wait":     "#b62324",
    "wait_text": "#f85149",
    "safeguard":  "#7d4e00",
    "safeguard_text": "#e3b341",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=COLORS["border"], showgrid=True),
    yaxis=dict(gridcolor=COLORS["border"], showgrid=True),
)

# ── CSS injection ──────────────────────────────────────────────────────────────

GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
}}

/* Hide default Streamlit header decorations */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* Metric cards */
.metric-card {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color 0.2s ease;
}}
.metric-card:hover {{ border-color: {COLORS['blue']}; }}
.metric-label {{
    font-size: 0.78rem;
    font-weight: 500;
    color: {COLORS['subtext']};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}}
.metric-value {{
    font-size: 2rem;
    font-weight: 700;
    color: {COLORS['text']};
    line-height: 1.1;
}}
.metric-sub {{
    font-size: 0.82rem;
    color: {COLORS['subtext']};
    margin-top: 4px;
}}

/* Recommendation banners */
.reco-go {{
    background: {COLORS['go']};
    border: 2px solid {COLORS['go_text']};
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 16px;
}}
.reco-wait {{
    background: {COLORS['wait']};
    border: 2px solid {COLORS['wait_text']};
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 16px;
}}
.reco-safeguard {{
    background: {COLORS['safeguard']};
    border: 2px solid {COLORS['safeguard_text']};
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 16px;
}}
.reco-title {{
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}}
.reco-reason {{
    font-size: 0.9rem;
    margin-top: 8px;
    opacity: 0.9;
}}

/* Section headers */
.section-header {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {COLORS['blue']};
    border-bottom: 1px solid {COLORS['border']};
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 8px;
}}

/* Factor pill */
.factor-row {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
    padding: 10px 14px;
    background: {COLORS['card']};
    border-radius: 8px;
    border: 1px solid {COLORS['border']};
}}
.factor-bar-bg {{
    flex: 1;
    background: {COLORS['border']};
    border-radius: 4px;
    height: 8px;
}}
.factor-bar {{
    height: 8px;
    border-radius: 4px;
    transition: width 0.4s ease;
}}

/* Data warning */
.data-warning {{
    background: rgba(227,179,65,0.1);
    border: 1px solid {COLORS['amber']};
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.84rem;
    color: {COLORS['amber']};
    margin-bottom: 8px;
}}
.data-synthetic {{
    font-size: 0.75rem;
    color: {COLORS['subtext']};
    font-style: italic;
    text-align: center;
    margin-top: 4px;
}}

/* Safeguard list */
.safeguard-item {{
    padding: 8px 12px;
    margin-bottom: 6px;
    background: rgba(88,166,255,0.07);
    border-left: 3px solid {COLORS['blue']};
    border-radius: 0 6px 6px 0;
    font-size: 0.88rem;
}}
</style>
"""


def inject_css() -> None:
    """Inject global CSS into the Streamlit app."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ── Metric card ───────────────────────────────────────────────────────────────


def metric_card(label: str, value: Any, sub: str = "", col=None) -> None:
    """Render a styled metric card."""
    html = f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {'<div class="metric-sub">' + sub + '</div>' if sub else ''}
    </div>
    """
    target = col if col is not None else st
    target.markdown(html, unsafe_allow_html=True)


# ── Recommendation banner ─────────────────────────────────────────────────────


def recommendation_banner(result: RiskResult) -> None:
    """Render the GO / WAIT / GO WITH SAFEGUARDS recommendation."""
    reco = result.recommendation
    score = result.risk_score
    reason = result.recommendation_reason

    if reco == "GO":
        css_class = "reco-go"
        icon = "✅"
        score_color = COLORS["teal"]
    elif reco == "WAIT":
        css_class = "reco-wait"
        icon = "🛑"
        score_color = COLORS["coral"]
    else:
        css_class = "reco-safeguard"
        icon = "⚠️"
        score_color = COLORS["amber"]

    st.markdown(
        f"""
        <div class="{css_class}">
          <div class="reco-title">{icon} {reco}</div>
          <div class="reco-reason">{reason}</div>
          <div style="margin-top:10px; font-size:0.95rem; font-weight:600;">
            Risk Score: <span style="font-size:1.5rem; color:{score_color};">{score:.0f}</span> / 100
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Risk gauge ────────────────────────────────────────────────────────────────


def risk_gauge(score: float, title: str = "Risk Score") -> go.Figure:
    """Return a Plotly gauge chart for the risk score."""
    if score <= 35:
        color = COLORS["teal"]
    elif score <= 65:
        color = COLORS["amber"]
    else:
        color = COLORS["coral"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title, "font": {"color": COLORS["text"], "size": 14}},
            number={"font": {"color": color, "size": 40}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": COLORS["subtext"],
                    "tickwidth": 1,
                    "tickfont": {"color": COLORS["subtext"]},
                },
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": COLORS["card"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35],  "color": "rgba(86,211,100,0.15)"},
                    {"range": [35, 65], "color": "rgba(227,179,65,0.15)"},
                    {"range": [65, 100],"color": "rgba(248,81,73,0.15)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ── Factor breakdown ──────────────────────────────────────────────────────────


def factor_breakdown_chart(factors: list[RiskFactor]) -> go.Figure:
    """Horizontal bar chart showing each factor's contribution."""
    names  = [f.name for f in factors]
    contribs = [f.score_contribution for f in factors]
    maxes  = [f.weight_max for f in factors]
    pcts   = [c / m * 100 if m > 0 else 0 for c, m in zip(contribs, maxes)]

    colors_list = []
    for p in pcts:
        if p >= 70:
            colors_list.append(COLORS["coral"])
        elif p >= 40:
            colors_list.append(COLORS["amber"])
        else:
            colors_list.append(COLORS["teal"])

    fig = go.Figure(
        go.Bar(
            x=pcts,
            y=names,
            orientation="h",
            marker_color=colors_list,
            text=[f"{p:.0f}%" for p in pcts],
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=11),
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        title=dict(text="Risk Factor Contributions (% of max weight)", font=dict(size=13)),
        xaxis=dict(title="% of factor weight", range=[0, 115], **PLOTLY_LAYOUT["xaxis"]),
        yaxis=dict(title="", **PLOTLY_LAYOUT["yaxis"]),
        showlegend=False,
    )
    return fig


def factor_evidence_table(factors: list[RiskFactor]) -> None:
    """Render factor evidence as a styled expandable list."""
    st.markdown('<div class="section-header">Risk Factor Details</div>', unsafe_allow_html=True)
    for f in sorted(factors, key=lambda x: x.score_contribution, reverse=True):
        pct = f.score_contribution / f.weight_max * 100 if f.weight_max > 0 else 0
        if pct >= 70:
            icon = "🔴"
        elif pct >= 40:
            icon = "🟡"
        else:
            icon = "🟢"

        with st.expander(f"{icon} {f.name}  —  {f.score_contribution:.1f} / {f.weight_max:.1f} pts"):
            st.write(f.evidence)


# ── Error-budget chart ────────────────────────────────────────────────────────


def budget_gauge(budget: BudgetResult) -> go.Figure:
    """Donut chart showing remaining vs consumed error budget."""
    consumed  = min(budget.budget_consumed_pct, 100)
    remaining = max(0, 100 - consumed)

    if consumed >= 80:
        consumed_color = COLORS["coral"]
    elif consumed >= 50:
        consumed_color = COLORS["amber"]
    else:
        consumed_color = COLORS["teal"]

    fig = go.Figure(
        go.Pie(
            values=[consumed, remaining],
            labels=["Consumed", "Remaining"],
            hole=0.65,
            marker_colors=[consumed_color, COLORS["border"]],
            textinfo="none",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"{consumed:.0f}%",
        x=0.5, y=0.55,
        font=dict(size=26, color=consumed_color, family="Inter"),
        showarrow=False,
    )
    fig.add_annotation(
        text="consumed",
        x=0.5, y=0.40,
        font=dict(size=12, color=COLORS["subtext"], family="Inter"),
        showarrow=False,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            font=dict(color=COLORS["text"]),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            y=-0.1,
        ),
        height=280,
        margin=dict(l=0, r=0, t=20, b=40),
    )
    return fig


# ── Metrics trend chart ───────────────────────────────────────────────────────


def error_rate_trend(service: str, metrics: pd.DataFrame, slo_target: Optional[float] = None) -> go.Figure:
    """Line chart of daily error rate with SLO target line."""
    svc = metrics[metrics["service"] == service].copy()
    svc = svc.sort_values("date")
    svc = svc[svc["total_requests"] > 0].copy()
    svc["error_rate_pct"] = (svc["failed_requests"] / svc["total_requests"]) * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=svc["date"],
            y=svc["error_rate_pct"],
            mode="lines+markers",
            name="Error Rate (%)",
            line=dict(color=COLORS["blue"], width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)",
            hovertemplate="Date: %{x|%b %d}<br>Error rate: %{y:.4f}%<extra></extra>",
        )
    )

    if slo_target is not None:
        allowed_error_pct = (1 - slo_target / 100) * 100
        fig.add_hline(
            y=allowed_error_pct,
            line_dash="dash",
            line_color=COLORS["coral"],
            annotation_text=f"SLO limit ({allowed_error_pct:.4f}%)",
            annotation_font_color=COLORS["coral"],
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        title=dict(text=f"Daily Error Rate — {service}", font=dict(size=13)),
        xaxis=dict(title="Date", **PLOTLY_LAYOUT["xaxis"]),
        yaxis=dict(title="Error Rate (%)", **PLOTLY_LAYOUT["yaxis"]),
    )
    return fig


def request_volume_chart(service: str, metrics: pd.DataFrame) -> go.Figure:
    """Stacked bar: total vs failed requests per day."""
    svc = metrics[metrics["service"] == service].copy().sort_values("date")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=svc["date"],
            y=svc["total_requests"] - svc["failed_requests"],
            name="Successful",
            marker_color=COLORS["teal"],
            hovertemplate="Date: %{x|%b %d}<br>Successful: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=svc["date"],
            y=svc["failed_requests"],
            name="Failed",
            marker_color=COLORS["coral"],
            hovertemplate="Date: %{x|%b %d}<br>Failed: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        height=280,
        title=dict(text=f"Daily Requests — {service}", font=dict(size=13)),
        xaxis=dict(title="Date", **PLOTLY_LAYOUT["xaxis"]),
        yaxis=dict(title="Requests", **PLOTLY_LAYOUT["yaxis"]),
    )
    return fig


# ── Historical releases table ─────────────────────────────────────────────────


def releases_with_incidents_table(
    releases: pd.DataFrame,
    incidents: pd.DataFrame,
    service: Optional[str] = None,
) -> pd.DataFrame:
    """Merge releases with incident data; return a display-ready DataFrame."""
    df = releases.copy()
    if service:
        df = df[df["service"] == service]

    # Count linked incidents per release
    inc_counts = (
        incidents.groupby("related_release")
        .agg(
            incident_count=("incident_id", "count"),
            max_severity=("severity", lambda x: x.tolist()),
        )
        .reset_index()
        .rename(columns={"related_release": "release_id"})
    )

    df = df.merge(inc_counts, on="release_id", how="left")
    df["incident_count"] = df["incident_count"].fillna(0).astype(int)
    df["has_incident"]   = df["incident_count"] > 0

    # Friendly severity string
    def fmt_sev(sev_list):
        if not isinstance(sev_list, list) or len(sev_list) == 0:
            return "—"
        return ", ".join(sorted(set(sev_list)))

    df["severity"] = df["max_severity"].apply(fmt_sev)
    df = df.drop(columns=["max_severity"], errors="ignore")

    return df.sort_values("date", ascending=False)


# ── What-if comparison ────────────────────────────────────────────────────────


def whatif_comparison_chart(comparison: WhatIfResult) -> go.Figure:
    """Side-by-side bar chart comparing original vs alternative scenario scores."""
    orig_factors = {f.name: f.score_contribution for f in comparison.original.factors}
    alt_factors  = {f.name: f.score_contribution for f in comparison.alternative.factors}
    names = list(orig_factors.keys())

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Original",
            x=names,
            y=[orig_factors[n] for n in names],
            marker_color=COLORS["coral"],
            hovertemplate="%{x}<br>Original: %{y:.1f} pts<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Alternative",
            x=names,
            y=[alt_factors.get(n, 0) for n in names],
            marker_color=COLORS["teal"],
            hovertemplate="%{x}<br>Alternative: %{y:.1f} pts<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        height=340,
        title=dict(text="Original vs Alternative — Factor Scores", font=dict(size=13)),
        xaxis=dict(title="", tickangle=-30, **PLOTLY_LAYOUT["xaxis"]),
        yaxis=dict(title="Points", **PLOTLY_LAYOUT["yaxis"]),
    )
    return fig


# ── Safeguards list ───────────────────────────────────────────────────────────


def safeguards_list(safeguards: list[str]) -> None:
    """Render safeguard suggestions as styled cards."""
    if not safeguards:
        st.success("No additional safeguards required.")
        return
    st.markdown('<div class="section-header">Recommended Safeguards</div>', unsafe_allow_html=True)
    for item in safeguards:
        st.markdown(f'<div class="safeguard-item">🛡 {item}</div>', unsafe_allow_html=True)


# ── Data warnings ─────────────────────────────────────────────────────────────


def data_warnings_box(warnings: list[str]) -> None:
    """Display data quality warnings."""
    if not warnings:
        return
    for w in warnings:
        st.markdown(
            f'<div class="data-warning">⚠ {w}</div>',
            unsafe_allow_html=True,
        )


# ── Synthetic data notice ─────────────────────────────────────────────────────


def synthetic_notice() -> None:
    st.markdown(
        '<div class="data-synthetic">⚡ All data shown is synthetic hackathon data — '
        "not real production observations.</div>",
        unsafe_allow_html=True,
    )
