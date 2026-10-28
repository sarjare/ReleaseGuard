"""
dashboard_components.py — Reusable Streamlit UI Components for ReleaseGuard
=============================================================================

Professional SRE / DevOps Control Room Design System:
- Dark navy control-room background (#0b0f19)
- Blue (#58a6ff), violet (#a78bfa), and teal (#2dd4bf) accents
- High-contrast accessible alerts and status indicators
- Clean typography with monospace numerical readouts
- Responsive Plotly charts styled for dark operations centers

All chart and table data is populated strictly from loaded datasets and the
risk engine; no fake live telemetry or unlabelled data is introduced.

Author: ReleaseGuard Team
"""

from __future__ import annotations

import datetime
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.risk_engine import BudgetResult, RiskFactor, RiskResult, WhatIfResult

# ── Color Palette ─────────────────────────────────────────────────────────────
COLORS = {
    "bg": "#0b0f19",
    "card": "#131a27",
    "card_hover": "#1a2335",
    "border": "#222f46",
    "border_light": "#334155",
    "blue": "#58a6ff",
    "blue_glow": "rgba(88, 166, 255, 0.15)",
    "violet": "#a78bfa",
    "violet_glow": "rgba(167, 139, 250, 0.15)",
    "teal": "#2dd4bf",
    "teal_glow": "rgba(45, 212, 191, 0.15)",
    "emerald": "#34d399",
    "amber": "#fbbf24",
    "coral": "#f87171",
    "rose": "#fb7185",
    "text": "#e6edf3",
    "text_muted": "#94a3b8",
    "subtext": "#64748b",
    # Semantic decision badges
    "go_bg": "rgba(16, 185, 129, 0.15)",
    "go_border": "#10b981",
    "go_text": "#34d399",
    "wait_bg": "rgba(239, 68, 68, 0.15)",
    "wait_border": "#ef4444",
    "wait_text": "#f87171",
    "safeguard_bg": "rgba(245, 158, 11, 0.15)",
    "safeguard_border": "#f59e0b",
    "safeguard_text": "#fbbf24",
}

# ── Standard Plotly Layout Helper ─────────────────────────────────────────────
def get_plotly_layout(
    height: int = 280,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    **overrides: Any,
) -> dict[str, Any]:
    """Return consistent dark theme layout dictionary without duplicate keywords."""
    layout: dict[str, Any] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": dict(color=COLORS["text"], family="Inter, system-ui, sans-serif"),
        "margin": dict(l=35, r=20, t=35, b=35),
        "height": height,
        "xaxis": {
            "title": xaxis_title,
            "gridcolor": "rgba(255, 255, 255, 0.06)",
            "showgrid": True,
            "zeroline": False,
            "tickfont": dict(color=COLORS["text_muted"], size=11),
        },
        "yaxis": {
            "title": yaxis_title,
            "gridcolor": "rgba(255, 255, 255, 0.06)",
            "showgrid": True,
            "zeroline": False,
            "tickfont": dict(color=COLORS["text_muted"], size=11),
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "font": dict(color=COLORS["text"], size=11),
        },
    }
    if title:
        layout["title"] = dict(text=title, font=dict(size=12, color=COLORS["text"]))
    for k, v in overrides.items():
        if k in layout and isinstance(layout[k], dict) and isinstance(v, dict):
            layout[k].update(v)
        else:
            layout[k] = v
    return layout

PLOTLY_LAYOUT = get_plotly_layout()


# ── CSS Injection ─────────────────────────────────────────────────────────────
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Base resets & typography */
html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    color: {COLORS['text']};
}}

/* Main container spacing */
.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px;
}}

/* Hide unnecessary default Streamlit chrome */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{background: transparent !important;}}

/* Control Room Header */
.control-room-header {{
    background: linear-gradient(135deg, #131a27 0%, #172136 100%);
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
}}
.control-room-title {{
    font-size: 1.4rem;
    font-weight: 700;
    color: {COLORS['text']};
    display: flex;
    align-items: center;
    gap: 12px;
}}
.control-room-subtitle {{
    font-size: 0.82rem;
    color: {COLORS['text_muted']};
    margin-top: 4px;
}}
.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}}
.status-online {{
    background: rgba(45, 212, 191, 0.12);
    border: 1px solid {COLORS['teal']};
    color: {COLORS['teal']};
}}
.pulse-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: {COLORS['teal']};
    box-shadow: 0 0 8px {COLORS['teal']};
}}

/* Metric Cards */
.metric-card {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
}}
.metric-card:hover {{
    border-color: {COLORS['blue']};
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(88, 166, 255, 0.12);
}}
.metric-card::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: {COLORS['blue']};
}}
.metric-card.teal::before {{ background: {COLORS['teal']}; }}
.metric-card.violet::before {{ background: {COLORS['violet']}; }}
.metric-card.amber::before {{ background: {COLORS['amber']}; }}
.metric-card.coral::before {{ background: {COLORS['coral']}; }}

.metric-label {{
    font-size: 0.76rem;
    font-weight: 600;
    color: {COLORS['text_muted']};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}}
.metric-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: {COLORS['text']};
    line-height: 1.1;
    letter-spacing: -0.02em;
}}
.metric-sub {{
    font-size: 0.8rem;
    color: {COLORS['subtext']};
    margin-top: 6px;
    display: flex;
    align-items: center;
    gap: 4px;
}}

/* Section Headers */
.section-header {{
    font-size: 1.15rem;
    font-weight: 600;
    color: {COLORS['blue']};
    display: flex;
    align-items: center;
    gap: 8px;
    border-bottom: 1px solid {COLORS['border']};
    padding-bottom: 10px;
    margin-top: 18px;
    margin-bottom: 16px;
}}

/* Recommendation Banners */
.reco-banner {{
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}}
.reco-go {{
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
    border: 2px solid {COLORS['go_border']};
}}
.reco-wait {{
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(185, 28, 28, 0.05) 100%);
    border: 2px solid {COLORS['wait_border']};
}}
.reco-safeguard {{
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.18) 0%, rgba(180, 83, 9, 0.05) 100%);
    border: 2px solid {COLORS['safeguard_border']};
}}
.reco-title-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}}
.reco-title {{
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    font-family: 'JetBrains Mono', monospace;
}}
.reco-go .reco-title {{ color: {COLORS['go_text']}; }}
.reco-wait .reco-title {{ color: {COLORS['wait_text']}; }}
.reco-safeguard .reco-title {{ color: {COLORS['safeguard_text']}; }}

.reco-score-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    padding: 6px 16px;
    border-radius: 8px;
    background: rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.1);
}}
.reco-reason {{
    font-size: 0.95rem;
    line-height: 1.5;
    margin-top: 10px;
    color: {COLORS['text']};
    opacity: 0.95;
}}

/* Safeguard Checklist */
.safeguard-box {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 16px;
    margin-top: 12px;
}}
.safeguard-item {{
    padding: 10px 14px;
    margin-bottom: 8px;
    background: rgba(88, 166, 255, 0.06);
    border-left: 3px solid {COLORS['blue']};
    border-radius: 0 8px 8px 0;
    font-size: 0.88rem;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    line-height: 1.45;
}}
.safeguard-item.freeze {{
    background: rgba(248, 113, 113, 0.08);
    border-left-color: {COLORS['coral']};
}}

/* Traffic Window Badge */
.traffic-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}}
.traffic-normal {{
    background: rgba(45, 212, 191, 0.12);
    color: {COLORS['teal']};
    border: 1px solid rgba(45, 212, 191, 0.3);
}}
.traffic-peak {{
    background: rgba(245, 158, 11, 0.15);
    color: {COLORS['amber']};
    border: 1px solid rgba(245, 158, 11, 0.3);
}}
.traffic-freeze {{
    background: rgba(239, 68, 68, 0.18);
    color: {COLORS['coral']};
    border: 1px solid rgba(239, 68, 68, 0.4);
}}

/* Data Warning & Compliance Notice */
.data-warning {{
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid {COLORS['amber']};
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.84rem;
    color: {COLORS['amber']};
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.synthetic-banner {{
    background: rgba(30, 41, 59, 0.6);
    border: 1px dashed {COLORS['border_light']};
    border-radius: 8px;
    padding: 10px 14px;
    margin: 18px 0;
    font-size: 0.78rem;
    color: {COLORS['text_muted']};
    display: flex;
    align-items: center;
    justify-content: space-between;
}}

/* Custom scrollbars */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{
    background: {COLORS['bg']};
}}
::-webkit-scrollbar-thumb {{
    background: {COLORS['border']};
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {COLORS['blue']};
}}
</style>
"""


def inject_css() -> None:
    """Inject control-room CSS into Streamlit."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ── Header Component ──────────────────────────────────────────────────────────
def control_room_header(active_page_title: str) -> None:
    """Render a unified top-level SRE control room header."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    html = f"""
    <div class="control-room-header">
      <div>
        <div class="control-room-title">
          <span>🛡️</span>
          <span>ReleaseGuard <span style="font-size:1.05rem; font-weight:400; color:{COLORS['blue']};">| {active_page_title}</span></span>
        </div>
        <div class="control-room-subtitle">
          AI Ops Release Safety & Error Budget Governance Control Center
        </div>
      </div>
      <div style="display:flex; align-items:center; gap:16px;">
        <div class="status-pill status-online">
          <span class="pulse-dot"></span>
          <span>SRE ENGINE ONLINE</span>
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:{COLORS['subtext']};">
          {now_str}
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ── Metric Card ───────────────────────────────────────────────────────────────
def metric_card(
    label: str,
    value: Any,
    sub: str = "",
    accent: str = "blue",
    col: Optional[Any] = None,
) -> None:
    """Render a styled SRE metric card.

    Parameters
    ----------
    label : str
        Metric title.
    value : Any
        Primary display value.
    sub : str
        Secondary explanation or delta.
    accent : str
        Color variant: 'blue', 'teal', 'violet', 'amber', 'coral'.
    col : Streamlit column
        Target column (defaults to st).
    """
    html = f"""
    <div class="metric-card {accent}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {'<div class="metric-sub">' + sub + '</div>' if sub else ''}
    </div>
    """
    target = col if col is not None else st
    target.markdown(html, unsafe_allow_html=True)


# ── Recommendation Banner ─────────────────────────────────────────────────────
def recommendation_banner(result: RiskResult) -> None:
    """Render the high-impact decision banner for GO / WAIT / GO WITH SAFEGUARDS."""
    reco = result.recommendation
    score = result.risk_score
    reason = result.recommendation_reason

    if reco == "GO":
        css_class = "reco-go"
        icon = "✅"
        badge_label = "SAFE TO DEPLOY"
        score_color = COLORS["go_text"]
    elif reco == "WAIT":
        css_class = "reco-wait"
        icon = "🛑"
        badge_label = "DEPLOYMENT BLOCKED"
        score_color = COLORS["wait_text"]
    else:
        css_class = "reco-safeguard"
        icon = "⚠️"
        badge_label = "SAFEGUARDS REQUIRED"
        score_color = COLORS["safeguard_text"]

    html = f"""
    <div class="reco-banner {css_class}">
      <div class="reco-title-row">
        <div class="reco-title">{icon} {reco}</div>
        <div class="reco-score-badge">
          Risk Score: <span style="color:{score_color}; font-size:1.35rem;">{score:.0f}</span> / 100
          <span style="font-size:0.75rem; color:{COLORS['text_muted']}; margin-left:6px;">[{badge_label}]</span>
        </div>
      </div>
      <div class="reco-reason">{reason}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ── Risk Gauge Chart ──────────────────────────────────────────────────────────
def risk_gauge(score: float, title: str = "Composite Release Risk") -> go.Figure:
    """Plotly gauge for release risk score (0-100)."""
    if score <= 35:
        bar_color = COLORS["teal"]
    elif score <= 65:
        bar_color = COLORS["amber"]
    else:
        bar_color = COLORS["coral"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"<b>{title}</b>", "font": {"color": COLORS["text"], "size": 13}},
            number={
                "font": {"color": bar_color, "size": 38, "family": "JetBrains Mono"},
                "suffix": " / 100",
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": COLORS["border_light"],
                    "tickwidth": 1,
                    "tickfont": {"color": COLORS["text_muted"], "size": 10},
                    "tickvals": [0, 35, 65, 100],
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": COLORS["card"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(45, 212, 191, 0.12)"},
                    {"range": [35, 65], "color": "rgba(251, 191, 36, 0.12)"},
                    {"range": [65, 100], "color": "rgba(248, 113, 113, 0.12)"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 3},
                    "thickness": 0.85,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        height=240,
        margin=dict(l=25, r=25, t=35, b=20),
    )
    return fig


# ── Budget Donut Chart ────────────────────────────────────────────────────────
def budget_gauge(budget: BudgetResult) -> go.Figure:
    """Plotly donut chart showing consumed vs remaining error budget."""
    consumed = min(budget.budget_consumed_pct, 100.0)
    remaining = max(0.0, 100.0 - consumed)

    if consumed >= 80:
        c_color = COLORS["coral"]
    elif consumed >= 50:
        c_color = COLORS["amber"]
    else:
        c_color = COLORS["teal"]

    fig = go.Figure(
        go.Pie(
            values=[consumed, remaining],
            labels=["Consumed", "Remaining"],
            hole=0.68,
            marker_colors=[c_color, "rgba(255, 255, 255, 0.08)"],
            textinfo="none",
            sort=False,
            hovertemplate="<b>%{label}</b>: %{value:.1f}%<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"{consumed:.1f}%",
        x=0.5,
        y=0.54,
        font=dict(size=24, color=c_color, family="JetBrains Mono", weight="bold"),
        showarrow=False,
    )
    fig.add_annotation(
        text="Consumed",
        x=0.5,
        y=0.40,
        font=dict(size=11, color=COLORS["text_muted"], family="Inter"),
        showarrow=False,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            font=dict(color=COLORS["text"], size=10),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            y=-0.15,
            x=0.2,
        ),
        height=240,
        margin=dict(l=10, r=10, t=10, b=30),
    )
    return fig


# ── Factor Breakdown Chart ────────────────────────────────────────────────────
def factor_breakdown_chart(factors: list[RiskFactor]) -> go.Figure:
    """Horizontal bar chart showing percentage contribution per risk factor."""
    names = [f.name for f in factors]
    contribs = [f.score_contribution for f in factors]
    maxes = [f.weight_max for f in factors]
    pcts = [c / m * 100 if m > 0 else 0 for c, m in zip(contribs, maxes)]

    colors = []
    for p in pcts:
        if p >= 70:
            colors.append(COLORS["coral"])
        elif p >= 40:
            colors.append(COLORS["amber"])
        else:
            colors.append(COLORS["teal"])

    fig = go.Figure(
        go.Bar(
            x=pcts,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{c:.1f}/{m:.0f} pts ({p:.0f}%)" for c, m, p in zip(contribs, maxes, pcts)],
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=10, family="JetBrains Mono"),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}% of max weight<extra></extra>",
        )
    )
    layout = get_plotly_layout(
        height=320,
        title="Risk Factor Breakdown (% of Max Weight)",
        xaxis_title="% of Factor Weight",
        showlegend=False,
    )
    layout["xaxis"]["range"] = [0, 125]
    layout["yaxis"]["autorange"] = "reversed"
    fig.update_layout(**layout)
    return fig


# ── Factor Evidence Cards ─────────────────────────────────────────────────────
def factor_evidence_table(factors: list[RiskFactor]) -> None:
    """Render factor evidence cards with severity indicators."""
    st.markdown('<div class="section-header">🔍 Contributing Risk Factors & Evidence</div>', unsafe_allow_html=True)
    sorted_factors = sorted(factors, key=lambda x: x.score_contribution, reverse=True)

    for f in sorted_factors:
        pct = (f.score_contribution / f.weight_max * 100) if f.weight_max > 0 else 0
        if pct >= 70:
            badge = "🔴 HIGH RISK"
            border_color = COLORS["coral"]
        elif pct >= 40:
            badge = "🟡 ELEVATED"
            border_color = COLORS["amber"]
        else:
            badge = "🟢 LOW RISK"
            border_color = COLORS["teal"]

        with st.expander(
            f"{badge} — {f.name} ({f.score_contribution:.1f} / {f.weight_max:.1f} pts)"
        ):
            st.markdown(
                f"""
                <div style="font-size:0.92rem; line-height:1.5; color:{COLORS['text']};">
                  <strong>Evidence:</strong> {f.evidence}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Safeguards List ───────────────────────────────────────────────────────────
def safeguards_list(safeguards: list[str]) -> None:
    """Render an operational safeguard checklist."""
    st.markdown('<div class="section-header">🛡️ Prescribed Deployment Safeguards</div>', unsafe_allow_html=True)
    if not safeguards:
        st.success("✅ No extra operational safeguards required for this release profile.")
        return

    html_items = []
    for item in safeguards:
        is_freeze = "freeze" in item.lower() or "violate" in item.lower()
        cls_name = "safeguard-item freeze" if is_freeze else "safeguard-item"
        icon = "🚨" if is_freeze else "🛡️"
        html_items.append(f'<div class="{cls_name}"><span>{icon}</span><span>{item}</span></div>')

    st.markdown(
        f'<div class="safeguard-box">{"".join(html_items)}</div>',
        unsafe_allow_html=True,
    )


# ── Error Rate Trend Chart ────────────────────────────────────────────────────
def error_rate_trend(
    service: str,
    metrics: pd.DataFrame,
    slo_target: Optional[float] = None,
) -> go.Figure:
    """Daily observed error rate vs SLO threshold."""
    svc = metrics[metrics["service"] == service].copy().sort_values("date")
    svc = svc[svc["total_requests"] > 0].copy()
    svc["error_rate_pct"] = (svc["failed_requests"] / svc["total_requests"]) * 100.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=svc["date"],
            y=svc["error_rate_pct"],
            mode="lines+markers",
            name="Observed Error Rate",
            line=dict(color=COLORS["blue"], width=2.5),
            marker=dict(size=5, color=COLORS["blue"]),
            fill="tozeroy",
            fillcolor=COLORS["blue_glow"],
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Error Rate: %{y:.4f}%<extra></extra>",
        )
    )

    if slo_target is not None:
        allowed_error_pct = (1.0 - slo_target / 100.0) * 100.0
        fig.add_hline(
            y=allowed_error_pct,
            line_dash="dash",
            line_color=COLORS["coral"],
            annotation_text=f"SLO Target Limit ({allowed_error_pct:.4f}%)",
            annotation_font_color=COLORS["coral"],
            annotation_position="top right",
        )

    layout = get_plotly_layout(
        height=280,
        title=f"Daily Error Rate vs SLO — {service}",
        yaxis_title="Error Rate (%)",
    )
    fig.update_layout(**layout)
    return fig


# ── Latency Trend Chart ───────────────────────────────────────────────────────
def latency_trend_chart(service: str, metrics: pd.DataFrame) -> Optional[go.Figure]:
    """Plotly chart for average latency over time."""
    if "avg_latency_ms" not in metrics.columns:
        return None

    svc = metrics[metrics["service"] == service].copy().sort_values("date")
    svc = svc.dropna(subset=["avg_latency_ms"])
    if svc.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=svc["date"],
            y=svc["avg_latency_ms"],
            mode="lines+markers",
            name="Avg Latency (ms)",
            line=dict(color=COLORS["violet"], width=2),
            marker=dict(size=5, color=COLORS["violet"]),
            fill="tozeroy",
            fillcolor=COLORS["violet_glow"],
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Latency: %{y:.1f} ms<extra></extra>",
        )
    )
    layout = get_plotly_layout(
        height=280,
        title=f"Average Response Latency (ms) — {service}",
        yaxis_title="Latency (ms)",
    )
    fig.update_layout(**layout)
    return fig


# ── Request Volume Chart ──────────────────────────────────────────────────────
def request_volume_chart(service: str, metrics: pd.DataFrame) -> go.Figure:
    """Stacked bar chart for daily traffic volume (Success vs Failed)."""
    svc = metrics[metrics["service"] == service].copy().sort_values("date")
    successful = svc["total_requests"] - svc["failed_requests"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=svc["date"],
            y=successful,
            name="Successful",
            marker_color=COLORS["teal"],
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Successful: %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=svc["date"],
            y=svc["failed_requests"],
            name="Failed",
            marker_color=COLORS["coral"],
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Failed: %{y:,.0f}<extra></extra>",
        )
    )
    layout = get_plotly_layout(
        height=280,
        title=f"Daily Request Volume — {service}",
        yaxis_title="Requests",
        barmode="stack",
    )
    fig.update_layout(**layout)
    return fig


# ── Historical Table Merger ───────────────────────────────────────────────────
def releases_with_incidents_table(
    releases: pd.DataFrame,
    incidents: pd.DataFrame,
    service: Optional[str] = None,
) -> pd.DataFrame:
    """Merge release records with incidents dataset for historical analysis."""
    df = releases.copy()
    if service and service != "All":
        df = df[df["service"] == service]

    if not incidents.empty and "related_release" in incidents.columns:
        inc_counts = (
            incidents.groupby("related_release")
            .agg(
                incident_count=("incident_id", "count"),
                severities=("severity", lambda x: list(set(x))),
                summaries=("summary", lambda x: " | ".join(list(set(x.dropna())))),
            )
            .reset_index()
            .rename(columns={"related_release": "release_id"})
        )
        df = df.merge(inc_counts, on="release_id", how="left")
        df["incident_count"] = df["incident_count"].fillna(0).astype(int)
        df["has_incident"] = df["incident_count"] > 0
        df["severity"] = df["severities"].apply(
            lambda s: ", ".join(sorted(s)) if isinstance(s, list) and s else "None"
        )
        df["incident_summary"] = df["summaries"].fillna("—")
        df = df.drop(columns=["severities", "summaries"], errors="ignore")
    else:
        df["incident_count"] = 0
        df["has_incident"] = False
        df["severity"] = "None"
        df["incident_summary"] = "—"

    return df.sort_values("date", ascending=False)


# ── What-If Comparison Components ─────────────────────────────────────────────
def whatif_comparison_chart(comparison: WhatIfResult) -> go.Figure:
    """Grouped bar chart comparing scenario factor points."""
    orig_factors = {f.name: f.score_contribution for f in comparison.original.factors}
    alt_factors = {f.name: f.score_contribution for f in comparison.alternative.factors}
    names = list(orig_factors.keys())

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Original (A)",
            x=names,
            y=[orig_factors[n] for n in names],
            marker_color=COLORS["coral"],
            text=[f"{orig_factors[n]:.1f}" for n in names],
            textposition="outside",
            textfont=dict(size=10, family="JetBrains Mono"),
            hovertemplate="<b>%{x}</b><br>Original: %{y:.1f} pts<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Alternative (B)",
            x=names,
            y=[alt_factors.get(n, 0) for n in names],
            marker_color=COLORS["teal"],
            text=[f"{alt_factors.get(n, 0):.1f}" for n in names],
            textposition="outside",
            textfont=dict(size=10, family="JetBrains Mono"),
            hovertemplate="<b>%{x}</b><br>Alternative: %{y:.1f} pts<extra></extra>",
        )
    )
    layout = get_plotly_layout(
        height=340,
        title="Risk Factor Comparison (Scenario A vs B)",
        yaxis_title="Factor Points",
        barmode="group",
    )
    layout["xaxis"]["tickangle"] = -25
    fig.update_layout(**layout)
    return fig


def whatif_delta_card(comparison: WhatIfResult) -> None:
    """Render a summary card showing the impact of the what-if scenario."""
    delta = comparison.score_delta
    orig_score = comparison.original.risk_score
    alt_score = comparison.alternative.risk_score
    orig_reco = comparison.original.recommendation
    alt_reco = comparison.alternative.recommendation

    if delta < -0.5:
        verdict = f"Risk Score Improved by {abs(delta):.1f} Points"
        accent_color = COLORS["teal"]
        badge_cls = "traffic-normal"
        icon = "📉"
    elif delta > 0.5:
        verdict = f"Risk Score Increased by {delta:.1f} Points"
        accent_color = COLORS["coral"]
        badge_cls = "traffic-freeze"
        icon = "📈"
    else:
        verdict = "Risk Score Remained Virtually Unchanged"
        accent_color = COLORS["amber"]
        badge_cls = "traffic-peak"
        icon = "⚖️"

    html = f"""
    <div style="background:{COLORS['card']}; border:1px solid {COLORS['border']}; border-radius:12px; padding:18px 22px; margin-bottom:18px;">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
        <div style="display:flex; align-items:center; gap:10px;">
          <span style="font-size:1.6rem;">{icon}</span>
          <div>
            <div style="font-size:1.15rem; font-weight:700; color:{accent_color};">{verdict}</div>
            <div style="font-size:0.84rem; color:{COLORS['text_muted']};">
              Scenario A ({orig_score:.1f} pts, {orig_reco}) ➔ Scenario B ({alt_score:.1f} pts, {alt_reco})
            </div>
          </div>
        </div>
        <div class="traffic-badge {badge_cls}">
          Δ {delta:+.1f} PTS
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ── Traffic Window Badge ──────────────────────────────────────────────────────
def traffic_window_badge(
    deploy_date: datetime.date,
    deploy_hour: int,
    traffic_calendar: Optional[pd.DataFrame],
) -> None:
    """Render a dynamic status badge explaining the operational traffic window."""
    is_biz = 9 <= deploy_hour <= 18
    is_peak = False
    is_freeze = False
    event_name = "Normal"

    if traffic_calendar is not None:
        match = traffic_calendar[traffic_calendar["date"].dt.date == deploy_date]
        if not match.empty:
            is_peak = bool(match.iloc[0]["is_peak"])
            is_freeze = bool(match.iloc[0]["freeze_period"])
            event_name = str(match.iloc[0].get("event", "Normal"))

    if is_freeze:
        badge_html = f"""
        <div class="traffic-badge traffic-freeze">
          🚫 FREEZE PERIOD ACTIVE — Event: '{event_name}' (Hour {deploy_hour:02d}:00)
        </div>
        """
    elif is_peak:
        badge_html = f"""
        <div class="traffic-badge traffic-peak">
          📈 CALENDAR PEAK WINDOW — Event: '{event_name}' (Hour {deploy_hour:02d}:00)
        </div>
        """
    elif is_biz:
        badge_html = f"""
        <div class="traffic-badge traffic-peak">
          ⏰ BUSINESS PEAK HOURS (09:00 - 18:00) — Deploying at {deploy_hour:02d}:00
        </div>
        """
    else:
        badge_html = f"""
        <div class="traffic-badge traffic-normal">
          🌙 OFF-PEAK WINDOW (Hour {deploy_hour:02d}:00) — Low Traffic
        </div>
        """

    st.markdown(badge_html, unsafe_allow_html=True)


# ── Data Warning & Compliance Notice ──────────────────────────────────────────
def data_warnings_box(warnings: list[str]) -> None:
    """Display SRE data quality warnings."""
    if not warnings:
        return
    for w in warnings:
        st.markdown(
            f'<div class="data-warning"><span>⚠️</span><span>{w}</span></div>',
            unsafe_allow_html=True,
        )


def synthetic_notice() -> None:
    """Mandatory compliance notice for synthetic dataset."""
    st.markdown(
        """
        <div class="synthetic-banner">
          <div>
            ⚡ <strong>Synthetic Dataset Disclaimer:</strong>
            All service metrics, release history, and SLO figures are synthetic illustrative records
            created for the AI Ops hackathon prototype. Not connected to live production infrastructure.
          </div>
          <div style="font-family:'JetBrains Mono',monospace; opacity:0.8;">
            AI Ops ReleaseGuard v1.0
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
