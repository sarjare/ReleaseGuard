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
import html
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.risk_engine import BudgetResult, RiskFactor, RiskResult, WhatIfResult

# ── Color Palette (Cinematic Cosmic SRE Control Room) ─────────────────────────
COLORS = {
    # Deep Cosmic Space Base
    "bg": "#030712",
    "bg_secondary": "#070c1e",
    "card": "rgba(10, 16, 32, 0.78)",
    "card_hover": "rgba(18, 28, 56, 0.90)",
    "border": "rgba(147, 197, 253, 0.12)",
    "border_light": "rgba(168, 85, 247, 0.28)",
    "border_glow": "rgba(56, 189, 248, 0.45)",

    # Luminous Filaments & Nebulae (Reference Image Palette)
    "blue": "#38bdf8",
    "blue_glow": "rgba(56, 189, 248, 0.35)",
    "violet": "#a855f7",
    "violet_glow": "rgba(168, 85, 247, 0.35)",
    "magenta": "#e879f9",
    "cyan": "#22d3ee",
    "nebula_indigo": "rgba(99, 102, 241, 0.15)",
    "nebula_violet": "rgba(168, 85, 247, 0.12)",
    "nebula_cyan": "rgba(34, 211, 238, 0.10)",

    # SRE Health State Indicators
    "teal": "#2dd4bf",
    "teal_glow": "rgba(45, 212, 191, 0.35)",
    "emerald": "#34d399",
    "amber": "#fbbf24",
    "amber_glow": "rgba(251, 191, 36, 0.35)",
    "coral": "#f43f5e",
    "coral_glow": "rgba(244, 63, 94, 0.35)",
    "rose": "#fb7185",

    # High-Contrast Accessible Typography
    "text": "#f8fafc",
    "text_muted": "#94a3b8",
    "subtext": "#64748b",

    # Semantic Decision Badges
    "go_bg": "rgba(16, 185, 129, 0.16)",
    "go_border": "#10b981",
    "go_text": "#34d399",
    "wait_bg": "rgba(244, 63, 94, 0.18)",
    "wait_border": "#f43f5e",
    "wait_text": "#fb7185",
    "safeguard_bg": "rgba(245, 158, 11, 0.18)",
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
    """Return consistent cinematic cosmic theme layout dictionary with modern glass styling."""
    layout: dict[str, Any] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": dict(color=COLORS["text"], family="Inter, -apple-system, system-ui, sans-serif"),
        "margin": dict(l=35, r=20, t=35, b=35),
        "height": height,
        "hoverlabel": dict(
            bgcolor="rgba(8, 14, 30, 0.95)",
            bordercolor="rgba(168, 85, 247, 0.45)",
            font=dict(color=COLORS["text"], size=12, family="JetBrains Mono, monospace"),
        ),
        "xaxis": {
            "title": xaxis_title,
            "gridcolor": "rgba(255, 255, 255, 0.045)",
            "showgrid": True,
            "zeroline": False,
            "tickfont": dict(color=COLORS["text_muted"], size=11, family="Inter, sans-serif"),
        },
        "yaxis": {
            "title": yaxis_title,
            "gridcolor": "rgba(255, 255, 255, 0.045)",
            "showgrid": True,
            "zeroline": False,
            "tickfont": dict(color=COLORS["text_muted"], size=11, family="Inter, sans-serif"),
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "font": dict(color=COLORS["text"], size=11, family="Inter, sans-serif"),
        },
    }
    if title:
        layout["title"] = dict(text=title, font=dict(size=13, color=COLORS["text"], family="Inter, sans-serif"))
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Base resets & typography */
html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: {COLORS['text']};
    -webkit-font-smoothing: antialiased;
}}

/* Deep Space Cosmic Canvas with Nebular Wash and Micro-Stars */
.stApp {{
    background-color: {COLORS['bg']} !important;
    background-image: 
        radial-gradient(ellipse at 15% 15%, rgba(99, 102, 241, 0.14) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 25%, rgba(168, 85, 247, 0.12) 0%, transparent 55%),
        radial-gradient(ellipse at 50% 85%, rgba(56, 189, 248, 0.09) 0%, transparent 60%),
        radial-gradient(circle at 75% 75%, rgba(232, 121, 249, 0.06) 0%, transparent 45%),
        radial-gradient(2px 2px at 25px 35px, rgba(255, 255, 255, 0.35), rgba(0,0,0,0)),
        radial-gradient(1.5px 1.5px at 110px 95px, rgba(168, 85, 247, 0.45), rgba(0,0,0,0)),
        radial-gradient(2px 2px at 190px 170px, rgba(56, 189, 248, 0.4), rgba(0,0,0,0)),
        radial-gradient(1px 1px at 280px 240px, rgba(255, 255, 255, 0.3), rgba(0,0,0,0)) !important;
    background-size: 100% 100%, 100% 100%, 100% 100%, 100% 100%, 360px 360px, 460px 460px, 300px 300px, 420px 420px !important;
    background-attachment: fixed !important;
}}

/* Main container spacing & max-width */
.block-container {{
    padding-top: 1.25rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1460px !important;
}}

/* Hide default Streamlit chrome */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{background: transparent !important;}}

/* Sleek Cosmic Sidebar Glassmorphism */
section[data-testid="stSidebar"] {{
    background: rgba(6, 10, 22, 0.88) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(168, 85, 247, 0.22) !important;
    box-shadow: 8px 0 36px rgba(0, 0, 0, 0.65) !important;
}}

/* Sidebar navigation radio buttons */
div[data-testid="stSidebar"] div[role="radiogroup"] > label {{
    padding: 10px 16px !important;
    border-radius: 10px !important;
    margin-bottom: 6px !important;
    border: 1px solid transparent !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}
div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
    background: rgba(168, 85, 247, 0.12) !important;
    border-color: rgba(168, 85, 247, 0.35) !important;
    box-shadow: 0 0 16px rgba(168, 85, 247, 0.2) !important;
}}

/* Control Room Header with Cosmic Glassmorphism */
.control-room-header {{
    background: linear-gradient(135deg, rgba(10, 16, 34, 0.90) 0%, rgba(18, 16, 44, 0.82) 100%);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(147, 197, 253, 0.14);
    border-top: 1px solid rgba(168, 85, 247, 0.5);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    position: relative;
    overflow: hidden;
}}
.control-room-header::after {{
    content: "";
    position: absolute;
    top: 0;
    right: 0;
    width: 260px;
    height: 100%;
    background: radial-gradient(circle at right, rgba(56, 189, 248, 0.12) 0%, rgba(168, 85, 247, 0.08) 50%, transparent 75%);
    pointer-events: none;
}}
.control-room-title {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {COLORS['text']};
    display: flex;
    align-items: center;
    gap: 12px;
    letter-spacing: -0.01em;
}}
.control-room-subtitle {{
    font-size: 0.84rem;
    color: {COLORS['text_muted']};
    margin-top: 5px;
    font-weight: 400;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    backdrop-filter: blur(10px);
}}
.status-online {{
    background: rgba(45, 212, 191, 0.12);
    border: 1px solid rgba(45, 212, 191, 0.45);
    color: {COLORS['teal']};
    box-shadow: 0 0 14px rgba(45, 212, 191, 0.25);
}}
.pulse-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: {COLORS['teal']};
    box-shadow: 0 0 10px {COLORS['teal']};
    animation: pulse 2.2s infinite ease-in-out;
}}
@keyframes pulse {{
    0%, 100% {{ transform: scale(1); opacity: 1; }}
    50% {{ transform: scale(1.3); opacity: 0.65; }}
}}

/* Metric Cards with Cosmic Glass Surfaces */
.metric-card {{
    background: linear-gradient(135deg, rgba(10, 16, 32, 0.82) 0%, rgba(16, 24, 48, 0.68) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(147, 197, 253, 0.10);
    border-top: 1px solid rgba(168, 85, 247, 0.38);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
    transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}}
.metric-card:hover {{
    border-color: rgba(56, 189, 248, 0.45);
    border-top-color: rgba(168, 85, 247, 0.75);
    transform: translateY(-3px);
    box-shadow: 0 14px 38px rgba(0, 0, 0, 0.55), 0 0 24px rgba(99, 102, 241, 0.22);
}}
.metric-card::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: {COLORS['blue']};
    border-radius: 4px 0 0 4px;
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
    font-size: 2.1rem;
    font-weight: 700;
    color: {COLORS['text']};
    line-height: 1.1;
    letter-spacing: -0.02em;
    text-shadow: 0 0 16px rgba(255, 255, 255, 0.15);
}}
.metric-sub {{
    font-size: 0.8rem;
    color: {COLORS['subtext']};
    margin-top: 7px;
    display: flex;
    align-items: center;
    gap: 4px;
}}

/* Section Headers */
.section-header {{
    font-size: 1.22rem;
    font-weight: 700;
    color: {COLORS['blue']};
    display: flex;
    align-items: center;
    gap: 10px;
    border-bottom: 1px solid rgba(168, 85, 247, 0.22);
    padding-bottom: 10px;
    margin-top: 24px;
    margin-bottom: 18px;
    letter-spacing: -0.01em;
    position: relative;
}}
.section-header::after {{
    content: "";
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 90px;
    height: 2px;
    background: linear-gradient(90deg, {COLORS['blue']}, {COLORS['violet']});
}}

/* Cosmic Topology Hero Section */
.topology-container {{
    background: linear-gradient(135deg, rgba(8, 12, 26, 0.88) 0%, rgba(16, 20, 44, 0.80) 100%);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(147, 197, 253, 0.14);
    border-top: 1px solid rgba(168, 85, 247, 0.5);
    border-radius: 16px;
    padding: 22px 26px 14px;
    margin-bottom: 24px;
    box-shadow: 0 12px 42px rgba(0, 0, 0, 0.5), 0 0 24px rgba(99, 102, 241, 0.14);
    position: relative;
    overflow: hidden;
}}

/* Recommendation Banners */
.reco-banner {{
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 22px;
    box-shadow: 0 10px 36px rgba(0, 0, 0, 0.45);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}}
.reco-go {{
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(5, 150, 105, 0.08) 100%);
    border: 2px solid {COLORS['go_border']};
    box-shadow: 0 8px 36px rgba(16, 185, 129, 0.22);
}}
.reco-wait {{
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.20) 0%, rgba(185, 28, 28, 0.08) 100%);
    border: 2px solid {COLORS['wait_border']};
    box-shadow: 0 8px 36px rgba(244, 63, 94, 0.24);
}}
.reco-safeguard {{
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.20) 0%, rgba(180, 83, 9, 0.08) 100%);
    border: 2px solid {COLORS['safeguard_border']};
    box-shadow: 0 8px 36px rgba(245, 158, 11, 0.22);
}}
.reco-title-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
}}
.reco-title {{
    font-size: 2.05rem;
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
    padding: 8px 20px;
    border-radius: 12px;
    background: rgba(4, 7, 18, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.16);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.12), 0 4px 16px rgba(0, 0, 0, 0.3);
}}
.reco-reason {{
    font-size: 1.0rem;
    line-height: 1.6;
    margin-top: 14px;
    color: {COLORS['text']};
    opacity: 0.96;
}}

/* Safeguard Checklist */
.safeguard-box {{
    background: rgba(10, 16, 32, 0.75);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 14px;
    padding: 20px;
    margin-top: 16px;
    box-shadow: 0 8px 26px rgba(0, 0, 0, 0.3);
}}
.safeguard-item {{
    padding: 13px 18px;
    margin-bottom: 10px;
    background: rgba(56, 189, 248, 0.07);
    border-left: 3px solid {COLORS['blue']};
    border-radius: 0 12px 12px 0;
    font-size: 0.91rem;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    line-height: 1.5;
    transition: background 0.2s ease;
}}
.safeguard-item:hover {{
    background: rgba(56, 189, 248, 0.14);
}}
.safeguard-item.freeze {{
    background: rgba(244, 63, 94, 0.12);
    border-left-color: {COLORS['coral']};
}}

/* Traffic Window Badges */
.traffic-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 14px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    backdrop-filter: blur(10px);
}}
.traffic-normal {{
    background: rgba(45, 212, 191, 0.14);
    color: {COLORS['teal']};
    border: 1px solid rgba(45, 212, 191, 0.4);
}}
.traffic-peak {{
    background: rgba(245, 158, 11, 0.16);
    color: {COLORS['amber']};
    border: 1px solid rgba(245, 158, 11, 0.4);
}}
.traffic-freeze {{
    background: rgba(244, 63, 94, 0.18);
    color: {COLORS['coral']};
    border: 1px solid rgba(244, 63, 94, 0.48);
}}

/* Data Warning & Compliance Notice */
.data-warning {{
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid {COLORS['amber']};
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.86rem;
    color: {COLORS['amber']};
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.synthetic-banner {{
    background: rgba(14, 20, 42, 0.65);
    backdrop-filter: blur(14px);
    border: 1px dashed rgba(168, 85, 247, 0.3);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 24px 0 10px;
    font-size: 0.82rem;
    color: {COLORS['text_muted']};
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
}}

/* Streamlit Native Input Widget Enhancements */
div[data-baseweb="select"] > div {{
    background: rgba(10, 16, 34, 0.85) !important;
    border: 1px solid rgba(147, 197, 253, 0.16) !important;
    border-radius: 10px !important;
    color: {COLORS['text']} !important;
}}
div[data-baseweb="input"] > div {{
    background: rgba(10, 16, 34, 0.85) !important;
    border: 1px solid rgba(147, 197, 253, 0.16) !important;
    border-radius: 10px !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #0284c7 0%, #6366f1 50%, #9333ea 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 11px 26px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 22px rgba(99, 102, 241, 0.45) !important;
    transition: all 0.25s ease !important;
}}
.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 6px 30px rgba(168, 85, 247, 0.65) !important;
    transform: translateY(-2px) !important;
}}

/* Expander glassmorphism */
div[data-testid="stExpander"] {{
    background: rgba(10, 16, 32, 0.6) !important;
    backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(147, 197, 253, 0.12) !important;
    border-radius: 12px !important;
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

/* Responsiveness media queries */
@media (max-width: 768px) {{
    .control-room-header {{
        flex-direction: column;
        align-items: flex-start;
        gap: 12px;
        padding: 16px;
    }}
    .metric-value {{
        font-size: 1.6rem;
    }}
    .reco-title {{
        font-size: 1.45rem;
    }}
    .reco-title-row {{
        flex-direction: column;
        align-items: flex-start;
    }}
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

    safe_reason = html.escape(reason)
    banner_html = (
        f'<div class="reco-banner {css_class}">'
        f'<div class="reco-title-row">'
        f'<div class="reco-title">{icon} {reco}</div>'
        f'<div class="reco-score-badge">'
        f'Risk Score: <span style="color:{score_color}; font-size:1.35rem;">{score:.0f}</span> / 100'
        f'<span style="font-size:0.75rem; color:{COLORS["text_muted"]}; margin-left:6px;">[{badge_label}]</span>'
        f'</div>'
        f'</div>'
        f'<div class="reco-reason">{safe_reason}</div>'
        f'</div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)


# ── Signature Cosmic Service Topology Map ────────────────────────────────────
def service_topology_chart(
    data: Any,
    selected_service: Optional[str] = None,
    height: int = 420,
) -> go.Figure:
    """Render signature interactive cosmic service topology map inspired by deep space filaments.

    Maps microservices onto an undulating celestial landscape connected by luminous multi-frequency filaments.
    Nodes are color-coded by real SLO error budget consumption and recent incident activity:
      - Teal: Healthy (< 50% budget consumed)
      - Amber: Warning (50% - 80% budget consumed)
      - Coral: Critical (> 80% budget consumed or recent incident)
    """
    import math
    from src.risk_engine import compute_budget_for_service

    fig = go.Figure()

    # 1. Star Dust Particle Field (Deep Space Atmosphere)
    star_x = [
        0.8, 1.5, 2.1, 2.9, 3.7, 4.3, 5.1, 5.9, 6.7, 7.3, 8.1, 8.9, 9.7, 10.4, 11.2, 11.9, 12.6,
        0.5, 1.8, 2.7, 3.4, 4.8, 5.5, 6.3, 7.8, 8.5, 9.2, 10.1, 11.5, 12.3,
        1.1, 2.3, 3.9, 4.6, 6.1, 7.0, 8.3, 9.5, 10.7, 11.7, 12.5,
        0.3, 1.4, 2.8, 4.1, 5.7, 6.9, 8.7, 9.9, 11.0, 12.2,
        0.9, 2.5, 3.8, 5.2, 6.5, 7.9, 9.1, 10.3, 11.4, 12.7,
    ]
    star_y = [
        5.8, 1.2, 6.2, 0.8, 6.5, 1.5, 6.1, 0.9, 6.4, 1.3, 6.6, 1.1, 6.2, 1.4, 6.5, 1.2, 6.1,
        3.8, 4.5, 2.1, 5.9, 2.4, 5.2, 1.9, 4.9, 2.2, 5.7, 2.1, 4.6, 2.5,
        0.5, 6.7, 0.6, 6.8, 0.7, 6.9, 0.5, 6.7, 0.6, 6.8, 0.5,
        2.2, 4.9, 3.1, 5.5, 2.8, 4.7, 3.3, 5.6, 2.7, 4.4,
        4.1, 1.8, 4.4, 1.7, 4.3, 1.6, 4.5, 1.8, 4.2, 1.9,
    ]
    star_sizes = [
        1.5, 2.5, 1.8, 3.0, 1.2, 2.2, 1.5, 2.8, 1.4, 2.0, 1.6, 2.4, 1.5, 2.2, 1.8, 2.6, 1.5,
        2.0, 1.5, 2.8, 1.6, 2.4, 1.5, 2.2, 1.7, 2.5, 1.4, 2.6, 1.8, 2.1,
        2.2, 1.8, 2.5, 1.4, 2.8, 1.6, 2.2, 1.5, 2.6, 1.7, 2.3,
        1.6, 2.4, 1.5, 2.8, 1.4, 2.2, 1.7, 2.5, 1.5, 2.3,
        2.0, 1.7, 2.4, 1.5, 2.6, 1.6, 2.3, 1.4, 2.5, 1.8,
    ]
    star_opacities = [
        "rgba(255, 255, 255, 0.45)" if i % 3 == 0 else ("rgba(168, 85, 247, 0.45)" if i % 3 == 1 else "rgba(56, 189, 248, 0.40)")
        for i in range(len(star_x))
    ]

    fig.add_trace(
        go.Scatter(
            x=star_x,
            y=star_y,
            mode="markers",
            marker=dict(size=star_sizes, color=star_opacities),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # 2. Ambient Flowing Cosmic Filaments (Undulating Waves from Reference Image)
    x_steps = [0.4 + i * 0.15 for i in range(82)]
    
    # Wave 1 (Upper Violet/Blue Crest)
    y_wave1 = [3.8 + 1.4 * math.sin(0.65 * x + 0.2) + 0.5 * math.cos(1.3 * x) for x in x_steps]
    fig.add_trace(
        go.Scatter(
            x=x_steps,
            y=y_wave1,
            mode="lines",
            line=dict(color="rgba(168, 85, 247, 0.35)", width=1.8, shape="spline"),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Wave 2 (Center Luminous Cyan Filament)
    y_wave2 = [3.4 + 1.2 * math.sin(0.70 * x + 1.2) - 0.4 * math.sin(1.5 * x) for x in x_steps]
    fig.add_trace(
        go.Scatter(
            x=x_steps,
            y=y_wave2,
            mode="lines",
            line=dict(color="rgba(56, 189, 248, 0.40)", width=2.0, shape="spline"),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Wave 3 (Lower Magenta/Purple Harmonic)
    y_wave3 = [2.9 + 1.1 * math.sin(0.58 * x + 2.4) + 0.3 * math.cos(1.1 * x) for x in x_steps]
    fig.add_trace(
        go.Scatter(
            x=x_steps,
            y=y_wave3,
            mode="lines",
            line=dict(color="rgba(232, 121, 249, 0.25)", width=1.4, shape="spline"),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Wave 4 (Atmospheric Deep Indigo Filament)
    y_wave4 = [4.3 + 1.0 * math.cos(0.62 * x + 0.8) for x in x_steps]
    fig.add_trace(
        go.Scatter(
            x=x_steps,
            y=y_wave4,
            mode="lines",
            line=dict(color="rgba(99, 102, 241, 0.25)", width=1.2, shape="spline"),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # 3. Known Microservice Nodes & Layout Coordinates
    service_coords: dict[str, tuple[float, float, str]] = {
        "client-ingress": (1.1, 3.4, "Edge API Gateway"),
        "auth-api": (3.2, 5.3, "Identity & Token Authority"),
        "user-api": (5.4, 3.1, "User Accounts & Profiles"),
        "search-api": (7.6, 5.7, "Catalog Discovery & Search"),
        "payment-api": (9.8, 3.0, "Financial Settlement Engine"),
        "notification-api": (11.7, 5.2, "Event Stream Dispatcher"),
        "database-cluster": (6.5, 1.4, "Primary Aurora Substrate"),
        "event-bus": (10.6, 1.5, "Kafka Streaming Fabric"),
    }

    # Inter-service dependency topology links
    dependency_edges = [
        ("client-ingress", "auth-api"),
        ("client-ingress", "user-api"),
        ("client-ingress", "search-api"),
        ("user-api", "auth-api"),
        ("search-api", "database-cluster"),
        ("payment-api", "auth-api"),
        ("payment-api", "database-cluster"),
        ("payment-api", "notification-api"),
        ("notification-api", "event-bus"),
        ("user-api", "notification-api"),
    ]

    # Draw Inter-Service Filaments
    for src, dst in dependency_edges:
        if src in service_coords and dst in service_coords:
            x0, y0, _ = service_coords[src]
            x1, y1, _ = service_coords[dst]

            is_active = (selected_service in (src, dst)) if selected_service else False
            if is_active:
                edge_color = "rgba(56, 189, 248, 0.95)"
                edge_width = 3.5
            else:
                edge_color = "rgba(168, 85, 247, 0.45)"
                edge_width = 1.8

            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2 + (0.35 if (x0 + x1) % 2 == 0 else -0.25)

            fig.add_trace(
                go.Scatter(
                    x=[x0, mid_x, x1],
                    y=[y0, mid_y, y1],
                    mode="lines",
                    line=dict(color=edge_color, width=edge_width, shape="spline"),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

    # 4. Compute Health & Telemetry for Services
    node_x: list[float] = []
    node_y: list[float] = []
    halo_colors: list[str] = []
    core_colors: list[str] = []
    halo_sizes: list[float] = []
    core_sizes: list[float] = []
    text_labels: list[str] = []
    hover_texts: list[str] = []

    for name, (nx, ny, role) in service_coords.items():
        node_x.append(nx)
        node_y.append(ny)
        text_labels.append(f"<b>{name}</b>")

        budget = None
        if hasattr(data, "service_metrics") and hasattr(data, "slo_definitions"):
            if data.service_metrics is not None and data.slo_definitions is not None:
                budget = compute_budget_for_service(name, data.service_metrics, data.slo_definitions)

        incident_count = 0
        if hasattr(data, "incidents") and data.incidents is not None:
            if "service" in data.incidents.columns:
                incident_count = int((data.incidents["service"] == name).sum())

        is_selected = (name == selected_service)

        if budget is not None:
            pct = budget.budget_consumed_pct
            burn = budget.burn_rate
            slo_t = budget.slo_target
            if pct >= 80.0 or burn >= 2.0 or incident_count > 0:
                status_str = "Critical Risk"
                h_col = "rgba(244, 63, 94, 0.45)"
                c_col = COLORS["coral"]
            elif pct >= 50.0 or burn >= 1.0:
                status_str = "Warning / Elevated"
                h_col = "rgba(251, 191, 36, 0.45)"
                c_col = COLORS["amber"]
            else:
                status_str = "Nominal / Healthy"
                h_col = "rgba(45, 212, 191, 0.45)"
                c_col = COLORS["teal"]
            htext = (
                f"<b>{name}</b><br>"
                f"Role: {role}<br>"
                f"Health: {status_str}<br>"
                f"SLO Target: {slo_t:.2f}%<br>"
                f"Error Budget Consumed: {pct:.1f}%<br>"
                f"Burn Rate: {burn:.2f}×<br>"
                f"Recent Incidents: {incident_count}"
            )
        else:
            status_str = "Infrastructure Component"
            h_col = "rgba(56, 189, 248, 0.35)"
            c_col = COLORS["blue"]
            htext = f"<b>{name}</b><br>Role: {role}<br>Status: Active Substrate"

        if is_selected:
            h_col = "rgba(56, 189, 248, 0.85)"
            c_col = "#38bdf8"
            halo_sizes.append(42.0)
            core_sizes.append(18.0)
        else:
            halo_sizes.append(30.0 if "api" in name else 24.0)
            core_sizes.append(14.0 if "api" in name else 11.0)

        halo_colors.append(h_col)
        core_colors.append(c_col)
        hover_texts.append(htext)

    # Outer Luminous Halos
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=halo_sizes, color=halo_colors),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Inner Radiant Cores
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=text_labels,
            textposition="top center",
            textfont=dict(
                size=11,
                color=COLORS["text"],
                family="Inter, -apple-system, sans-serif",
            ),
            marker=dict(
                size=core_sizes,
                color=core_colors,
                line=dict(color="#ffffff", width=1.5),
            ),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Center Star Pinpricks
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=4, color="#ffffff"),
            hoverinfo="none",
            showlegend=False,
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(visible=False, range=[0.2, 12.8], fixedrange=True),
        yaxis=dict(visible=False, range=[0.4, 6.8], fixedrange=True),
        hoverlabel=dict(
            bgcolor="rgba(8, 14, 30, 0.95)",
            bordercolor="rgba(168, 85, 247, 0.5)",
            font=dict(color=COLORS["text"], size=12, family="JetBrains Mono, monospace"),
        ),
    )

    return fig


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
                f'<div style="font-size:0.92rem; line-height:1.5; color:{COLORS["text"]};"><strong>Evidence:</strong> {html.escape(f.evidence)}</div>',
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
    """Daily observed error rate vs SLO threshold with smooth spline interpolation."""
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
            line=dict(color=COLORS["blue"], width=2.5, shape="spline"),
            marker=dict(size=6, color=COLORS["blue"], line=dict(width=1.5, color="#ffffff")),
            fill="tozeroy",
            fillcolor=COLORS["blue_glow"],
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Error Rate:</b> %{y:.4f}%<extra></extra>",
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
    """Plotly chart for average latency over time with smooth spline curves."""
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
            line=dict(color=COLORS["violet"], width=2.5, shape="spline"),
            marker=dict(size=6, color=COLORS["violet"], line=dict(width=1.5, color="#ffffff")),
            fill="tozeroy",
            fillcolor=COLORS["violet_glow"],
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Latency:</b> %{y:.1f} ms<extra></extra>",
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
    <div style="background:linear-gradient(135deg, rgba(19, 26, 39, 0.85) 0%, rgba(23, 33, 54, 0.7) 100%); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); border:1px solid rgba(255, 255, 255, 0.08); border-radius:14px; padding:20px 24px; margin-bottom:20px; box-shadow:0 8px 30px rgba(0,0,0,0.3);">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
        <div style="display:flex; align-items:center; gap:12px;">
          <span style="font-size:1.8rem; filter: drop-shadow(0 2px 8px rgba(0,0,0,0.3));">{icon}</span>
          <div>
            <div style="font-size:1.2rem; font-weight:700; color:{accent_color}; letter-spacing:-0.01em;">{verdict}</div>
            <div style="font-size:0.86rem; color:{COLORS['text_muted']}; margin-top:3px;">
              Scenario A ({orig_score:.1f} pts, {orig_reco}) ➔ Scenario B ({alt_score:.1f} pts, {alt_reco})
            </div>
          </div>
        </div>
        <div class="traffic-badge {badge_cls}" style="font-size:0.9rem; padding:8px 16px;">
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
        '<div class="synthetic-banner">'
        '<div>⚡ <strong>Synthetic Dataset Disclaimer:</strong> '
        'All service metrics, release history, and SLO figures are synthetic illustrative records '
        'created for the AI Ops hackathon prototype. Not connected to live production infrastructure.</div>'
        '<div style="font-family:\'JetBrains Mono\',monospace; opacity:0.8;">AI Ops ReleaseGuard v1.0</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── ML Prediction Card ────────────────────────────────────────────────────────
def ml_prediction_card(
    probability: Optional[float],
    status: str,
    warning: str,
    metrics: Optional[dict[str, float]] = None,
) -> None:
    """Display machine learning release risk prediction card with operational status."""
    if probability is None:
        st.markdown(
            f'<div style="background:{COLORS["card"]}; border:1px dashed {COLORS["border"]}; border-radius:12px; padding:18px 22px; margin-top:14px;">'
            f'<div style="display:flex; align-items:center; gap:10px;">'
            f'<span style="font-size:1.4rem;">🤖</span>'
            f'<div>'
            f'<div style="font-size:1rem; font-weight:700; color:{COLORS["text_muted"]};">ML Risk Advisor: {html.escape(status.upper())}</div>'
            f'<div style="font-size:0.82rem; color:{COLORS["subtext"]}; margin-top:3px;">{html.escape(warning)}</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    prob_pct = probability * 100.0
    if prob_pct >= 60.0:
        accent = COLORS["coral"]
        badge_cls = "traffic-freeze"
        badge_text = "HIGH ML RISK"
    elif prob_pct >= 35.0:
        accent = COLORS["amber"]
        badge_cls = "traffic-peak"
        badge_text = "MODERATE ML RISK"
    else:
        accent = COLORS["teal"]
        badge_cls = "traffic-normal"
        badge_text = "LOW ML RISK"

    metrics_html = ""
    if metrics:
        metrics_chips = " &nbsp;|&nbsp; ".join([f"<strong>{k.replace('_', ' ').title()}:</strong> <code>{v:.2f}</code>" for k, v in metrics.items()])
        metrics_html = (
            f'<div style="font-size:0.78rem; font-family:\'JetBrains Mono\',monospace; color:{COLORS["text_muted"]}; '
            f'margin-top:12px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.06);">'
            f'Validation Holdout: {metrics_chips}</div>'
        )

    safe_warning = html.escape(warning)
    safe_status = html.escape(status)
    card_html = (
        f'<div style="background:linear-gradient(135deg, rgba(19, 26, 39, 0.9) 0%, rgba(23, 33, 54, 0.75) 100%); '
        f'backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); border:1px solid rgba(88, 166, 255, 0.25); '
        f'border-radius:14px; padding:20px 24px; margin-top:16px; margin-bottom:16px; box-shadow:0 8px 30px rgba(0,0,0,0.35);">'
        f'<div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">'
        f'<div style="display:flex; align-items:center; gap:12px;">'
        f'<span style="font-size:2rem; filter: drop-shadow(0 2px 8px rgba(0,0,0,0.3));">🤖</span>'
        f'<div>'
        f'<div style="display:flex; align-items:center; gap:8px;">'
        f'<span style="font-size:1.15rem; font-weight:700; color:{COLORS["text"]};">ML Release Safety Prediction</span>'
        f'<span class="traffic-badge {badge_cls}" style="font-size:0.75rem; padding:2px 8px;">{badge_text}</span>'
        f'</div>'
        f'<div style="font-size:0.84rem; color:{COLORS["text_muted"]}; margin-top:3px;">'
        f'Status: <code style="color:{COLORS["teal"]};">{safe_status}</code> &nbsp;|&nbsp; Estimated post-release incident probability'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="font-size:1.8rem; font-weight:800; font-family:\'JetBrains Mono\',monospace; color:{accent};">'
        f'{prob_pct:.1f}%'
        f'</div>'
        f'<div style="font-size:0.75rem; color:{COLORS["text_muted"]}; text-transform:uppercase; letter-spacing:0.05em;">Failure Likelihood</div>'
        f'</div>'
        f'</div>'
        f'<div style="font-size:0.8rem; color:{COLORS["subtext"]}; margin-top:10px; line-height:1.4;">'
        f'ℹ️ {safe_warning}'
        f'</div>'
        f'{metrics_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


# ── Hybrid AI & Deterministic Explainability Panel ─────────────────────────────
def explanation_panel(explanation: Any) -> None:
    """Render comprehensive hybrid explainability panel combining deterministic evidence and Gemini AI summaries."""
    if explanation is None:
        return

    # Status badge styling
    ai_status = getattr(explanation, "ai_status", "not_configured")
    if ai_status == "generated":
        badge_cls = "traffic-normal"
        badge_text = f"🤖 {explanation.ai_provider or 'Google Gemini'}"
    elif ai_status == "not_configured":
        badge_cls = "traffic-peak"
        badge_text = "⚠️ Gemini API key not configured"
    elif ai_status == "rejected":
        badge_cls = "traffic-peak"
        badge_text = "⚠️ Gemini response rejected — retry required"
    else:  # request_failed or fallback or error
        badge_cls = "traffic-peak"
        badge_text = "⚠️ Gemini request failed — retry required"

    # Executive Operational Summary (Generated Exclusively by Google Gemini)
    if ai_status == "generated" and explanation.ai_summary:
        exec_summary_html = (
            f'<div style="font-size:0.95rem; line-height:1.6; color:{COLORS["text"]}; background:rgba(0,0,0,0.25); '
            f'border-left:3px solid {COLORS["blue"]}; padding:14px 18px; border-radius:4px 8px 8px 4px;">'
            f'{html.escape(explanation.ai_summary)}'
            f'</div>'
            f'<div style="font-size:0.75rem; color:{COLORS["subtext"]}; margin-top:6px; font-style:italic;">'
            f'ℹ️ {html.escape(explanation.ai_disclaimer or "Executive operational summary generated exclusively by Google Gemini from verified telemetry.")}'
            f'</div>'
        )
    elif ai_status == "not_configured":
        exec_summary_html = (
            f'<div style="background:rgba(251, 191, 36, 0.08); border:1px solid rgba(251, 191, 36, 0.3); '
            f'border-left:4px solid {COLORS["amber"]}; padding:14px 18px; border-radius:4px 8px 8px 4px;">'
            f'<div style="font-weight:700; color:{COLORS["amber"]}; font-size:0.92rem; margin-bottom:4px;">'
            f'⚠️ Gemini API key not configured'
            f'</div>'
            f'<div style="font-size:0.86rem; line-height:1.5; color:{COLORS["text_muted"]};">'
            f'The Executive Operational Summary is generated exclusively by Google Gemini. Please configure the <code>GEMINI_API_KEY</code> environment variable to enable AI generation. Deterministic text is not substituted.'
            f'</div>'
            f'</div>'
        )
    elif ai_status == "rejected":
        exec_summary_html = (
            f'<div style="background:rgba(251, 113, 133, 0.08); border:1px solid rgba(251, 113, 133, 0.3); '
            f'border-left:4px solid {COLORS["rose"]}; padding:14px 18px; border-radius:4px 8px 8px 4px;">'
            f'<div style="font-weight:700; color:{COLORS["rose"]}; font-size:0.92rem; margin-bottom:4px;">'
            f'⚠️ Gemini response rejected — retry required'
            f'</div>'
            f'<div style="font-size:0.86rem; line-height:1.5; color:{COLORS["text_muted"]};">'
            f'{html.escape(explanation.ai_disclaimer or "The summary generated by Gemini contradicted the verified release decision or was malformed. It was rejected by safety guardrails. Deterministic text is not substituted.")}'
            f'</div>'
            f'</div>'
        )
    else:  # request_failed
        exec_summary_html = (
            f'<div style="background:rgba(248, 113, 113, 0.08); border:1px solid rgba(248, 113, 113, 0.3); '
            f'border-left:4px solid {COLORS["coral"]}; padding:14px 18px; border-radius:4px 8px 8px 4px;">'
            f'<div style="font-weight:700; color:{COLORS["coral"]}; font-size:0.92rem; margin-bottom:4px;">'
            f'⚠️ Gemini request failed — retry required'
            f'</div>'
            f'<div style="font-size:0.86rem; line-height:1.5; color:{COLORS["text_muted"]};">'
            f'{html.escape(explanation.ai_disclaimer or "The Gemini API request failed. Please check network connectivity, API key validity, or quota and retry. Deterministic text is not substituted.")}'
            f'</div>'
            f'</div>'
        )

    # Header section
    safe_headline = html.escape(str(explanation.summary_headline))
    panel_html = (
        f'<div style="background:linear-gradient(135deg, rgba(19, 26, 39, 0.95) 0%, rgba(23, 33, 54, 0.85) 100%); '
        f'backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); border:1px solid rgba(88, 166, 255, 0.25); '
        f'border-radius:14px; padding:22px 26px; margin-top:20px; margin-bottom:20px; box-shadow:0 8px 30px rgba(0,0,0,0.35);">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:14px; padding-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.08);">'
        f'<div style="display:flex; align-items:center; gap:12px;">'
        f'<span style="font-size:1.8rem;">🔍</span>'
        f'<div>'
        f'<div style="font-size:1.2rem; font-weight:700; color:{COLORS["text"]};">Decision Explainability & Evidence</div>'
        f'<div style="font-size:0.84rem; color:{COLORS["text_muted"]};">'
        f'Deterministic risk factor attribution + Executive Operational Summary exclusively via Google Gemini'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div class="traffic-badge {badge_cls}" style="font-size:0.75rem; padding:4px 10px;">'
        f'{badge_text}'
        f'</div>'
        f'</div>'
        f'<div style="margin-bottom:18px;">'
        f'<div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em; font-weight:700; color:{COLORS["teal"]}; margin-bottom:6px;">'
        f'1. Executive Operational Summary (Generated Exclusively by Google Gemini)'
        f'</div>'
        f'{exec_summary_html}'
        f'<div style="margin-top:14px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.06);">'
        f'<div style="font-size:0.76rem; text-transform:uppercase; letter-spacing:0.05em; font-weight:600; color:{COLORS["text_muted"]}; margin-bottom:4px;">'
        f'Deterministic Risk Engine Baseline Rationale:'
        f'</div>'
        f'<div style="font-size:0.88rem; color:{COLORS["text"]}; font-style:italic;">'
        f'"{safe_headline}"'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(panel_html, unsafe_allow_html=True)

    if ai_status in ("request_failed", "rejected"):
        col_retry, col_hint = st.columns([1, 3])
        with col_retry:
            if st.button("🔄 Retry Gemini Summary", key="retry_gemini_summary", type="primary"):
                st.session_state["retry_gemini_summary_requested"] = True
                st.rerun()
        with col_hint:
            st.caption("Re-request the operational summary from Google Gemini with the configured API key.")

    # 2. Top Contributing Factors Details
    if explanation.top_factors:
        st.markdown(
            f'<div style="font-size:0.82rem; text-transform:uppercase; letter-spacing:0.06em; font-weight:700; color:{COLORS["blue"]}; margin-bottom:10px;">'
            f'2. Top Contributing Risk Factors (Ranked by Measured Impact)'
            f'</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(min(len(explanation.top_factors), 3))
        for idx, factor in enumerate(explanation.top_factors[:3]):
            col_target = cols[idx % len(cols)]
            with col_target:
                if factor.severity == "CRITICAL":
                    sev_color = COLORS["coral"]
                    sev_bg = COLORS["wait_bg"]
                elif factor.severity == "HIGH":
                    sev_color = COLORS["rose"]
                    sev_bg = "rgba(251, 113, 133, 0.15)"
                elif factor.severity == "MEDIUM":
                    sev_color = COLORS["amber"]
                    sev_bg = COLORS["safeguard_bg"]
                elif factor.severity == "LOW":
                    sev_color = COLORS["teal"]
                    sev_bg = "rgba(45, 212, 191, 0.15)"
                else:
                    sev_color = COLORS["emerald"]
                    sev_bg = COLORS["go_bg"]

                safe_name = html.escape(factor.name)
                safe_evidence = html.escape(factor.evidence)
                safe_mitigation = html.escape(factor.mitigation)
                factor_html = (
                    f'<div style="background:{COLORS["card"]}; border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:14px 16px; margin-bottom:12px; height:100%;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">'
                    f'<span style="font-size:0.85rem; font-weight:700; color:{COLORS["text"]};">#{idx + 1} {safe_name}</span>'
                    f'<span style="font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:10px; background:{sev_bg}; color:{sev_color};">'
                    f'{factor.severity}'
                    f'</span>'
                    f'</div>'
                    f'<div style="font-size:1.3rem; font-weight:800; font-family:\'JetBrains Mono\',monospace; color:{sev_color}; margin-bottom:4px;">'
                    f'{factor.contribution:.1f} <span style="font-size:0.78rem; font-weight:400; color:{COLORS["text_muted"]};">/ {factor.max_weight:.1f} pts ({factor.contribution_pct:.0f}%)</span>'
                    f'</div>'
                    f'<div style="font-size:0.8rem; color:{COLORS["text_muted"]}; margin-bottom:8px;">'
                    f'<strong>Evidence:</strong> {safe_evidence}'
                    f'</div>'
                    f'<div style="font-size:0.76rem; color:{COLORS["teal"]}; background:rgba(45,212,191,0.08); padding:6px 10px; border-radius:6px; border:1px solid rgba(45,212,191,0.2);">'
                    f'🛠️ <strong>Mitigation:</strong> {safe_mitigation}'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(factor_html, unsafe_allow_html=True)

    # 3. Factor Evidence & Full Mitigation Table Expander
    with st.expander("📋 View All Factors & Remediation Guide", expanded=False):
        for f in explanation.all_factors_ranked:
            st.markdown(
                f"• **{f.name}** ({f.contribution:.1f}/{f.max_weight:.1f} pts — `{f.severity}`): "
                f"{f.plain_english}  \n"
                f"&nbsp;&nbsp;↳ *Mitigation:* `{f.mitigation}`"
            )

    # 4. Separate ML Demonstration Context Disclosure
    if explanation.ml_context and explanation.ml_context.get("probability") is not None:
        ml_p = explanation.ml_context["probability"] * 100.0
        ml_stat = html.escape(explanation.ml_context.get("status", "simulated demo only"))
        ml_disclosure_html = (
            f'<div style="background:rgba(88, 166, 255, 0.05); border:1px solid rgba(88, 166, 255, 0.2); border-radius:10px; padding:12px 16px; margin-top:12px; margin-bottom:16px;">'
            f'<div style="font-size:0.8rem; font-weight:700; color:{COLORS["blue"]}; margin-bottom:3px;">'
            f'🤖 ML Predictor Layer (Separate Pipeline Disclosure)'
            f'</div>'
            f'<div style="font-size:0.82rem; color:{COLORS["text_muted"]};">'
            f'ML Failure Likelihood: <strong style="color:{COLORS["text"]};">{ml_p:.1f}%</strong> ({ml_stat}). '
            f'<em>The ML predictor operates independently of the deterministic release risk score ({explanation.risk_score:.0f}/100) and recommendation ({explanation.recommendation}).</em>'
            f'</div>'
            f'</div>'
        )
        st.markdown(ml_disclosure_html, unsafe_allow_html=True)
