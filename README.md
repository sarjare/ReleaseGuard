<!-- Banner -->
<div align="center">

# 🛡 ReleaseGuard
### AI Ops Release Safety Advisor

**"Is this release safe to ship right now?"**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

> ⚠ **Data notice**: All datasets bundled with this project are **synthetic hackathon data**. They are not real production records and must not be presented as actual observations.

---

## Problem Statement

Software teams deploy multiple times per day. Some releases cause incidents; others do not. Release decisions are often made on intuition rather than current service reliability and historical evidence.

**ReleaseGuard helps a developer answer:** *"Is this release safe to ship right now?"*

It evaluates a proposed release against:
- Current SLO (Service Level Objective) compliance and error-budget consumption
- Recent error-budget burn rate
- Historical release-linked incidents for the target service
- Change size (files and lines changed)
- Number of services affected (blast radius)
- Deployment timing (peak traffic windows, freeze periods)
- Recent service instability

It recommends one of three outcomes:
| Recommendation | Meaning |
|----------------|---------|
| ✅ **GO** | Risk is low — proceed with normal care |
| ⚠ **GO WITH SAFEGUARDS** | Moderate risk — proceed only with listed mitigations |
| 🛑 **WAIT** | Risk is high or freeze policy violated — resolve issues first |

> **ReleaseGuard is a human decision-support tool.** It does not automatically block, trigger, or roll back deployments.

---

## What's Innovative

1. **Transparent heuristic engine** — every score contribution is shown with its evidence. No black-box output.
2. **What-if analysis** — adjust change size, affected services, or deployment timing and instantly see how the score changes. Critically, the tool makes clear that changing timing does *not* fix underlying reliability issues.
3. **SLO-aware risk** — the score is anchored to real SLO targets and error-budget math, not just gut feel.
4. **Honest about association vs causation** — the tool never claims a release *caused* an incident; it reports *associated* incidents only.
5. **Data-quality warnings** — every calculation surface displays warnings when data is incomplete, low-traffic, or synthetic.

---

## System Architecture

```
releaseguard/
├── app.py                      # Streamlit entry point — all 6 dashboard sections
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/                       # CSV datasets (synthetic)
│   ├── slo_definitions.csv     # SLO targets per service
│   ├── service_metrics.csv     # Daily request/failure counts
│   ├── release_history.csv     # Historical deployments
│   ├── incidents.csv           # Incidents linked to releases
│   ├── traffic_calendar.csv    # Peak windows and freeze periods
│   └── README.md               # Dataset notes and caveats
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # CSV loading, validation, cross-dataset checks
│   ├── risk_engine.py          # SLO math, heuristic scoring, what-if
│   └── dashboard_components.py # Reusable Streamlit UI components + CSS
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py     # Data loading and validation tests
│   └── test_risk_engine.py     # SLO math, risk score, and what-if tests
│
└── assets/                     # Static assets (logos etc.)
```

**Data flow:**
```
CSV files → data_loader.py (validate) → risk_engine.py (calculate) → app.py (display)
```

---

## Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Frontend | [Streamlit](https://streamlit.io) | Rapid data app development, no JS required |
| Data processing | [Pandas](https://pandas.pydata.org) + [NumPy](https://numpy.org) | Dataframe operations and aggregations |
| Visualisation | [Plotly](https://plotly.com/python/) | Interactive dark-mode charts |
| ML (optional) | [scikit-learn](https://scikit-learn.org) | Available if dataset grows; not used currently — see §ML below |
| Testing | [pytest](https://pytest.org) | Deterministic unit tests |
| Version control | Git / GitHub | Collaboration and CI |

---

## Datasets

### Relationships

```
slo_definitions.service  ←── service_metrics.service
slo_definitions.service  ←── release_history.service
release_history.release_id ←── incidents.related_release
traffic_calendar.date    ←── release_history.date  (join for context)
```

### slo_definitions.csv
| Column | Type | Description |
|--------|------|-------------|
| service | string | Service identifier |
| slo_target | float | SLO target as a **percentage** (e.g. 99.9 means 99.9%, not 0.999) |
| sli_type | string | `availability` (only availability SLIs are calculated) |
| description | string | Human-readable description |

### service_metrics.csv
| Column | Type | Description |
|--------|------|-------------|
| date | date | Observation date |
| service | string | Service name |
| total_requests | int | Total requests that day |
| failed_requests | int | Failed requests that day |
| avg_latency_ms | float | Average latency (informational; not used in budget calc) |

### release_history.csv
| Column | Type | Description |
|--------|------|-------------|
| release_id | string | Unique release identifier |
| date | date | Deployment date |
| deployment_hour | int | Hour of deployment (0–23) |
| service | string | Primary service deployed |
| files_changed | int | Files modified |
| lines_changed | int | Lines changed (additions + deletions) |
| affected_services | int | Number of services touched |
| author_tenure_months | int | Months the author has been at the company (**not used to rank individuals**) |
| change_type | string | bugfix / feature / refactor / dependency_update |
| release_notes | string | Free-text description |

### incidents.csv
| Column | Type | Description |
|--------|------|-------------|
| incident_id | string | Unique incident identifier |
| date | date | Incident date |
| service | string | Affected service |
| related_release | string | Release ID manually linked to this incident |
| severity | string | Critical / Major / Minor |
| duration_min | int | Duration in minutes |
| summary | string | Free-text summary |

### traffic_calendar.csv
| Column | Type | Description |
|--------|------|-------------|
| date | date | Calendar date |
| is_peak | bool | True if high-traffic day (campaign, event) |
| freeze_period | bool | True if deployments are restricted |
| event | string | Event name |
| description | string | Additional notes |

---

## Calculation Formulas

### Error Budget (Availability SLOs only)

```
allowed_error_rate      = 1 - (slo_target / 100)
actual_error_rate       = failed_requests / total_requests
allowed_errors          = total_requests × allowed_error_rate
budget_consumed_pct     = (failed_requests / allowed_errors) × 100
remaining_budget_pct    = max(0, 100 - budget_consumed_pct)
burn_rate               = actual_error_rate / allowed_error_rate
```

**Important limitations:**
- Daily rows are aggregated over the measurement window before calculating budget. This may obscure intra-day spikes.
- Latency-based SLOs cannot be calculated without latency thresholds — the tool skips them and explains why.
- Low request volumes (< 100 requests) produce unreliable percentages — a warning is shown.

### Risk Score

The risk score (0–100) is a **heuristic, not a calibrated failure probability**. Weights are configurable constants:

| Factor | Max Weight | Description |
|--------|-----------|-------------|
| Error Budget Consumption | 25 pts | % of budget consumed |
| Error Budget Burn Rate | 20 pts | Speed of consumption |
| Historical Release Incidents | 20 pts | Past incidents associated with service releases |
| Change Size | 15 pts | Files and lines changed |
| Freeze Period | 15 pts | Hard penalty for freeze violation |
| Affected Services | 10 pts | Blast radius |
| Recent Service Instability | 10 pts | Error-rate volatility in last 7 days |
| Peak Traffic Window | 5 pts | Business hours / calendar peak |

**Decision thresholds:**
```
score ≤ 35              → GO
35 < score ≤ 65         → GO WITH SAFEGUARDS
score > 65              → WAIT
freeze_period = true    → WAIT (override, regardless of score)
```

These thresholds are visible in `src/risk_engine.py` and can be adjusted.

---

## Installation

### Prerequisites
- Python 3.11 or compatible installed version
- pip

### Setup (macOS / Linux)

```bash
# Clone the repository
git clone https://github.com/sarjare/ReleaseGuard.git
cd ReleaseGuard

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Application

```bash
# Activate virtual environment if not already active
source .venv/bin/activate

# Launch the Streamlit dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Running Tests

Execute the full automated test suite (76 tests across unit, integration, and UI):

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules:
pytest tests/test_data_loader.py -v     # Data loading & validation tests
pytest tests/test_risk_engine.py -v     # SLO math & risk engine tests
pytest tests/test_dashboard_ui.py -v    # Streamlit UI component & AppTest tests
pytest tests/test_integration.py -v     # Full end-to-end integration tests
```

Expected output: `76 passed in ~3s`.

---

## Demo Scenarios

### Scenario 1 — Healthy Release (expected: GO)

Use the **Evaluate Release** page:

| Field | Value |
|-------|-------|
| Service | `notification-api` |
| Files Changed | `3` |
| Lines Changed | `100` |
| Affected Services | `1` |
| Deploy Date | A non-peak, non-freeze date (e.g. `2026-10-27`) |
| Deploy Hour | `7` (before business hours) |

Expected: **GO** — low change size, healthy service metrics, no freeze, off-peak hour.

---

### Scenario 2 — Risky Release (expected: WAIT)

| Field | Value |
|-------|-------|
| Service | `auth-api` |
| Files Changed | `35` |
| Lines Changed | `5000` |
| Affected Services | `4` |
| Deploy Date | `2026-10-13` (calendar peak + freeze day) |
| Deploy Hour | `14` |

Expected: **WAIT** — freeze period override, large change, elevated history of incidents on `auth-api`, peak traffic day.

---

### Scenario 3 — Safer Alternative (What-If Analysis)

After running Scenario 2, go to **What-If Analysis**:

**Scenario A** (original): same inputs as Scenario 2.  
**Scenario B** (alternative): same service, but:
- Files Changed: `8`
- Lines Changed: `800`
- Affected Services: `1`
- Deploy Hour: `7`
- Deploy Date: `2026-10-27` (not a freeze day)

The score comparison will show which factors improved and explain that the budget-related risk for `auth-api` persists regardless of timing, because it depends on the service's actual error rate — not when you deploy.

---

## Machine Learning

The current dataset (24 releases, 9 incidents) is **too small** for a meaningful ML model. A scikit-learn classifier would have extremely high variance on a dataset of this size and would provide less transparency than the heuristic engine.

Conditions for enabling ML:
- ≥ 200 labelled release outcomes with balanced classes
- A clear, consistent definition of "incident" that can be linked to releases
- A held-out test set to evaluate precision, recall, and PR-AUC
- No features that use post-deployment information

When those conditions are met, a logistic regression or gradient-boosted tree can be trained on the same features and compared to the heuristic baseline.

---

## Limitations and Future Improvements

| Limitation | Future Improvement |
|------------|--------------------|
| Synthetic data only | Integrate with real DORA metrics / incident management APIs |
| Availability SLOs only | Add latency SLO support using threshold data |
| Daily aggregation hides intra-day spikes | Use hourly or per-request data |
| Heuristic weights are not validated | Train on historical data when sufficient labelled examples exist |
| No user authentication | Add role-based views for SRE / developer / manager |
| No CI/CD integration | Read deployment metadata from GitHub Actions or Spinnaker |
| Author tenure not used | Aggregate team-level experience (not individual) as a factor |
| No real-time data | Connect to monitoring APIs (Datadog, New Relic, Prometheus) |

---

## Team

MIT Hackathon — AIOps Quantum Crew

---

## License

MIT License
