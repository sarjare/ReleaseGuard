"""
data_loader.py — ReleaseGuard Data Loading and Validation

Loads and validates the five CSV datasets used by ReleaseGuard.
All datasets are synthetic hackathon data; they are NOT real production observations.

Datasets:
    slo_definitions.csv  — SLO targets and SLI types per service
    service_metrics.csv  — Daily request/failure counts per service
    release_history.csv  — Historical deployments with change metadata
    incidents.csv        — Incidents linked to releases by release_id
    traffic_calendar.csv — Peak periods and freeze windows per date

Author: ReleaseGuard Team
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Path helpers ──────────────────────────────────────────────────────────────

# Resolve the data directory relative to this file so the project works
# regardless of which directory the user launches streamlit from.
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
DATA_DIR = _PROJECT_ROOT / "data"

_FILES = {
    "slo_definitions": DATA_DIR / "slo_definitions.csv",
    "service_metrics": DATA_DIR / "service_metrics.csv",
    "release_history": DATA_DIR / "release_history.csv",
    "incidents": DATA_DIR / "incidents.csv",
    "traffic_calendar": DATA_DIR / "traffic_calendar.csv",
}

# ── Required columns ──────────────────────────────────────────────────────────

_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "slo_definitions": ["service", "slo_target", "sli_type", "description"],
    "service_metrics": ["date", "service", "total_requests", "failed_requests"],
    "release_history": [
        "release_id",
        "date",
        "deployment_hour",
        "service",
        "files_changed",
        "lines_changed",
        "affected_services",
        "change_type",
    ],
    "incidents": ["incident_id", "date", "service", "related_release", "severity"],
    "traffic_calendar": ["date", "is_peak", "freeze_period"],
}

# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class LoadedData:
    """Container returned by load_all_data().

    All DataFrames are None when a file is missing or unreadable.
    Check `warnings` for non-fatal validation messages.
    `errors` contains fatal loading failures.
    """

    slo_definitions: Optional[pd.DataFrame] = None
    service_metrics: Optional[pd.DataFrame] = None
    release_history: Optional[pd.DataFrame] = None
    incidents: Optional[pd.DataFrame] = None
    traffic_calendar: Optional[pd.DataFrame] = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """True when the minimum required datasets are available."""
        return (
            self.slo_definitions is not None
            and self.service_metrics is not None
            and self.release_history is not None
        )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _load_csv(name: str, path: Path, result: LoadedData) -> Optional[pd.DataFrame]:
    """Load a single CSV and perform basic structural validation.

    Returns the DataFrame on success, None on failure (error recorded in result).
    """
    if not path.exists():
        result.errors.append(
            f"Required file not found: {path}\n"
            f"  → Please place '{path.name}' in the data/ directory."
        )
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"Could not parse '{path.name}': {exc}")
        return None

    if df.empty:
        result.warnings.append(f"'{path.name}' loaded but contains no rows.")
        return df

    # Check required columns
    required = _REQUIRED_COLUMNS.get(name, [])
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        result.errors.append(
            f"'{path.name}' is missing required columns: {missing_cols}\n"
            f"  Found columns: {list(df.columns)}"
        )
        return None

    return df


def _validate_slo_definitions(df: pd.DataFrame, result: LoadedData) -> pd.DataFrame:
    df = df.copy()

    # slo_target must be numeric and in (0, 100)
    df["slo_target"] = pd.to_numeric(df["slo_target"], errors="coerce")
    bad_targets = df[df["slo_target"].isna() | (df["slo_target"] <= 0) | (df["slo_target"] >= 100)]
    if not bad_targets.empty:
        result.warnings.append(
            f"slo_definitions: {len(bad_targets)} rows have invalid slo_target values "
            f"(must be between 0 and 100 exclusive). Rows dropped: "
            f"{bad_targets['service'].tolist()}"
        )
        df = df.drop(bad_targets.index)

    # Strip whitespace from service names
    df["service"] = df["service"].str.strip()

    # Only availability SLIs are supported; warn about others
    unsupported = df[df["sli_type"] != "availability"]
    if not unsupported.empty:
        result.warnings.append(
            f"slo_definitions: {len(unsupported)} service(s) use SLI type(s) other than "
            f"'availability' ({unsupported['sli_type'].unique().tolist()}). "
            f"ReleaseGuard currently only supports availability-based SLO calculations. "
            f"These services will be included but budget calculations may not apply."
        )

    return df


def _validate_service_metrics(df: pd.DataFrame, result: LoadedData) -> pd.DataFrame:
    df = df.copy()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = df["date"].isna().sum()
    if bad_dates:
        result.warnings.append(
            f"service_metrics: {bad_dates} rows have unparseable dates and will be dropped."
        )
        df = df.dropna(subset=["date"])

    # Numeric columns
    for col in ["total_requests", "failed_requests"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where total_requests or failed_requests is NaN or negative
    bad_numeric = df[
        df["total_requests"].isna()
        | df["failed_requests"].isna()
        | (df["total_requests"] < 0)
        | (df["failed_requests"] < 0)
    ]
    if not bad_numeric.empty:
        result.warnings.append(
            f"service_metrics: {len(bad_numeric)} rows with invalid numeric values dropped."
        )
        df = df.drop(bad_numeric.index)

    # Warn where failed > total
    impossible = df[df["failed_requests"] > df["total_requests"]]
    if not impossible.empty:
        result.warnings.append(
            f"service_metrics: {len(impossible)} rows where failed_requests > total_requests. "
            f"These rows are kept but calculations will be inaccurate."
        )

    # avg_latency_ms is optional; coerce if present
    if "avg_latency_ms" in df.columns:
        df["avg_latency_ms"] = pd.to_numeric(df["avg_latency_ms"], errors="coerce")

    df["service"] = df["service"].str.strip()
    return df


def _validate_release_history(df: pd.DataFrame, result: LoadedData) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["deployment_hour"] = pd.to_numeric(df["deployment_hour"], errors="coerce")
    df["files_changed"] = pd.to_numeric(df["files_changed"], errors="coerce")
    df["lines_changed"] = pd.to_numeric(df["lines_changed"], errors="coerce")
    df["affected_services"] = pd.to_numeric(df["affected_services"], errors="coerce")

    bad_dates = df["date"].isna().sum()
    if bad_dates:
        result.warnings.append(
            f"release_history: {bad_dates} rows have unparseable dates and will be dropped."
        )
        df = df.dropna(subset=["date"])

    bad_hours = df[~df["deployment_hour"].between(0, 23, inclusive="both")]
    if not bad_hours.empty:
        result.warnings.append(
            f"release_history: {len(bad_hours)} rows have deployment_hour outside 0–23."
        )

    df["service"] = df["service"].str.strip()

    # author_tenure_months — aggregate statistics only; do not use per-individual
    if "author_tenure_months" in df.columns:
        df["author_tenure_months"] = pd.to_numeric(df["author_tenure_months"], errors="coerce")

    return df


def _validate_incidents(df: pd.DataFrame, result: LoadedData) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["service"] = df["service"].str.strip()

    if "duration_min" in df.columns:
        df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")

    bad_dates = df["date"].isna().sum()
    if bad_dates:
        result.warnings.append(
            f"incidents: {bad_dates} rows have unparseable dates and will be dropped."
        )
        df = df.dropna(subset=["date"])

    return df


def _validate_traffic_calendar(df: pd.DataFrame, result: LoadedData) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Parse boolean-ish columns — CSV may store "True"/"False" as strings
    for col in ["is_peak", "freeze_period"]:
        if df[col].dtype == object:
            df[col] = df[col].map({"True": True, "False": False, True: True, False: False})
        df[col] = df[col].astype(bool)

    bad_dates = df["date"].isna().sum()
    if bad_dates:
        result.warnings.append(
            f"traffic_calendar: {bad_dates} rows have unparseable dates and will be dropped."
        )
        df = df.dropna(subset=["date"])

    return df


def _cross_validate(data: LoadedData) -> None:
    """Check referential integrity between datasets."""
    if data.release_history is None or data.incidents is None:
        return

    release_ids = set(data.release_history["release_id"].dropna())
    incident_release_refs = set(data.incidents["related_release"].dropna())
    orphaned = incident_release_refs - release_ids
    if orphaned:
        data.warnings.append(
            f"incidents: {len(orphaned)} incident(s) reference release IDs not found in "
            f"release_history: {sorted(orphaned)}. These incidents are kept but cannot "
            f"be linked to release metadata."
        )

    if data.service_metrics is not None and data.slo_definitions is not None:
        metric_services = set(data.service_metrics["service"].unique())
        slo_services = set(data.slo_definitions["service"].unique())
        no_slo = metric_services - slo_services
        if no_slo:
            data.warnings.append(
                f"service_metrics: {len(no_slo)} service(s) have metrics but no SLO definition: "
                f"{sorted(no_slo)}. Budget calculations are unavailable for these services."
            )
        no_metrics = slo_services - metric_services
        if no_metrics:
            data.warnings.append(
                f"slo_definitions: {len(no_metrics)} service(s) have an SLO but no metrics data: "
                f"{sorted(no_metrics)}."
            )


# ── Public API ────────────────────────────────────────────────────────────────


def load_all_data(data_dir: Optional[Path] = None) -> LoadedData:
    """Load and validate all ReleaseGuard datasets.

    Parameters
    ----------
    data_dir:
        Override the default data directory (useful in tests).

    Returns
    -------
    LoadedData
        Container holding validated DataFrames plus lists of warnings and errors.
        Always returns a LoadedData object; never raises on missing/bad data.

    Notes
    -----
    All datasets are synthetic hackathon data. They are NOT real production records.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    files = {
        name: data_dir / path.name for name, path in _FILES.items()
    }

    result = LoadedData()

    # ── Load raw CSVs ─────────────────────────────────────────────────────────
    raw_slo = _load_csv("slo_definitions", files["slo_definitions"], result)
    raw_metrics = _load_csv("service_metrics", files["service_metrics"], result)
    raw_releases = _load_csv("release_history", files["release_history"], result)
    raw_incidents = _load_csv("incidents", files["incidents"], result)
    raw_calendar = _load_csv("traffic_calendar", files["traffic_calendar"], result)

    # ── Validate each dataset ─────────────────────────────────────────────────
    if raw_slo is not None and not raw_slo.empty:
        result.slo_definitions = _validate_slo_definitions(raw_slo, result)

    if raw_metrics is not None and not raw_metrics.empty:
        result.service_metrics = _validate_service_metrics(raw_metrics, result)

    if raw_releases is not None and not raw_releases.empty:
        result.release_history = _validate_release_history(raw_releases, result)

    if raw_incidents is not None and not raw_incidents.empty:
        result.incidents = _validate_incidents(raw_incidents, result)

    if raw_calendar is not None and not raw_calendar.empty:
        result.traffic_calendar = _validate_traffic_calendar(raw_calendar, result)

    # ── Cross-dataset validation ───────────────────────────────────────────────
    _cross_validate(result)

    return result


def get_services(data: LoadedData) -> list[str]:
    """Return sorted list of all known service names across datasets."""
    services: set[str] = set()
    if data.slo_definitions is not None:
        services.update(data.slo_definitions["service"].dropna().unique())
    if data.service_metrics is not None:
        services.update(data.service_metrics["service"].dropna().unique())
    if data.release_history is not None:
        services.update(data.release_history["service"].dropna().unique())
    return sorted(services)
