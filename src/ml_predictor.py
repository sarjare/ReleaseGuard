"""Leakage-aware release outcome labels and ML demonstration pipeline.

The checked-in release and incident data is synthetic and too small to train a
validated predictor. The demo model is trained only on reproducible synthetic
examples and must not be interpreted as production performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.risk_engine import THRESHOLD_GO, score_release

OUTCOME_WINDOW_DAYS = 7
MIN_HISTORICAL_RELEASES = 200
DEMO_SAMPLE_COUNT = 600
DEMO_RANDOM_STATE = 20261028

FEATURE_COLUMNS = (
    "files_changed",
    "lines_changed",
    "affected_services",
    "deploy_hour",
    "budget_consumed_pct",
    "burn_rate",
    "recent_error_rate_pct",
    "historical_incidents_30d",
)


class FeatureDataUnavailable(ValueError):
    """Raised when a proposed release lacks required pre-release feature data."""


class InsufficientTrainingData(ValueError):
    """Raised when labelled examples cannot support chronological evaluation."""


@dataclass
class TrainedReleaseModel:
    """A fitted classifier and its explicitly scoped holdout evaluation."""

    estimator: Any
    mode: str
    validation_metrics: dict[str, float]
    baseline_metrics: Optional[dict[str, float]] = None


def build_release_outcomes(
    release_history: pd.DataFrame,
    incidents: pd.DataFrame,
    observation_end: date | datetime | pd.Timestamp,
    observation_days: int = OUTCOME_WINDOW_DAYS,
) -> pd.DataFrame:
    """Label mature releases using directly linked incidents in the next N days.

    A positive is at least one incident whose ``related_release`` exactly
    matches the release ID, whose service matches, and whose date is strictly
    after the release date and no later than release date + ``observation_days``.
    Releases without a complete observation window are excluded.

    This defines association, not causation. The repository's synthetic data
    does not provide validated labels suitable for production model training.
    """
    required_releases = {"release_id", "service", "date"}
    required_incidents = {"related_release", "service", "date"}
    if not required_releases.issubset(release_history.columns):
        raise ValueError("release_history must contain release_id, service, and date.")
    if not required_incidents.issubset(incidents.columns):
        raise ValueError("incidents must contain related_release, service, and date.")
    if observation_days < 1:
        raise ValueError("observation_days must be at least 1.")

    end = pd.Timestamp(observation_end).normalize()
    releases = release_history.loc[:, ["release_id", "service", "date"]].copy()
    incident_rows = incidents.loc[:, ["related_release", "service", "date"]].copy()
    releases["date"] = pd.to_datetime(releases["date"], errors="coerce").dt.normalize()
    incident_rows["date"] = pd.to_datetime(incident_rows["date"], errors="coerce").dt.normalize()
    releases = releases.dropna(subset=["date"])
    incident_rows = incident_rows.dropna(subset=["date"])

    mature = releases[releases["date"] + pd.Timedelta(days=observation_days) <= end]
    results: list[dict[str, Any]] = []
    for row in mature.itertuples(index=False):
        window_end = row.date + pd.Timedelta(days=observation_days)
        linked = incident_rows[
            (incident_rows["related_release"] == row.release_id)
            & (incident_rows["service"] == row.service)
            & (incident_rows["date"] > row.date)
            & (incident_rows["date"] <= window_end)
        ]
        results.append(
            {
                "release_id": row.release_id,
                "service": row.service,
                "release_date": row.date,
                "outcome": int(not linked.empty),
            }
        )
    return pd.DataFrame(
        results, columns=["release_id", "service", "release_date", "outcome"]
    )


def prepare_release_features(
    *,
    service: str,
    files_changed: int,
    lines_changed: int,
    affected_services: int,
    deploy_date: date | datetime | pd.Timestamp,
    deploy_hour: int,
    service_metrics: Optional[pd.DataFrame],
    slo_definitions: Optional[pd.DataFrame],
    release_history: Optional[pd.DataFrame],
    incidents: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Create one model row from information available strictly before deploy.

    Metrics use observations in the seven calendar days before deployment.
    Incident history uses directly linked, same-service records in the prior
    30 days. Same-day and future telemetry/incidents are excluded.
    """
    if service_metrics is None or service_metrics.empty:
        raise FeatureDataUnavailable("No service metrics are available.")
    if slo_definitions is None or slo_definitions.empty:
        raise FeatureDataUnavailable("No SLO definitions are available.")
    if release_history is None or incidents is None:
        raise FeatureDataUnavailable(
            "Release history and incident data are required for historical context."
        )

    metric_columns = {"date", "service", "total_requests", "failed_requests"}
    if not metric_columns.issubset(service_metrics.columns):
        raise FeatureDataUnavailable("Service metrics are missing required columns.")
    if not {"service", "slo_target", "sli_type"}.issubset(slo_definitions.columns):
        raise FeatureDataUnavailable("SLO definitions are missing required columns.")
    if not {"release_id", "service", "date"}.issubset(release_history.columns):
        raise FeatureDataUnavailable("Release history is missing required columns.")
    if not {"related_release", "service", "date"}.issubset(incidents.columns):
        raise FeatureDataUnavailable("Incident data is missing required columns.")

    slo_rows = slo_definitions[slo_definitions["service"] == service]
    if slo_rows.empty:
        raise FeatureDataUnavailable(f"No SLO definition is available for {service}.")
    slo = slo_rows.iloc[0]
    if slo["sli_type"] != "availability":
        raise FeatureDataUnavailable(
            f"The SLO for {service} is not availability-based."
        )
    slo_target = float(slo["slo_target"])
    if not 0 < slo_target < 100:
        raise FeatureDataUnavailable(f"The SLO target for {service} is invalid.")

    deployment = pd.Timestamp(deploy_date).normalize()
    metrics = service_metrics[service_metrics["service"] == service].copy()
    metrics["date"] = pd.to_datetime(metrics["date"], errors="coerce").dt.normalize()
    metrics = metrics[
        (metrics["date"] < deployment)
        & (metrics["date"] >= deployment - pd.Timedelta(days=7))
    ]
    if metrics.empty:
        raise FeatureDataUnavailable(
            f"No pre-release metrics for {service} in the seven days before deployment."
        )

    total_requests = pd.to_numeric(metrics["total_requests"], errors="coerce").sum()
    failed_requests = pd.to_numeric(metrics["failed_requests"], errors="coerce").sum()
    if not np.isfinite(total_requests) or total_requests <= 0 or not np.isfinite(failed_requests):
        raise FeatureDataUnavailable("Pre-release request metrics are invalid or empty.")

    actual_error_rate = max(0.0, float(failed_requests)) / float(total_requests)
    allowed_error_rate = 1.0 - slo_target / 100.0
    burn_rate = actual_error_rate / allowed_error_rate
    budget_consumed_pct = burn_rate * 100.0

    releases = release_history[release_history["service"] == service].copy()
    releases["date"] = pd.to_datetime(releases["date"], errors="coerce").dt.normalize()
    releases = releases[
        releases["date"].notna()
        & (releases["date"] < deployment)
        & (releases["date"] >= deployment - pd.Timedelta(days=30))
    ]
    linked_ids = set(releases["release_id"].dropna())

    incident_rows = incidents[
        (incidents["service"] == service)
        & incidents["related_release"].isin(linked_ids)
    ].copy()
    incident_rows["date"] = pd.to_datetime(
        incident_rows["date"], errors="coerce"
    ).dt.normalize()
    historical_incidents = incident_rows[
        incident_rows["date"].notna()
        & (incident_rows["date"] < deployment)
        & (incident_rows["date"] >= deployment - pd.Timedelta(days=30))
    ]

    return pd.DataFrame(
        [
            {
                "files_changed": int(files_changed),
                "lines_changed": int(lines_changed),
                "affected_services": int(affected_services),
                "deploy_hour": int(deploy_hour),
                "budget_consumed_pct": budget_consumed_pct,
                "burn_rate": burn_rate,
                "recent_error_rate_pct": actual_error_rate * 100.0,
                "historical_incidents_30d": int(len(historical_incidents)),
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def predict_release_risk(
    model: Optional[TrainedReleaseModel],
    features: Optional[pd.DataFrame],
) -> Optional[float]:
    """Return the model's positive-class probability, or None if unavailable."""
    if model is None or features is None or features.empty:
        return None
    missing = set(FEATURE_COLUMNS) - set(features.columns)
    if missing:
        raise ValueError(f"Prediction features are missing columns: {sorted(missing)}")
    values = features.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Prediction features must contain finite numeric values.")
    return float(model.estimator.predict_proba(values)[0, 1])


def estimate_release_probability(
    *,
    service: str,
    files_changed: int,
    lines_changed: int,
    affected_services: int,
    deploy_date: date | datetime | pd.Timestamp,
    deploy_hour: int,
    service_metrics: Optional[pd.DataFrame],
    slo_definitions: Optional[pd.DataFrame],
    release_history: Optional[pd.DataFrame],
    incidents: Optional[pd.DataFrame],
) -> tuple[Optional[float], str, str, Optional[dict[str, float]], Optional[pd.DataFrame]]:
    """Return a probability estimate along with a clear status and disclaimer."""
    try:
        features = prepare_release_features(
            service=service,
            files_changed=files_changed,
            lines_changed=lines_changed,
            affected_services=affected_services,
            deploy_date=deploy_date,
            deploy_hour=deploy_hour,
            service_metrics=service_metrics,
            slo_definitions=slo_definitions,
            release_history=release_history,
            incidents=incidents,
        )
    except FeatureDataUnavailable as exc:
        return None, "unavailable", f"ML estimate unavailable: {exc}", None, None

    demo_model = train_simulated_demo_model()
    probability = predict_release_risk(demo_model, features)
    if probability is None:
        return None, "unavailable", "ML estimate unavailable: demo model did not produce a probability.", None, features

    return (
        probability,
        "simulated demo only",
        "Demonstration mode only: the project data is too sparse for a validated production predictor. "
        "This model uses synthetic, reproducible examples and does not represent real deployment risk.",
        demo_model.validation_metrics,
        features,
    )


def train_historical_model(
    features: pd.DataFrame,
    outcomes: pd.Series,
    release_dates: pd.Series,
    baseline_predictions: Optional[pd.Series] = None,
    random_state: int = DEMO_RANDOM_STATE,
) -> TrainedReleaseModel:
    """Fit a real-data pipeline only when chronological evaluation is viable."""
    if len(features) < MIN_HISTORICAL_RELEASES:
        raise InsufficientTrainingData(
            f"Need at least {MIN_HISTORICAL_RELEASES} labelled releases; received {len(features)}."
        )
    return _train_chronological_model(
        features,
        outcomes,
        release_dates,
        mode="historical",
        baseline_predictions=baseline_predictions,
        random_state=random_state,
        min_class_count=20,
    )


def generate_simulated_training_data(
    sample_count: int = DEMO_SAMPLE_COUNT,
    random_state: int = DEMO_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate reproducible examples with a documented synthetic outcome process."""
    if sample_count < 100:
        raise ValueError("sample_count must be at least 100 for chronological demo validation.")
    rng = np.random.default_rng(random_state)
    budget_consumed = rng.uniform(0.0, 220.0, sample_count)
    burn_rate = np.clip(
        budget_consumed / 100.0 + rng.normal(0.0, 0.35, sample_count), 0.0, 8.0
    )
    rows = pd.DataFrame(
        {
            "files_changed": rng.integers(1, 81, sample_count),
            "lines_changed": rng.integers(20, 10001, sample_count),
            "affected_services": rng.integers(1, 9, sample_count),
            "deploy_hour": rng.integers(0, 24, sample_count),
            "budget_consumed_pct": budget_consumed,
            "burn_rate": burn_rate,
            "recent_error_rate_pct": budget_consumed * 0.001,
            "historical_incidents_30d": rng.integers(0, 4, sample_count),
        },
        columns=FEATURE_COLUMNS,
    )

    # Synthetic outcomes are sampled from a declared toy process, not from
    # risk_engine scores or labels inferred from the repository's CSVs.
    logits = (
        -4.8
        + 0.70 * rows["files_changed"] / 20.0
        + 0.65 * rows["lines_changed"] / 2000.0
        + 0.35 * (rows["affected_services"] - 1)
        + 0.70 * rows["budget_consumed_pct"] / 100.0
        + 0.25 * rows["burn_rate"].clip(upper=5.0)
        + 0.55 * rows["historical_incidents_30d"]
        + 0.25 * rows["deploy_hour"].between(9, 18).astype(float)
        + 1.5 * rows["recent_error_rate_pct"]
    )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    labels = pd.Series(
        rng.binomial(1, probabilities), name="outcome", dtype="int64"
    )
    dates = pd.Series(
        pd.date_range("2020-01-01", periods=sample_count, freq="D"),
        name="release_date",
    )
    return rows, labels, dates


def train_simulated_demo_model(
    sample_count: int = DEMO_SAMPLE_COUNT,
    random_state: int = DEMO_RANDOM_STATE,
) -> TrainedReleaseModel:
    """Train and evaluate the explicitly simulated demonstration classifier."""
    features, outcomes, dates = generate_simulated_training_data(
        sample_count=sample_count, random_state=random_state
    )
    split_index = _chronological_split_index(len(features))
    baseline = pd.Series(
        [
            int(_score_demo_rule_baseline(row) > THRESHOLD_GO)
            for row in features.iloc[split_index:].itertuples(index=False)
        ],
        index=features.index[split_index:],
    )
    return _train_chronological_model(
        features,
        outcomes,
        dates,
        mode="simulated demo only",
        baseline_predictions=baseline,
        random_state=random_state,
        min_class_count=1,
    )


def _chronological_split_index(sample_count: int) -> int:
    split_index = int(sample_count * 0.8)
    if split_index < 2 or split_index >= sample_count:
        raise InsufficientTrainingData("Not enough records for chronological validation.")
    return split_index


def _training_metrics(
    actual: np.ndarray, predicted: np.ndarray, probabilities: Optional[np.ndarray] = None
) -> dict[str, float]:
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }
    if probabilities is not None and len(np.unique(actual)) == 2:
        result["roc_auc"] = float(roc_auc_score(actual, probabilities))
        result["average_precision"] = float(average_precision_score(actual, probabilities))
    return result


def _train_chronological_model(
    features: pd.DataFrame,
    outcomes: pd.Series,
    release_dates: pd.Series,
    *,
    mode: str,
    baseline_predictions: Optional[pd.Series],
    random_state: int,
    min_class_count: int,
) -> TrainedReleaseModel:
    if not set(FEATURE_COLUMNS).issubset(features.columns):
        raise ValueError("Training features do not contain the required model columns.")
    if not (len(features) == len(outcomes) == len(release_dates)):
        raise ValueError("Features, outcomes, and release dates must have equal lengths.")
    labels = pd.to_numeric(outcomes, errors="coerce").to_numpy()
    if not np.isfinite(labels).all() or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Outcomes must contain only binary 0/1 labels.")
    dates = pd.to_datetime(release_dates, errors="coerce")
    if dates.isna().any():
        raise ValueError("Release dates must be valid for chronological validation.")
    if baseline_predictions is not None and len(baseline_predictions) != len(features) - _chronological_split_index(len(features)):
        raise ValueError("Baseline predictions must align with the chronological validation set.")

    order = np.argsort(dates.to_numpy(), kind="stable")
    split_index = _chronological_split_index(len(features))
    train_indices, validation_indices = order[:split_index], order[split_index:]
    y_train, y_validation = labels[train_indices].astype(int), labels[validation_indices].astype(int)
    if min(np.bincount(y_train, minlength=2)) < min_class_count:
        raise InsufficientTrainingData("Training partition lacks enough examples of both outcomes.")
    if min(np.bincount(y_validation, minlength=2)) < min_class_count:
        raise InsufficientTrainingData("Validation partition lacks enough examples of both outcomes.")

    x = features.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Training features must contain finite numeric values.")
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=random_state),
    )
    estimator.fit(x.iloc[train_indices], y_train)
    validation_probabilities = estimator.predict_proba(x.iloc[validation_indices])[:, 1]
    validation_predictions = (validation_probabilities >= 0.5).astype(int)
    validation_metrics = _training_metrics(
        y_validation, validation_predictions, validation_probabilities
    )

    baseline_metrics = None
    if baseline_predictions is not None:
        baseline_values = np.asarray(baseline_predictions, dtype=int)
        if not set(np.unique(baseline_values)).issubset({0, 1}):
            raise ValueError("Baseline predictions must contain only binary 0/1 values.")
        baseline_metrics = _training_metrics(y_validation, baseline_values)

    return TrainedReleaseModel(
        estimator=estimator,
        mode=mode,
        validation_metrics=validation_metrics,
        baseline_metrics=baseline_metrics,
    )


def _score_demo_rule_baseline(row: Any) -> float:
    """Score one synthetic example with ReleaseGuard's unchanged rule engine."""
    values = row._asdict()
    today = date.today()
    service = "simulated-service"
    dates = [today - timedelta(days=offset) for offset in range(7, 0, -1)]
    total_per_day = 100_000
    failed_per_day = round(values["recent_error_rate_pct"] / 100.0 * total_per_day)
    metrics = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "service": service,
            "total_requests": total_per_day,
            "failed_requests": failed_per_day,
        }
    )
    release_rows = []
    incident_rows = []
    for index in range(int(values["historical_incidents_30d"])):
        release_id = f"SIM-PRIOR-{index}"
        release_date = today - timedelta(days=index + 2)
        release_rows.append(
            {"release_id": release_id, "service": service, "date": pd.Timestamp(release_date)}
        )
        incident_rows.append(
            {
                "related_release": release_id,
                "service": service,
                "date": pd.Timestamp(release_date + timedelta(days=1)),
                "severity": "Major",
            }
        )
    result = score_release(
        service=service,
        files_changed=int(values["files_changed"]),
        lines_changed=int(values["lines_changed"]),
        affected_services=int(values["affected_services"]),
        deploy_date=today,
        deploy_hour=int(values["deploy_hour"]),
        service_metrics=metrics,
        slo_definitions=pd.DataFrame(
            [{"service": service, "slo_target": 99.9, "sli_type": "availability"}]
        ),
        release_history=pd.DataFrame(
            release_rows, columns=["release_id", "service", "date"]
        ),
        incidents=pd.DataFrame(
            incident_rows, columns=["related_release", "service", "date", "severity"]
        ),
    )
    return result.risk_score
