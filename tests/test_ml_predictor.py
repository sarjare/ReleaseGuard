from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import load_all_data
from src.ml_predictor import (
    FeatureDataUnavailable,
    InsufficientTrainingData,
    estimate_release_probability,
    generate_simulated_training_data,
    prepare_release_features,
    predict_release_risk,
    train_historical_model,
    train_simulated_demo_model,
)


def test_prepare_release_features_uses_only_pre_release_data():
    data = load_all_data()
    features = prepare_release_features(
        service="user-api",
        files_changed=4,
        lines_changed=300,
        affected_services=2,
        deploy_date="2026-10-28",
        deploy_hour=14,
        service_metrics=data.service_metrics,
        slo_definitions=data.slo_definitions,
        release_history=data.release_history,
        incidents=data.incidents,
    )
    assert list(features.columns) == [
        "files_changed",
        "lines_changed",
        "affected_services",
        "deploy_hour",
        "budget_consumed_pct",
        "burn_rate",
        "recent_error_rate_pct",
        "historical_incidents_30d",
    ]
    assert features.loc[0, "deploy_hour"] == 14
    assert features.loc[0, "affected_services"] == 2


def test_train_simulated_demo_model_predicts_probability():
    model = train_simulated_demo_model(sample_count=200)
    features, _, _ = generate_simulated_training_data(sample_count=200)
    prob = predict_release_risk(model, features.iloc[:1])
    assert 0.0 <= prob <= 1.0


def test_estimate_release_probability_returns_demo_warning():
    data = load_all_data()
    prob, status, warning, metrics, features = estimate_release_probability(
        service="user-api",
        files_changed=4,
        lines_changed=300,
        affected_services=2,
        deploy_date="2026-10-28",
        deploy_hour=14,
        service_metrics=data.service_metrics,
        slo_definitions=data.slo_definitions,
        release_history=data.release_history,
        incidents=data.incidents,
    )
    assert status == "simulated demo only"
    assert 0.0 <= prob <= 1.0
    assert "Demonstration mode only" in warning
    assert features is not None


def test_prepare_release_features_requires_data():
    data = load_all_data()
    with pytest.raises(FeatureDataUnavailable):
        prepare_release_features(
            service="user-api",
            files_changed=4,
            lines_changed=300,
            affected_services=2,
            deploy_date="2026-10-28",
            deploy_hour=14,
            service_metrics=None,
            slo_definitions=data.slo_definitions,
            release_history=data.release_history,
            incidents=data.incidents,
        )


def test_historical_model_requires_enough_labelled_examples():
    rows = pd.DataFrame({
        "files_changed": [1, 2, 3, 4, 5],
        "lines_changed": [100, 200, 300, 400, 500],
        "affected_services": [1, 1, 2, 2, 2],
        "deploy_hour": [10, 11, 12, 13, 14],
        "budget_consumed_pct": [10.0, 12.0, 15.0, 17.0, 20.0],
        "burn_rate": [0.5, 0.7, 0.9, 1.0, 1.2],
        "recent_error_rate_pct": [0.2, 0.4, 0.9, 1.2, 1.4],
        "historical_incidents_30d": [0, 0, 1, 1, 2],
    })
    outcomes = pd.Series([0, 1, 0, 1, 0])
    dates = pd.Series(pd.date_range("2020-01-01", periods=5, freq="D"))
    with pytest.raises(InsufficientTrainingData):
        train_historical_model(rows, outcomes, dates)
