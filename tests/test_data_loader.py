"""
test_data_loader.py — Tests for src/data_loader.py

Uses small, deterministic in-memory fixtures labelled as synthetic test data.
All fixtures are NOT real production observations.

Run with:
    pytest tests/ -v

Author: ReleaseGuard Team
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    LoadedData,
    _validate_slo_definitions,
    _validate_service_metrics,
    _validate_release_history,
    _validate_incidents,
    _validate_traffic_calendar,
    _cross_validate,
    load_all_data,
    get_services,
)


# ── Synthetic fixture helpers ──────────────────────────────────────────────────

def _make_slo_df(rows=None) -> pd.DataFrame:
    """Synthetic SLO definitions fixture."""
    if rows is None:
        rows = [
            {"service": "svc-a", "slo_target": 99.9, "sli_type": "availability", "description": "Test A"},
            {"service": "svc-b", "slo_target": 99.5, "sli_type": "availability", "description": "Test B"},
        ]
    return pd.DataFrame(rows)


def _make_metrics_df(rows=None) -> pd.DataFrame:
    """Synthetic service metrics fixture."""
    if rows is None:
        rows = [
            {"date": "2026-10-01", "service": "svc-a", "total_requests": 10000, "failed_requests": 5, "avg_latency_ms": 200},
            {"date": "2026-10-02", "service": "svc-a", "total_requests": 12000, "failed_requests": 8, "avg_latency_ms": 210},
            {"date": "2026-10-01", "service": "svc-b", "total_requests": 5000,  "failed_requests": 20, "avg_latency_ms": 350},
        ]
    return pd.DataFrame(rows)


def _make_releases_df(rows=None) -> pd.DataFrame:
    """Synthetic release history fixture."""
    if rows is None:
        rows = [
            {
                "release_id": "REL-001", "date": "2026-10-01", "deployment_hour": 14,
                "service": "svc-a", "files_changed": 5, "lines_changed": 200,
                "affected_services": 1, "author_tenure_months": 12, "change_type": "bugfix",
                "release_notes": "Test fix",
            },
            {
                "release_id": "REL-002", "date": "2026-10-02", "deployment_hour": 22,
                "service": "svc-b", "files_changed": 30, "lines_changed": 3000,
                "affected_services": 3, "author_tenure_months": 2, "change_type": "feature",
                "release_notes": "New feature",
            },
        ]
    return pd.DataFrame(rows)


def _make_incidents_df(rows=None) -> pd.DataFrame:
    """Synthetic incidents fixture."""
    if rows is None:
        rows = [
            {
                "incident_id": "INC-001", "date": "2026-10-01", "service": "svc-a",
                "related_release": "REL-001", "severity": "Major",
                "duration_min": 30, "summary": "Test incident",
            },
        ]
    return pd.DataFrame(rows)


def _make_calendar_df(rows=None) -> pd.DataFrame:
    """Synthetic traffic calendar fixture."""
    if rows is None:
        rows = [
            {"date": "2026-10-01", "is_peak": "False", "freeze_period": "False", "event": "Normal", "description": "Test"},
            {"date": "2026-10-02", "is_peak": "True",  "freeze_period": "True",  "event": "Campaign", "description": "Test"},
        ]
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# SLO DEFINITIONS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateSloDefinitions:
    def test_valid_slo_df_passes(self):
        df = _make_slo_df()
        result = LoadedData()
        out = _validate_slo_definitions(df, result)
        assert len(out) == 2
        assert result.warnings == []

    def test_invalid_slo_target_zero_dropped(self):
        rows = [
            {"service": "svc-a", "slo_target": 0, "sli_type": "availability", "description": "bad"},
            {"service": "svc-b", "slo_target": 99.9, "sli_type": "availability", "description": "ok"},
        ]
        df = _make_slo_df(rows)
        result = LoadedData()
        out = _validate_slo_definitions(df, result)
        assert len(out) == 1
        assert out.iloc[0]["service"] == "svc-b"
        assert len(result.warnings) == 1

    def test_slo_target_100_dropped(self):
        rows = [{"service": "svc-a", "slo_target": 100, "sli_type": "availability", "description": "edge"}]
        df = _make_slo_df(rows)
        result = LoadedData()
        out = _validate_slo_definitions(df, result)
        assert out.empty
        assert len(result.warnings) == 1

    def test_non_availability_sli_warns(self):
        rows = [
            {"service": "svc-a", "slo_target": 99.0, "sli_type": "latency", "description": "latency"},
        ]
        df = _make_slo_df(rows)
        result = LoadedData()
        out = _validate_slo_definitions(df, result)
        assert len(out) == 1  # Not dropped, just warned
        assert any("availability" in w for w in result.warnings)

    def test_service_names_stripped(self):
        rows = [{"service": "  svc-a  ", "slo_target": 99.9, "sli_type": "availability", "description": "test"}]
        df = _make_slo_df(rows)
        result = LoadedData()
        out = _validate_slo_definitions(df, result)
        assert out.iloc[0]["service"] == "svc-a"


# ═══════════════════════════════════════════════════════════════════════════════
# SERVICE METRICS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateServiceMetrics:
    def test_valid_metrics_pass(self):
        df = _make_metrics_df()
        result = LoadedData()
        out = _validate_service_metrics(df, result)
        assert len(out) == 3
        assert pd.api.types.is_datetime64_any_dtype(out["date"])

    def test_negative_requests_dropped(self):
        rows = [
            {"date": "2026-10-01", "service": "svc-a", "total_requests": -1, "failed_requests": 0},
            {"date": "2026-10-01", "service": "svc-b", "total_requests": 1000, "failed_requests": 5},
        ]
        df = _make_metrics_df(rows)
        result = LoadedData()
        out = _validate_service_metrics(df, result)
        assert len(out) == 1
        assert len(result.warnings) >= 1

    def test_invalid_date_rows_dropped(self):
        rows = [
            {"date": "not-a-date", "service": "svc-a", "total_requests": 1000, "failed_requests": 5},
            {"date": "2026-10-01", "service": "svc-b", "total_requests": 500,  "failed_requests": 2},
        ]
        df = _make_metrics_df(rows)
        result = LoadedData()
        out = _validate_service_metrics(df, result)
        assert len(out) == 1
        assert len(result.warnings) >= 1

    def test_failed_greater_than_total_warns(self):
        rows = [{"date": "2026-10-01", "service": "svc-a", "total_requests": 100, "failed_requests": 200}]
        df = _make_metrics_df(rows)
        result = LoadedData()
        out = _validate_service_metrics(df, result)
        # Row kept but warning issued
        assert len(out) == 1
        assert any("failed_requests > total_requests" in w for w in result.warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# RELEASE HISTORY VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateReleaseHistory:
    def test_valid_releases_pass(self):
        df = _make_releases_df()
        result = LoadedData()
        out = _validate_release_history(df, result)
        assert len(out) == 2
        assert pd.api.types.is_datetime64_any_dtype(out["date"])

    def test_invalid_deployment_hour_warns(self):
        rows = [
            {
                "release_id": "REL-001", "date": "2026-10-01", "deployment_hour": 25,
                "service": "svc-a", "files_changed": 5, "lines_changed": 100,
                "affected_services": 1, "change_type": "bugfix", "release_notes": "",
            }
        ]
        df = _make_releases_df(rows)
        result = LoadedData()
        out = _validate_release_history(df, result)
        assert len(result.warnings) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossValidation:
    def test_orphaned_incident_release_ref_warns(self):
        releases = _make_releases_df()
        result = LoadedData()
        result.release_history = _validate_release_history(releases, result)

        # Incident points to a release ID that doesn't exist
        inc_rows = [
            {
                "incident_id": "INC-999", "date": "2026-10-01", "service": "svc-a",
                "related_release": "REL-NONEXISTENT", "severity": "Major",
                "duration_min": 10, "summary": "test",
            }
        ]
        inc_df = _make_incidents_df(inc_rows)
        result.incidents = _validate_incidents(inc_df, result)

        _cross_validate(result)
        assert any("REL-NONEXISTENT" in w for w in result.warnings)

    def test_service_with_metrics_but_no_slo_warns(self):
        releases = _make_releases_df()
        result = LoadedData()
        result.release_history = _validate_release_history(releases, result)

        # Metrics has svc-x, SLO only has svc-a
        metrics_rows = [
            {"date": "2026-10-01", "service": "svc-x", "total_requests": 1000, "failed_requests": 5},
        ]
        slo_rows = [
            {"service": "svc-a", "slo_target": 99.9, "sli_type": "availability", "description": "test"},
        ]
        result.service_metrics  = _validate_service_metrics(_make_metrics_df(metrics_rows), result)
        result.slo_definitions  = _validate_slo_definitions(_make_slo_df(slo_rows), result)
        result.incidents = None

        _cross_validate(result)
        assert any("svc-x" in w for w in result.warnings)

    def test_valid_linked_incidents_no_orphan_warning(self):
        releases = _make_releases_df()
        result = LoadedData()
        result.release_history = _validate_release_history(releases, result)

        # REL-001 exists in release_history
        inc_rows = [
            {
                "incident_id": "INC-001", "date": "2026-10-01", "service": "svc-a",
                "related_release": "REL-001", "severity": "Major",
                "duration_min": 30, "summary": "Test",
            }
        ]
        result.incidents = _validate_incidents(_make_incidents_df(inc_rows), result)

        initial_warning_count = len(result.warnings)
        _cross_validate(result)
        # No new orphan warnings
        assert len(result.warnings) == initial_warning_count


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD ALL DATA — FILE-LEVEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadAllData:
    def test_missing_data_dir_records_errors(self, tmp_path):
        """Loading from a non-existent directory records errors, returns LoadedData."""
        empty_dir = tmp_path / "nonexistent"
        result = load_all_data(data_dir=empty_dir)
        assert isinstance(result, LoadedData)
        assert len(result.errors) > 0
        assert not result.is_usable

    def test_loads_real_data_files(self):
        """Smoke test: load the actual project datasets and check usability."""
        result = load_all_data()
        # Should be usable — if this fails, check data/ directory
        assert result.is_usable, (
            f"Data failed to load. Errors: {result.errors}"
        )
        assert result.slo_definitions is not None
        assert result.service_metrics is not None
        assert result.release_history is not None

    def test_empty_csv_warns_not_crashes(self, tmp_path):
        """An empty CSV should produce a warning, not a crash."""
        # Create directory with just an empty slo_definitions.csv
        (tmp_path / "slo_definitions.csv").write_text(
            "service,slo_target,sli_type,description\n"
        )
        # Other files missing → errors, but no unhandled exception
        result = load_all_data(data_dir=tmp_path)
        assert isinstance(result, LoadedData)

    def test_malformed_csv_records_error(self, tmp_path):
        """A completely malformed CSV should record an error gracefully."""
        (tmp_path / "slo_definitions.csv").write_text("not,,,,valid\x00csv content\n\x01\x02")
        result = load_all_data(data_dir=tmp_path)
        assert isinstance(result, LoadedData)

    def test_get_services_returns_sorted_list(self):
        """get_services should return a sorted list of unique service names."""
        result = load_all_data()
        services = get_services(result)
        assert services == sorted(services)
        assert len(services) > 0
        assert all(isinstance(s, str) for s in services)

    def test_duplicate_release_id_warns(self):
        """Duplicate release IDs should trigger a warning."""
        result = LoadedData()
        dup_releases = _make_releases_df([
            {
                "release_id": "REL-DUP", "date": "2026-10-01", "deployment_hour": 14,
                "service": "svc-a", "files_changed": 5, "lines_changed": 200,
                "affected_services": 1, "change_type": "bugfix",
            },
            {
                "release_id": "REL-DUP", "date": "2026-10-02", "deployment_hour": 15,
                "service": "svc-a", "files_changed": 8, "lines_changed": 400,
                "affected_services": 1, "change_type": "refactor",
            },
        ])
        _validate_release_history(dup_releases, result)
        assert any("duplicate release_id" in w.lower() for w in result.warnings)

    def test_duplicate_incident_id_warns(self):
        """Duplicate incident IDs should trigger a warning."""
        result = LoadedData()
        dup_incidents = _make_incidents_df([
            {
                "incident_id": "INC-DUP", "date": "2026-10-01", "service": "svc-a",
                "related_release": "REL-001", "severity": "Major",
            },
            {
                "incident_id": "INC-DUP", "date": "2026-10-02", "service": "svc-a",
                "related_release": "REL-002", "severity": "Minor",
            },
        ])
        _validate_incidents(dup_incidents, result)
        assert any("duplicate incident_id" in w.lower() for w in result.warnings)

    def test_unknown_service_in_release_history_warns(self):
        """Services in release history without SLO definitions should warn."""
        result = LoadedData()
        result.release_history = _validate_release_history(
            _make_releases_df([{"release_id": "REL-X", "date": "2026-10-01", "deployment_hour": 10, "service": "unknown-svc", "files_changed": 1, "lines_changed": 10, "affected_services": 1, "change_type": "bugfix"}]),
            result,
        )
        result.slo_definitions = _validate_slo_definitions(_make_slo_df(), result)
        _cross_validate(result)
        assert any("unknown-svc" in w for w in result.warnings)

    def test_missing_required_column_fails_file_load(self, tmp_path):
        """A CSV missing required columns should record an error and not set DataFrame."""
        bad_slo = tmp_path / "slo_definitions.csv"
        bad_slo.write_text("service,description\nsvc-a,only two columns\n")
        result = LoadedData()
        from src.data_loader import _load_csv
        df = _load_csv("slo_definitions", bad_slo, result)
        assert df is None
        assert any("missing required columns" in e.lower() for e in result.errors)

    def test_original_csv_files_unmodified(self):
        """Verify that loading datasets does not mutate files on disk."""
        import hashlib
        from src.data_loader import DATA_DIR
        csv_files = list(DATA_DIR.glob("*.csv"))
        assert len(csv_files) >= 5
        hashes_before = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in csv_files}
        
        # Load all data multiple times
        _ = load_all_data()
        _ = load_all_data()

        hashes_after = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in csv_files}
        assert hashes_before == hashes_after
