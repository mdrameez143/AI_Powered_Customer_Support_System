"""
Tests focus on the deterministic parts (query_engine.execute_spec, anomaly
detection) since those don't require an LLM key to run in CI. The NL->spec
translation and answer synthesis (which do call the LLM) are covered by a
mocked test.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from app.anomaly import find_stale_high_priority, get_anomaly_report
from app.data_loader import _validate_data_quality, load_tickets
from app.query_engine import execute_spec


@pytest.fixture(scope="module")
def df():
    return load_tickets()


def test_load_tickets_schema(df):
    assert len(df) == 500
    assert "age_hrs" in df.columns


def test_count_open_tickets(df):
    spec = {"operation": "count", "filters": [{"column": "status", "op": "==", "value": "Open"}]}
    result = execute_spec(spec, df)
    assert result.data["count"] == int((df["status"] == "Open").sum())


def test_groupby_average_rating_by_agent(df):
    spec = {"operation": "groupby_average", "group_by": "agent_id", "target_column": "customer_rating", "limit": 1}
    result = execute_spec(spec, df)
    assert len(result.data) == 1
    lowest_agent = min(result.data, key=result.data.get)
    manual = df.groupby("agent_id")["customer_rating"].mean().idxmin()
    assert lowest_agent == manual


def test_average_technical_rating(df):
    spec = {
        "operation": "average",
        "target_column": "customer_rating",
        "filters": [{"column": "category", "op": "==", "value": "Technical"}],
    }
    result = execute_spec(spec, df)
    expected = df.loc[df["category"] == "Technical", "customer_rating"].mean()
    assert abs(result.data["average"] - round(expected, 2)) < 0.01


def test_rejects_unknown_column(df):
    spec = {"operation": "count", "filters": [{"column": "not_a_column", "op": "==", "value": "x"}]}
    with pytest.raises(ValueError):
        execute_spec(spec, df)


def test_stale_high_priority_only_flags_open_or_escalated(df):
    stale = find_stale_high_priority(df)
    assert set(stale["status"].unique()) <= {"Open", "Escalated"}
    assert set(stale["priority"].unique()) <= {"High", "Critical"}
    assert (stale["age_hrs"] > 24).all()


def test_anomaly_report_shape(df):
    report = get_anomaly_report(df)
    assert "resolution_time_outliers" in report
    assert "stale_high_priority" in report
    assert isinstance(report["resolution_time_outliers"]["count"], int)


def test_anomaly_detection_operation_routes_to_resolution_time(df):
    spec = {"operation": "anomaly_detection", "anomaly_type": "resolution_time"}
    result = execute_spec(spec, df)
    assert "tickets" in result.data
    assert result.data == get_anomaly_report(df)["resolution_time_outliers"]


def test_anomaly_detection_operation_routes_to_stale(df):
    spec = {"operation": "anomaly_detection", "anomaly_type": "stale_high_priority"}
    result = execute_spec(spec, df)
    assert result.data == get_anomaly_report(df)["stale_high_priority"]


def test_anomaly_detection_all_returns_both(df):
    spec = {"operation": "anomaly_detection", "anomaly_type": "all"}
    result = execute_spec(spec, df)
    assert "resolution_time_outliers" in result.data
    assert "stale_high_priority" in result.data


def test_anomaly_detection_rejects_unknown_type(df):
    spec = {"operation": "anomaly_detection", "anomaly_type": "not_a_real_type"}
    with pytest.raises(ValueError):
        execute_spec(spec, df)


def test_data_quality_warnings_attached_to_frame(df):
    # The bundled dataset is expected to be clean, but the attribute must
    # always exist so callers (e.g. /health) don't need a hasattr check.
    assert "data_quality_warnings" in df.attrs
    assert isinstance(df.attrs["data_quality_warnings"], list)


def test_validate_data_quality_raises_on_duplicate_ticket_id():
    bad = pd.DataFrame({
        "ticket_id": ["TKT-001", "TKT-001"],
        "created_at": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "category": ["Billing", "General"],
        "priority": ["Low", "Low"],
        "status": ["Resolved", "Resolved"],
        "response_time_hrs": [1.0, 1.0],
        "resolution_time_hrs": [2.0, 2.0],
        "agent_id": ["AGT-01", "AGT-01"],
        "customer_rating": [5, 5],
        "issue_summary": ["x", "y"],
    })
    with pytest.raises(ValueError, match="ticket_id must be unique"):
        _validate_data_quality(bad)


def test_validate_data_quality_warns_on_out_of_range_rating():
    bad = pd.DataFrame({
        "ticket_id": ["TKT-001"],
        "created_at": pd.to_datetime(["2024-01-01"]),
        "category": ["Billing"],
        "priority": ["Low"],
        "status": ["Resolved"],
        "response_time_hrs": [1.0],
        "resolution_time_hrs": [2.0],
        "agent_id": ["AGT-01"],
        "customer_rating": [9],
        "issue_summary": ["x"],
    })
    warnings = _validate_data_quality(bad)
    assert any("customer_rating" in w for w in warnings)
