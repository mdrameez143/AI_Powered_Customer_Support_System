"""
End-to-end tests for the five sample queries listed in the assessment brief
(Section 9). The LLM's two calls (question -> spec, result -> NL answer) are
mocked with the spec a well-behaved LLM should produce for that question, so
these tests check the *execution* layer's correctness deterministically
without needing a live LLM key — but they exercise the exact same
execute_spec()/get_anomaly_report() code path a real answer would go through.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.data_loader import load_tickets
from app.query_engine import answer_question


@pytest.fixture(scope="module")
def df():
    return load_tickets()


def _mock_answer(spec, expected_answer_substring, df):
    """Run answer_question with chat_json mocked to `spec` and chat mocked to
    a plausible NL answer; return the result dict."""
    with patch("app.query_engine.llm_client.chat_json", return_value=spec), \
         patch("app.query_engine.llm_client.chat", return_value=expected_answer_substring):
        return answer_question("(question text irrelevant — spec is mocked)", df)


# 1. "How many tickets are currently open?"
def test_sample_query_how_many_open(df):
    spec = {"operation": "count", "filters": [{"column": "status", "op": "==", "value": "Open"}]}
    result = _mock_answer(spec, "There are 111 open tickets.", df)
    assert result["result"]["count"] == int((df["status"] == "Open").sum()) == 111


# 2. "Which agent resolved the most tickets this month?"
# "This month" is dataset-relative (see query_engine._build_system_prompt),
# i.e. March 2024 — the month containing the dataset's latest record.
def test_sample_query_agent_most_resolved_this_month(df):
    spec = {
        "operation": "groupby_count",
        "filters": [
            {"column": "status", "op": "==", "value": "Resolved"},
            {"column": "created_at", "op": ">=", "value": "2024-03-01"},
        ],
        "group_by": "agent_id",
        "limit": 1,
    }
    result = _mock_answer(spec, "AGT-04 resolved the most tickets this month.", df)

    manual = (
        df[(df["status"] == "Resolved") & (df["created_at"] >= "2024-03-01")]
        .groupby("agent_id").size().sort_values(ascending=False)
    )
    top_agent, top_count = next(iter(result["result"].items()))
    assert top_agent == manual.index[0]
    assert top_count == int(manual.iloc[0])


# 3. "Show me all Critical tickets not resolved within 12 hours."
# Must include BOTH tickets resolved late (resolution_time_hrs > 12) AND
# tickets still open/escalated past 12 hours (resolution_time_hrs is null).
def test_sample_query_critical_not_resolved_within_12h(df):
    spec = {
        "operation": "list",
        "filters": [
            {"column": "priority", "op": "==", "value": "Critical"},
            {"column": "resolution_time_hrs", "op": "not_resolved_within", "value": 12},
        ],
    }
    result = _mock_answer(spec, "There are 34 such Critical tickets.", df)

    crit = df[df["priority"] == "Critical"]
    resolved_late = crit["resolution_time_hrs"].notna() & (crit["resolution_time_hrs"] > 12)
    still_open = crit["resolution_time_hrs"].isna() & (crit["age_hrs"] > 12)
    manual_ids = set(crit[resolved_late | still_open]["ticket_id"])

    returned_ids = {row["ticket_id"] for row in result["result"]}
    assert returned_ids == manual_ids
    assert still_open.sum() > 0  # sanity: the still-unresolved branch isn't vacuous
    import math
    assert any(
        row["resolution_time_hrs"] is None or (isinstance(row["resolution_time_hrs"], float) and math.isnan(row["resolution_time_hrs"]))
        for row in result["result"]
    )  # unresolved ones present, not just resolved-late ones


# 4. "What is the average customer rating for Technical category tickets?"
def test_sample_query_avg_rating_technical(df):
    spec = {
        "operation": "average",
        "target_column": "customer_rating",
        "filters": [{"column": "category", "op": "==", "value": "Technical"}],
    }
    result = _mock_answer(spec, "The average rating for Technical tickets is 3.4.", df)
    expected = round(df.loc[df["category"] == "Technical", "customer_rating"].mean(), 2)
    assert result["result"]["average"] == expected


# 5. "Are there any anomalies in resolution times this week?"
# Baseline stats must come from the full dataset; only the returned tickets
# are scoped to the requested week.
def test_sample_query_anomalies_this_week(df):
    spec = {
        "operation": "anomaly_detection",
        "anomaly_type": "resolution_time",
        "filters": [{"column": "created_at", "op": ">=", "value": "2024-03-24"}],
    }
    result = _mock_answer(spec, "Yes, a few tickets from this week are resolution-time outliers.", df)

    window_ids = set(df.loc[df["created_at"] >= "2024-03-24", "ticket_id"])
    returned_ids = {t["ticket_id"] for t in result["result"]["tickets"]}
    assert returned_ids <= window_ids  # every returned ticket is actually in the window
    assert result["result"]["count"] == len(returned_ids)
