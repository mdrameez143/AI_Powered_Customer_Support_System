import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.data_loader import load_tickets
from app.query_engine import answer_question


def test_answer_question_with_mocked_llm():
    df = load_tickets()

    spec = {"operation": "count", "filters": [{"column": "status", "op": "==", "value": "Open"}]}

    with patch("app.query_engine.llm_client.chat_json", return_value=spec), \
         patch("app.query_engine.llm_client.chat", return_value="There are 111 open tickets."):
        result = answer_question("How many tickets are open?", df)

    assert result["result"]["count"] == int((df["status"] == "Open").sum())
    assert "open" in result["answer"].lower()


def test_answer_question_handles_bad_spec_gracefully():
    df = load_tickets()
    bad_spec = {"operation": "count", "filters": [{"column": "nonexistent", "op": "==", "value": "x"}]}

    with patch("app.query_engine.llm_client.chat_json", return_value=bad_spec):
        result = answer_question("garbage question", df)

    assert "error" in result
