"""
Natural-language query handling for Support Ticket AI.

Design:
- The LLM is a planner/translator, not an executor.
- It emits a small structured JSON query specification.
- The specification is normalized and strictly validated locally.
- Pandas performs the actual computation.
- A second LLM call optionally turns the deterministic result into a short answer.

The public API intentionally preserves the existing:
    QueryResult
    execute_spec(spec, df)
    answer_question(question, df)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import re
from typing import Any

import pandas as pd

from . import llm_client
from .anomaly import get_anomaly_report
from .data_loader import schema_description


# ---------------------------------------------------------------------------
# Fixed query vocabulary
# ---------------------------------------------------------------------------

ALLOWED_COLUMNS = {
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
    "age_hrs",
}

NUMERIC_COLUMNS = {
    "response_time_hrs",
    "resolution_time_hrs",
    "customer_rating",
    "age_hrs",
}

ALLOWED_OPS = {
    "==",
    "!=",
    ">",
    "<",
    ">=",
    "<=",
    "contains",
    "not_resolved_within",
}

ALLOWED_ANOMALY_TYPES = {
    "resolution_time",
    "stale_high_priority",
    "all",
}

ALLOWED_OPERATIONS = {
    "count",
    "average",
    "sum",
    "list",
    "groupby_count",
    "groupby_average",
    "anomaly_detection",
}

ALLOWED_SORTS = {"asc", "desc"}

# Common natural-language aliases. Values are canonical dataset values.
VALUE_SYNONYMS: dict[str, dict[str, str]] = {
    "status": {
        "open": "Open",
        "unresolved": "Open",
        "unresolved tickets": "Open",
        "pending": "Open",
        "pending tickets": "Open",
        "still open": "Open",
        "not resolved": "Open",
        "not resolved tickets": "Open",
        "escalated": "Escalated",
        "escalated tickets": "Escalated",
        "closed": "Closed",
        "resolved": "Resolved",
        "completed": "Resolved",
        "complete": "Resolved",
    },
    "priority": {
        "critical": "Critical",
        "urgent": "Critical",
        "severe": "Critical",
        "high": "High",
        "medium": "Medium",
        "normal": "Medium",
        "low": "Low",
    },
    "category": {
        "technical": "Technical",
        "tech": "Technical",
        "billing": "Billing",
        "payment": "Billing",
        "payments": "Billing",
        "account": "Account",
        "accounts": "Account",
        "shipping": "Shipping",
        "delivery": "Shipping",
        "refund": "Billing",
    },
}

COLUMN_SYNONYMS = {
    "ticket": "ticket_id",
    "ticket id": "ticket_id",
    "ticket number": "ticket_id",
    "created": "created_at",
    "created date": "created_at",
    "creation date": "created_at",
    "response time": "response_time_hrs",
    "response time hours": "response_time_hrs",
    "resolution time": "resolution_time_hrs",
    "resolution time hours": "resolution_time_hrs",
    "rating": "customer_rating",
    "customer rating": "customer_rating",
    "satisfaction": "customer_rating",
    "customer satisfaction": "customer_rating",
    "agent": "agent_id",
    "agent id": "agent_id",
    "issue": "issue_summary",
    "issue description": "issue_summary",
    "age": "age_hrs",
    "age hours": "age_hrs",
}

AMBIGUOUS_PHRASES = (
    "bad tickets",
    "good tickets",
    "problematic tickets",
    "worst tickets",
    "best tickets",
    "important tickets",
    "old tickets",
    "better agent",
    "bad agents",
)

SYSTEM_PROMPT_TEMPLATE = """You are a query planner for a customer-support ticket dataset.

Translate the user's question into ONE JSON object and nothing else.
Do not output markdown or commentary.

Schema:
{{
  "operation": one of {operations},
  "filters": [
    {{"column": <column>, "op": <operator>, "value": <value>}}
  ],
  "group_by": <column or null>,
  "target_column": <column or null>,
  "limit": <positive integer or null>,
  "sort": "asc" or "desc" or null,
  "anomaly_type": "resolution_time" or "stale_high_priority" or "all" or null
}}

Allowed columns:
{columns}

Allowed operators:
{ops}

Allowed operations:
{operations}

Dataset schema:
{schema}

Dataset date range: {dataset_start} through {dataset_end}.

Rules:
- Use only the allowed columns, operations and operators.
- "open", "unresolved", "pending", and "still open" normally mean status == "Open".
- "resolved", "closed", and "completed" normally mean status == "Closed".
- "critical", "urgent", and "severe" normally mean priority == "Critical".
- "tech" means category == "Technical".
- "payment", "payments", and "refund" may map to Billing when the question clearly refers to ticket category.
- For groupby_count, group_by is required.
- For groupby_average, group_by and target_column are required.
- For average/sum/groupby_average, target_column must be numeric.
- "highest", "most", "maximum", "top", "best" => sort="desc".
- "lowest", "least", "minimum", "bottom", "worst" => sort="asc".
- "top N" => sort="desc", limit=N.
- "bottom N" => sort="asc", limit=N.
- If a single highest/lowest entity is requested, use limit=1.
- If the user asks only for a count, use operation="count".
- "not resolved within X hours", "unresolved after X hours", and "still open after X hours"
  should use:
  column="resolution_time_hrs", op="not_resolved_within", value=X.
  This includes both late resolved tickets and still-open tickets older than X hours.
- For a time period, use created_at filters. Date ranges are inclusive.
- For "this week", "last week", "this month", "last month", "recently", or "lately",
  calculate the dates relative to the latest date in the dataset ({dataset_end}),
  not today's real-world date.
- For an explicit date range, use >= for the start date and <= for the end date.
- For anomaly questions, use operation="anomaly_detection".
  Use resolution_time for resolution-time outliers, stale_high_priority for stale
  High/Critical tickets, and all for generic anomaly questions.
- When an anomaly question includes a date period, add created_at filters to scope
  returned anomalies. The anomaly baseline is still computed from the full dataset.
- Never generate Python, pandas, SQL, or executable code.
- Never invent columns or values.
- If the question cannot be represented safely by this schema, return:
  {{"operation":"clarification_required","reason":"<short reason>"}}
"""


ANSWER_SYSTEM_PROMPT = (
    "You answer questions about support tickets using ONLY the computed result "
    "given to you. Respond in 1-2 plain sentences. Do not invent numbers, names, "
    "dates, or facts that are not present in the result."
)


@dataclass
class QueryResult:
    spec: dict[str, Any]
    data: Any
    row_count: int
    warning: str | None = None


# ---------------------------------------------------------------------------
# Natural-language normalization helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _canonical_column(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    key = value.strip().lower().replace("_", " ")
    return COLUMN_SYNONYMS.get(key, value.strip())


def _canonical_value(column: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value

    raw = value.strip()
    key = raw.lower()

    aliases = VALUE_SYNONYMS.get(column, {})
    return aliases.get(key, raw)


def _parse_limit(value: Any) -> int | None:
    if value is None or value == "":
        return None

    try:
        limit = int(value)
    except (TypeError, ValueError):
        raise ValueError("limit must be a positive integer")

    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    if limit > 100:
        raise ValueError("limit cannot exceed 100")
    return limit


def _normalize_spec(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("LLM query spec must be a JSON object")

    normalized = dict(spec)

    operation = normalized.get("operation")
    if isinstance(operation, str):
        operation = operation.strip().lower()

    # Permit the planner's explicit clarification response.
    if operation == "clarification_required":
        reason = str(normalized.get("reason") or "The question is ambiguous.")
        return {
            "operation": "clarification_required",
            "reason": reason[:500],
        }

    normalized["operation"] = operation
    normalized["group_by"] = _canonical_column(normalized.get("group_by"))
    normalized["target_column"] = _canonical_column(normalized.get("target_column"))

    sort = normalized.get("sort")
    if isinstance(sort, str):
        sort = sort.strip().lower()
    normalized["sort"] = sort

    normalized["limit"] = _parse_limit(normalized.get("limit"))

    raw_filters = normalized.get("filters") or []
    if not isinstance(raw_filters, list):
        raise ValueError("filters must be a list")

    filters: list[dict[str, Any]] = []
    for item in raw_filters:
        if not isinstance(item, dict):
            raise ValueError("each filter must be an object")

        f = dict(item)
        f["column"] = _canonical_column(f.get("column"))
        f["op"] = str(f.get("op") or "").strip()
        f["value"] = _canonical_value(f["column"], f.get("value"))
        filters.append(f)

    normalized["filters"] = filters

    anomaly_type = normalized.get("anomaly_type")
    if isinstance(anomaly_type, str):
        anomaly_type = anomaly_type.strip().lower()
    normalized["anomaly_type"] = anomaly_type

    return normalized


def validate_spec(spec: Any) -> dict[str, Any]:
    """Normalize and validate an LLM-generated query specification."""
    normalized = _normalize_spec(spec)

    if normalized["operation"] == "clarification_required":
        return normalized

    operation = normalized.get("operation")
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(f"Unsupported operation: {operation}")

    sort = normalized.get("sort")
    if sort is not None and sort not in ALLOWED_SORTS:
        raise ValueError("sort must be 'asc', 'desc', or null")

    group_by = normalized.get("group_by")
    target = normalized.get("target_column")
    limit = normalized.get("limit")

    if group_by is not None and group_by not in ALLOWED_COLUMNS:
        raise ValueError(f"Unknown group_by column: {group_by}")

    if target is not None and target not in ALLOWED_COLUMNS:
        raise ValueError(f"Unknown target_column: {target}")

    if operation == "groupby_count" and group_by is None:
        raise ValueError("groupby_count requires group_by")

    if operation == "groupby_average":
        if group_by is None or target is None:
            raise ValueError(
                "groupby_average requires group_by and target_column"
            )
        if target not in NUMERIC_COLUMNS:
            raise ValueError(
                f"groupby_average target must be numeric: {target}"
            )

    if operation in {"average", "sum"}:
        if target is None:
            raise ValueError(f"{operation} requires target_column")
        if target not in NUMERIC_COLUMNS:
            raise ValueError(
                f"{operation} target must be numeric: {target}"
            )

    if operation == "anomaly_detection":
        anomaly_type = normalized.get("anomaly_type") or "all"
        if anomaly_type not in ALLOWED_ANOMALY_TYPES:
            raise ValueError(f"Unknown anomaly_type: {anomaly_type}")
        normalized["anomaly_type"] = anomaly_type

    for f in normalized["filters"]:
        column = f.get("column")
        op = f.get("op")

        if column not in ALLOWED_COLUMNS:
            raise ValueError(f"Unknown column in filter: {column}")

        if op not in ALLOWED_OPS:
            raise ValueError(f"Unknown filter operator: {op}")

        if f.get("value") is None:
            raise ValueError(f"Filter value cannot be null: {column}")

        if op == "not_resolved_within":
            if column != "resolution_time_hrs":
                raise ValueError(
                    "not_resolved_within must filter on resolution_time_hrs"
                )
            try:
                hours = float(f["value"])
            except (TypeError, ValueError):
                raise ValueError("not_resolved_within value must be numeric")
            if hours < 0:
                raise ValueError("not_resolved_within cannot be negative")

        if column == "created_at" and op in {
            "==", "!=", ">", "<", ">=", "<="
        }:
            try:
                pd.to_datetime(f["value"])
            except Exception as exc:
                raise ValueError(
                    f"Invalid created_at date: {f['value']}"
                ) from exc

        if column in NUMERIC_COLUMNS and op != "contains":
            try:
                float(f["value"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Numeric filter value required for {column}"
                )

    if operation in {"count", "list"} and target is not None:
        # Ignore harmless extra target fields from the planner.
        normalized["target_column"] = None

    if operation not in {"groupby_count", "groupby_average"}:
        normalized["group_by"] = None

    if operation not in {
        "groupby_count",
        "groupby_average",
        "list",
    }:
        # limit is useful for grouped/list results; for count/average it is
        # not meaningful.
        normalized["limit"] = None

    return normalized


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _date_filter(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict[str, Any]]:
    return [
        {
            "column": "created_at",
            "op": ">=",
            "value": start.date().isoformat(),
        },
        {
            "column": "created_at",
            "op": "<=",
            "value": end.date().isoformat(),
        },
    ]


def _relative_date_filters(question: str, dataset_end: pd.Timestamp) -> list[dict[str, Any]]:
    """Deterministically add inclusive created_at bounds for common date phrases."""
    q = question.lower()
    end = dataset_end.normalize()

    if re.search(r"\bthis week\b", q):
        start = end - timedelta(days=6)
        return _date_filter(start, end)

    if re.search(r"\blast week\b", q):
        start = end - timedelta(days=13)
        end_prev = end - timedelta(days=7)
        return _date_filter(start, end_prev)

    if re.search(r"\bthis month\b", q):
        start = end.replace(day=1)
        return _date_filter(start, end)

    if re.search(r"\blast month\b", q):
        current_month_start = end.replace(day=1)
        previous_month_end = current_month_start - timedelta(days=1)
        previous_month_start = previous_month_end.replace(day=1)
        return _date_filter(previous_month_start, previous_month_end)

    if re.search(r"\b(recently|lately|recent)\b", q):
        start = end - timedelta(days=6)
        return _date_filter(start, end)

    match = re.search(r"\b(?:last|past)\s+(\d+)\s+days?\b", q)
    if match:
        days = int(match.group(1))
        if days <= 0 or days > 3650:
            return []
        start = end - timedelta(days=days - 1)
        return _date_filter(start, end)

    return []


def _merge_date_filters(
    spec: dict[str, Any],
    question: str,
    dataset_end: pd.Timestamp,
) -> None:
    """Add deterministic inclusive relative-date filters if the planner omitted them."""
    if any(
        f.get("column") == "created_at"
        for f in spec.get("filters", [])
    ):
        return

    filters = _relative_date_filters(question, dataset_end)
    if filters:
        spec["filters"].extend(filters)


# ---------------------------------------------------------------------------
# Filter execution
# ---------------------------------------------------------------------------

def _apply_filters(
    df: pd.DataFrame,
    filters: list[dict[str, Any]],
) -> pd.DataFrame:
    result = df.copy()

    for f in filters or []:
        col, op, val = f.get("column"), f.get("op"), f.get("value")

        if col not in ALLOWED_COLUMNS:
            raise ValueError(f"Unknown column in filter: {col}")
        if op not in ALLOWED_OPS:
            raise ValueError(f"Unknown filter operator: {op}")

        series = result[col]

        if col == "created_at":
            val = pd.to_datetime(val)

        if col in NUMERIC_COLUMNS and op != "contains":
            val = float(val)

        if op == "==":
            result = result[series == val]
        elif op == "!=":
            result = result[series != val]
        elif op == ">":
            result = result[series > val]
        elif op == "<":
            result = result[series < val]
        elif op == ">=":
            result = result[series >= val]
        elif op == "<=":
            result = result[series <= val]
        elif op == "contains":
            result = result[
                series.astype(str)
                .str.contains(str(val), case=False, na=False, regex=False)
            ]
        elif op == "not_resolved_within":
            if col != "resolution_time_hrs":
                raise ValueError(
                    "not_resolved_within must filter on resolution_time_hrs"
                )

            hours = float(val)

            resolved_late = (
                result["resolution_time_hrs"].notna()
                & (result["resolution_time_hrs"] > hours)
            )

            still_open_past_threshold = (
                result["resolution_time_hrs"].isna()
                & (result["age_hrs"] > hours)
            )

            result = result[
                resolved_late | still_open_past_threshold
            ]

    return result


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def execute_spec(
    spec: dict[str, Any],
    df: pd.DataFrame,
) -> QueryResult:
    spec = validate_spec(spec)

    if spec["operation"] == "clarification_required":
        return QueryResult(
            spec=spec,
            data=None,
            row_count=0,
            warning=spec["reason"],
        )

    op = spec["operation"]
    filtered = _apply_filters(df, spec.get("filters") or [])

    target = spec.get("target_column")
    group_by = spec.get("group_by")
    limit = spec.get("limit")
    sort = spec.get("sort")

    if op == "count":
        return QueryResult(
            spec,
            {"count": int(len(filtered))},
            len(filtered),
        )

    if op in {"average", "sum"}:
        series = filtered[target].dropna()

        value = (
            float(series.mean())
            if op == "average"
            else float(series.sum())
        )

        return QueryResult(
            spec,
            {
                op: round(value, 2),
                "n": int(len(series)),
            },
            len(filtered),
        )

    if op == "list":
        cols = [
            "ticket_id",
            "category",
            "priority",
            "status",
            "agent_id",
            "resolution_time_hrs",
            "customer_rating",
            "issue_summary",
        ]

        result = filtered[cols]

        if limit:
            result = result.head(limit)

        return QueryResult(
            spec,
            result.to_dict(orient="records"),
            len(filtered),
        )

    if op == "groupby_count":
        counts = filtered.groupby(group_by).size()

        # Preserve the original engine behavior: groupby_count defaults
        # to descending order (highest count first).
        # An explicit sort="asc" reverses the order.
        ascending = sort == "asc" if sort is not None else False
        counts = counts.sort_values(ascending=ascending)

        if limit:
            counts = counts.head(limit)

        return QueryResult(
            spec,
            counts.to_dict(),
            len(filtered),
        )

    if op == "groupby_average":
        avgs = (
            filtered.groupby(group_by)[target]
            .mean()
            .dropna()
        )

        ascending = sort != "desc"
        avgs = avgs.sort_values(ascending=ascending).round(2)

        if limit:
            avgs = avgs.head(limit)

        return QueryResult(
            spec,
            avgs.to_dict(),
            len(filtered),
        )

    if op == "anomaly_detection":
        anomaly_type = spec.get("anomaly_type") or "all"

        # Baseline is always calculated from the full dataset.
        full_report = get_anomaly_report(df)

        row_filters = spec.get("filters") or []

        if row_filters:
            allowed_ids = set(filtered["ticket_id"])

            full_report = {
                key: {
                    **val,
                    "tickets": [
                        ticket
                        for ticket in val["tickets"]
                        if ticket["ticket_id"] in allowed_ids
                    ],
                    "count": sum(
                        1
                        for ticket in val["tickets"]
                        if ticket["ticket_id"] in allowed_ids
                    ),
                }
                for key, val in full_report.items()
            }

        if anomaly_type == "resolution_time":
            data = full_report["resolution_time_outliers"]
        elif anomaly_type == "stale_high_priority":
            data = full_report["stale_high_priority"]
        else:
            data = full_report

        row_count = len(filtered) if row_filters else len(df)

        return QueryResult(
            spec,
            data,
            row_count,
        )

    raise ValueError(f"Unhandled operation: {op}")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_system_prompt(df: pd.DataFrame) -> str:
    dataset_start = pd.to_datetime(df["created_at"].min()).normalize()
    dataset_end = pd.to_datetime(df["created_at"].max()).normalize()

    return SYSTEM_PROMPT_TEMPLATE.format(
        operations=sorted(ALLOWED_OPERATIONS),
        columns=sorted(ALLOWED_COLUMNS),
        ops=sorted(ALLOWED_OPS),
        schema=schema_description(),
        dataset_start=dataset_start.date().isoformat(),
        dataset_end=dataset_end.date().isoformat(),
    )


# ---------------------------------------------------------------------------
# Ambiguity handling
# ---------------------------------------------------------------------------

def _local_clarification(question: str) -> str | None:
    q = question.lower()

    for phrase in AMBIGUOUS_PHRASES:
        if phrase in q:
            if phrase in {
                "bad tickets",
                "good tickets",
                "problematic tickets",
                "worst tickets",
                "best tickets",
            }:
                return (
                    "Could you define what you mean by that? "
                    "You can specify priority, status, customer rating, "
                    "or resolution time."
                )

            if phrase == "recent tickets":
                return (
                    "What time period should I use for 'recent' "
                    "(for example, the last 7 days)?"
                )

            if phrase == "old tickets":
                return (
                    "What age threshold should I use for 'old' "
                    "(for example, older than 24 hours)?"
                )

            return (
                "Could you be more specific about the metric or filter "
                "you want me to use?"
            )

    if len(q) < 4:
        return "Please provide a more specific question about the support tickets."

    return None


# ---------------------------------------------------------------------------
# Public NL query entry point
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    df: pd.DataFrame,
) -> dict[str, Any]:
    question = _normalize_text(question)

    local_clarification = _local_clarification(question)

    if local_clarification:
        return {
            "question": question,
            "query_spec": {
                "operation": "clarification_required",
                "reason": local_clarification,
            },
            "result": None,
            "matched_rows": 0,
            "answer": local_clarification,
        }

    if df.empty:
        return {
            "question": question,
            "query_spec": None,
            "result": None,
            "matched_rows": 0,
            "answer": "The ticket dataset is empty, so I cannot answer this question.",
        }

    system_prompt = _build_system_prompt(df)
    user_prompt = f'Question: "{question}"'

    try:
        raw_spec = llm_client.chat_json(system_prompt, user_prompt)
        spec = validate_spec(raw_spec)
    except (ValueError, KeyError, TypeError) as exc:
        return {
            "question": question,
            "query_spec": None,
            "result": None,
            "matched_rows": 0,
            "answer": (
                "I couldn't safely interpret that question. "
                "Please rephrase it with a specific ticket metric, "
                "filter, or grouping."
            ),
            "error": f"Could not execute the generated query: {exc}",
        }
    except llm_client.LLMError as exc:
        raise

    if spec["operation"] == "clarification_required":
        reason = spec["reason"]

        return {
            "question": question,
            "query_spec": spec,
            "result": None,
            "matched_rows": 0,
            "answer": reason,
        }

    # Deterministically handle common relative-date phrases. This prevents
    # the LLM from silently using today's date instead of the dataset snapshot.
    dataset_end = pd.to_datetime(df["created_at"].max()).normalize()
    _merge_date_filters(spec, question, dataset_end)

    try:
        result = execute_spec(spec, df)
    except (ValueError, KeyError, TypeError) as exc:
        return {
            "question": question,
            "query_spec": spec,
            "result": None,
            "matched_rows": 0,
            "answer": (
                "I couldn't safely execute the interpreted query. "
                "Please rephrase the question."
            ),
            "error": f"Could not execute the generated query: {exc}",
        }

    if result.warning:
        return {
            "question": question,
            "query_spec": spec,
            "result": None,
            "matched_rows": 0,
            "answer": result.warning,
        }

    synthesis_prompt = (
        f'Question: "{question}"\n'
        f"Computed result (JSON): {result.data}\n"
        f"Matched rows: {result.row_count}"
    )

    try:
        nl_answer = llm_client.chat(
            ANSWER_SYSTEM_PROMPT,
            synthesis_prompt,
        ).strip()
    except llm_client.LLMError:
        nl_answer = None

    return {
        "question": question,
        "query_spec": spec,
        "result": result.data,
        "matched_rows": result.row_count,
        "answer": nl_answer or f"Result: {result.data}",
    }
