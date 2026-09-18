"""
Loads and preprocesses the support ticket CSV into a clean pandas DataFrame.

Design note: we load once at process start and keep the DataFrame in memory
(500 rows is trivial). For a larger dataset this would move to a real
database (DuckDB/SQLite) with the same query_engine interface on top.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import pandas as pd

logger = logging.getLogger("support_ticket_ai.data_loader")

DATA_PATH = os.getenv("TICKETS_CSV_PATH") or os.path.join(os.path.dirname(__file__), "data", "support_tickets.csv")

REQUIRED_COLUMNS = [
    "ticket_id", "created_at", "category", "priority", "status",
    "response_time_hrs", "resolution_time_hrs", "agent_id",
    "customer_rating", "issue_summary",
]


VALID_CATEGORIES = {"Billing", "Technical", "General"}
VALID_PRIORITIES = {"Low", "Medium", "High", "Critical"}
VALID_STATUSES = {"Open", "Resolved", "Escalated"}


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")


def _validate_data_quality(df: pd.DataFrame) -> list[str]:
    """
    Row-level sanity checks. These raise on structural corruption (duplicate
    IDs, unparseable dates) since downstream logic assumes those invariants,
    but only *warn* (return messages) on out-of-vocabulary values or
    out-of-range numbers — a malformed row in a 500-row CSV shouldn't take
    the whole service down, but it's worth surfacing to whoever's running it.
    """
    warnings: list[str] = []

    if not df["ticket_id"].is_unique:
        dupes = df.loc[df["ticket_id"].duplicated(), "ticket_id"].tolist()
        raise ValueError(f"ticket_id must be unique; duplicates found: {dupes}")

    if df["created_at"].isna().any():
        bad = df.loc[df["created_at"].isna(), "ticket_id"].tolist()
        raise ValueError(f"created_at could not be parsed for rows: {bad}")

    bad_category = df.loc[~df["category"].isin(VALID_CATEGORIES), "ticket_id"].tolist()
    if bad_category:
        warnings.append(f"Unexpected category values on rows: {bad_category}")

    bad_priority = df.loc[~df["priority"].isin(VALID_PRIORITIES), "ticket_id"].tolist()
    if bad_priority:
        warnings.append(f"Unexpected priority values on rows: {bad_priority}")

    bad_status = df.loc[~df["status"].isin(VALID_STATUSES), "ticket_id"].tolist()
    if bad_status:
        warnings.append(f"Unexpected status values on rows: {bad_status}")

    rating = df["customer_rating"].dropna()
    if not rating.between(1, 5).all():
        bad = df.loc[df["customer_rating"].notna() & ~df["customer_rating"].between(1, 5), "ticket_id"].tolist()
        warnings.append(f"customer_rating outside 1-5 on rows: {bad}")

    for col in ("response_time_hrs", "resolution_time_hrs"):
        neg = df.loc[df[col].notna() & (df[col] < 0), "ticket_id"].tolist()
        if neg:
            warnings.append(f"Negative {col} on rows: {neg}")

    # A ticket marked Resolved with no resolution_time_hrs (or vice versa) is
    # a data-consistency smell worth flagging, though not fatal.
    inconsistent = df.loc[
        (df["status"] == "Resolved") & df["resolution_time_hrs"].isna(), "ticket_id"
    ].tolist()
    if inconsistent:
        warnings.append(f"Status=Resolved but resolution_time_hrs is null on rows: {inconsistent}")

    return warnings


@lru_cache(maxsize=1)
def load_tickets(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_schema(df)

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    for col in ["response_time_hrs", "resolution_time_hrs", "customer_rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    warnings = _validate_data_quality(df)
    for w in warnings:
        logger.warning("Data quality issue: %s", w)
    df.attrs["data_quality_warnings"] = warnings

    # Derived column: age of a ticket in hours, relative to "now" for open tickets,
    # or its actual resolution time for closed ones. Used for the "unresolved > N hrs" checks.
    now = pd.Timestamp.now()
    df["age_hrs"] = (now - df["created_at"]).dt.total_seconds() / 3600.0

    return df


def reload_tickets() -> pd.DataFrame:
    """Bust the cache — useful if the underlying CSV changes at runtime."""
    load_tickets.cache_clear()
    return load_tickets()


def schema_description() -> str:
    """Human/LLM-readable schema summary, used inside the NL->query prompt."""
    return """
Columns available in the `tickets` table:
- ticket_id (string, e.g. "TKT-001")
- created_at (datetime)
- category (string: one of "Billing", "Technical", "General")
- priority (string: one of "Low", "Medium", "High", "Critical")
- status (string: one of "Open", "Resolved", "Escalated")
- response_time_hrs (float): hours from creation to first agent response
- resolution_time_hrs (float, null if unresolved): hours from creation to resolution
- agent_id (string, e.g. "AGT-04")
- customer_rating (integer 1-5, null if unresolved)
- issue_summary (free text)
- age_hrs (float, derived): hours since the ticket was created (as of now)
""".strip()
