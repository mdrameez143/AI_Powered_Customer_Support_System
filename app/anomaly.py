"""
Anomaly detection over the ticket dataset.

Two independent, complementary rules (both mentioned explicitly in the brief):

1. Abnormally long resolution times: flagged per-category using a z-score
   against that category's own resolved tickets (z > 2). Per-category is
   important — "Technical" tickets naturally take longer than "General" ones,
   so a single global threshold would just flag every Technical ticket.

2. Unresolved high-priority tickets older than 24 hours: status in
   (Open, Escalated) AND priority in (High, Critical) AND age_hrs > 24.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

Z_SCORE_THRESHOLD = 2.0
STALE_HOURS_THRESHOLD = 24.0
STALE_PRIORITIES = {"High", "Critical"}
UNRESOLVED_STATUSES = {"Open", "Escalated"}


def find_resolution_time_outliers(df: pd.DataFrame, z_threshold: float = Z_SCORE_THRESHOLD) -> pd.DataFrame:
    resolved = df[df["resolution_time_hrs"].notna()].copy()
    if resolved.empty:
        return resolved

    resolved["category_mean_hrs"] = resolved.groupby("category")["resolution_time_hrs"].transform("mean")
    resolved["category_std_hrs"] = resolved.groupby("category")["resolution_time_hrs"].transform("std")
    resolved["category_std_hrs"] = resolved["category_std_hrs"].replace(0, pd.NA)
    resolved["z_score"] = (
        (resolved["resolution_time_hrs"] - resolved["category_mean_hrs"]) / resolved["category_std_hrs"]
    )

    outliers = resolved[resolved["z_score"] > z_threshold].copy()
    outliers["z_score"] = outliers["z_score"].round(2)
    outliers["category_mean_hrs"] = outliers["category_mean_hrs"].round(2)
    return outliers.sort_values("z_score", ascending=False)


def find_stale_high_priority(
    df: pd.DataFrame,
    hours_threshold: float = STALE_HOURS_THRESHOLD,
) -> pd.DataFrame:
    mask = (
        df["status"].isin(UNRESOLVED_STATUSES)
        & df["priority"].isin(STALE_PRIORITIES)
        & (df["age_hrs"] > hours_threshold)
    )
    stale = df[mask].copy()
    stale["age_hrs"] = stale["age_hrs"].round(1)
    return stale.sort_values("age_hrs", ascending=False)


def get_anomaly_report(df: pd.DataFrame) -> dict[str, Any]:
    outlier_cols = ["ticket_id", "category", "resolution_time_hrs", "category_mean_hrs", "z_score", "agent_id"]
    stale_cols = ["ticket_id", "priority", "status", "age_hrs", "agent_id", "issue_summary"]

    outliers = find_resolution_time_outliers(df)
    stale = find_stale_high_priority(df)

    return {
        "resolution_time_outliers": {
            "description": (
                f"Resolved tickets whose resolution time is a statistical outlier "
                f"(z-score > {Z_SCORE_THRESHOLD}) relative to their own category's average."
            ),
            "count": int(len(outliers)),
            "tickets": outliers[outlier_cols].to_dict(orient="records"),
        },
        "stale_high_priority": {
            "description": (
                f"High/Critical priority tickets still Open or Escalated after "
                f"more than {STALE_HOURS_THRESHOLD:g} hours."
            ),
            "count": int(len(stale)),
            "tickets": stale[stale_cols].to_dict(orient="records"),
        },
    }
