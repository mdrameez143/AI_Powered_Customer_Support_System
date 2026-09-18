from __future__ import annotations

# Must run before importing data_loader/llm_client — those modules read
# environment variables (GROQ_API_KEY, TICKETS_CSV_PATH, etc.) at import
# time, so .env has to be loaded into the process environment first.
from dotenv import load_dotenv

load_dotenv()

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import llm_client
from .anomaly import get_anomaly_report
from .data_loader import load_tickets, reload_tickets
from .query_engine import answer_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("support_ticket_ai")

app = FastAPI(
    title="AI Powered Customer Support Ticket Analysis",
    description="NL querying + anomaly detection over a customer support ticket dataset.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["How many tickets are currently open?"])


class QueryResponse(BaseModel):
    question: str
    answer: str
    result: dict | list | None = None
    matched_rows: int | None = None
    query_spec: dict | None = None


@app.get("/health")
def health():
    try:
        df = load_tickets()
        return {
            "status": "ok",
            "rows_loaded": int(len(df)),
            "llm_provider": llm_client.LLM_PROVIDER,
            "data_quality_warnings": df.attrs.get("data_quality_warnings", []),
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Data failed to load: {e}") from e


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    df = load_tickets()
    try:
        result = answer_question(req.question, df)
    except llm_client.LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("Unexpected error handling query")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.get("/anomalies")
def anomalies():
    df = load_tickets()
    return get_anomaly_report(df)


@app.post("/admin/reload")
def admin_reload():
    """
    Re-read the CSV from disk without restarting the process.

    Intentionally unauthenticated: this is a local assessment prototype, not
    a production deployment. In production this would sit behind the same
    auth/authorization as the rest of an admin surface, not be bolted on here.
    """
    df = reload_tickets()
    return {"status": "reloaded", "rows_loaded": int(len(df))}
