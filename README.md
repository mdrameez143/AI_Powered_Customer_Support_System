# AI Powered Customer Support System

An AI-powered system for querying and monitoring a customer support
ticket dataset in natural language, with rule-based anomaly detection.
Built for the DOTMappers AI Engineer take-home assessment.

## What it does

-   Loads `support_tickets.csv` (500 rows) into an in-memory pandas
    DataFrame.
-   Answers natural-language questions about the ticket data using an
    LLM.
-   Detects resolution-time outliers and stale unresolved High/Critical
    tickets.
-   Exposes the functionality through a FastAPI REST API and a Streamlit
    UI.
-   Supports two LLM backends: Groq and Ollama.

## Architecture Overview

``` text
                     ┌──────────────────────┐
                     │   Streamlit UI       │
                     │   app/ui.py          │
                     └──────────┬───────────┘
                                │ HTTP
                                ▼
                     ┌──────────────────────┐
                     │   FastAPI Backend    │
                     │   app/main.py        │
                     └──────────┬───────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
       ┌─────────────────┐             ┌─────────────────┐
       │  LLM Provider   │             │  Pandas Engine  │
       │ Groq / Ollama   │             │ query_engine.py │
       └────────┬────────┘             └────────┬────────┘
                │                              │
                │ Query → JSON spec            │ Execute
                └──────────────┬───────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ support_tickets.csv  │
                    │ → Pandas DataFrame    │
                    └──────────────────────┘
```

### How it works

1.  Streamlit provides the user interface and sends questions to FastAPI
    over HTTP.
2.  FastAPI coordinates the query workflow.
3.  Groq or Ollama interprets the natural-language question and produces
    a structured query specification.
4.  Pandas executes the validated specification against the in-memory
    ticket DataFrame.
5.  The LLM can turn the deterministic result into a concise
    natural-language answer.
6.  Anomaly detection uses deterministic rules rather than relying on
    the LLM.

**Key design principle:** The LLM does not execute generated Python,
pandas, or SQL code. It produces a constrained specification that the
application validates and executes itself.

## Project Structure

``` text
Support-Ticket-AI/
│
├── app/
│   ├── main.py                 # FastAPI application and API endpoints
│   ├── ui.py                   # Streamlit user interface
│   ├── llm_client.py           # Groq/Ollama LLM abstraction
│   ├── query_engine.py         # NL query planning and pandas execution
│   ├── anomaly.py              # Rule-based anomaly detection
│   ├── data_loader.py          # CSV loading and validation
│   └── data/
│       └── support_tickets.csv # 500-ticket dataset
│
├── tests/
│   └── ...                     # Unit and API tests
│
├── .env                        # Local configuration and API key (not committed)
├── .env.example                # Environment configuration template
├── .gitignore                  # Git exclusions
├── requirements.txt             # Python dependencies
├── Dockerfile                  # FastAPI container
├── Dockerfile.ui               # Streamlit container
├── docker-compose.yml          # API + UI orchestration
└── README.md                   # Project documentation
```

## Setup

### 1. Create and activate the virtual environment --- Windows

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

``` powershell
python -m pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example environment file:

``` powershell
Copy-Item .env.example .env
```

Edit `.env` and choose either Groq or Ollama.

### 3. Choose an LLM backend

#### Option A --- Groq

Use Groq when you want a hosted LLM without installing a local model.

``` env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Get a Groq API key from:

`https://console.groq.com/keys`

Keep the real `.env` file private and never commit it to Git.

#### Option B --- Ollama

Use Ollama for a locally running LLM.

``` env
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

Install Ollama, then:

``` powershell
ollama pull llama3.1
ollama serve
```

If Ollama runs on the host while the API runs inside Docker:

``` env
OLLAMA_HOST=http://host.docker.internal:11434
```

### 4. Run the API

Open a terminal in the project directory:

``` powershell
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

API:

`http://localhost:8000`

Interactive API documentation:

`http://localhost:8000/docs`

### 5. Run the Streamlit UI

Open a second terminal:

``` powershell
.\venv\Scripts\Activate.ps1
streamlit run app/ui.py
```

Open:

`http://localhost:8502`

The Streamlit UI communicates with FastAPI through:

``` env
API_BASE_URL= http://localhost:8000
```

### 6. Run tests

``` powershell
pytest tests/ -v
```

## Configuration Reference

  --------------------------------------------------------------------------------
  Variable                Purpose                 Example
  ----------------------- ----------------------- --------------------------------
  `LLM_PROVIDER`          Selects the LLM backend `groq` or `ollama`

  `GROQ_API_KEY`          Groq API authentication `gsk_...`
                          key                     

  `GROQ_MODEL`            Groq model used for     `openai/gpt-oss-20b`
                          queries                 

  `OLLAMA_HOST`           Ollama server address   `http://localhost:11434`

  `OLLAMA_MODEL`          Local Ollama model      `llama3.1`

  `TICKETS_CSV_PATH`      Optional ticket CSV     `app/data/support_tickets.csv`
                          path                    

  `API_BASE_URL`          FastAPI URL used by     `http://localhost:8000`
                          Streamlit               
  --------------------------------------------------------------------------------

`GROQ_API_KEY` is required only when `LLM_PROVIDER=groq`. Ollama does
not require an API key.

## Docker

Run both services together:

``` powershell
docker compose up --build
```

Then open:

-   API: `http://localhost:8000`
-   Swagger: `http://localhost:8000/docs`
-   Streamlit: `http://localhost:8502`

For Docker + Ollama, use:

``` env
OLLAMA_HOST=http://host.docker.internal:11434
```

Groq is simpler for the Docker setup because the API container does not
need to reach a host-running Ollama server.

## API

  -----------------------------------------------------------------------
  Endpoint                Method                  Description
  ----------------------- ----------------------- -----------------------
  `/health`               GET                     Health, row count,
                                                  provider, and
                                                  data-quality
                                                  information

  `/query`                POST                    Converts a
                                                  natural-language
                                                  question into a query
                                                  spec, executes it, and
                                                  returns the result

  `/anomalies`            GET                     Returns detected
                                                  anomaly categories and
                                                  ticket details

  `/admin/reload`         POST                    Reloads the CSV without
                                                  restarting the API
  -----------------------------------------------------------------------

### Example query

``` bash
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which agent has the lowest average customer rating?"}'
```

Example request body:

``` json
{
  "question": "How many tickets are currently open?"
}
```

### Windows PowerShell curl

To avoid native PowerShell quoting issues, place the JSON in a file:

``` powershell
Set-Content -Path body.json -Value '{"question":"How many tickets are currently open?"}' -Encoding utf8

curl.exe -X POST "http://localhost:8000/query" `
  -H "Content-Type: application/json" `
  --data-binary "@body.json"
```

`body.json` is only a local testing file and should not be committed.

## Assessment Requirement Coverage

  Requirement                  Implementation
  ---------------------------- ----------------------------------
  CSV ingestion + validation   `data_loader.py`
  Natural-language querying    `query_engine.py` + LLM
  Anomaly detection            `anomaly.py`
  REST API                     FastAPI (`main.py`)
  Minimal UI                   Streamlit (`ui.py`)
  LLM integration              Groq or Ollama (`llm_client.py`)
  Health check                 `GET /health`
  NL query endpoint            `POST /query`
  Anomalies endpoint           `GET /anomalies`
  Tests                        `pytest`
  Containerization             Docker Compose

## Query Examples

Try questions such as:

-   `How many tickets are currently open?`
-   `Which agent resolved the most tickets?`
-   `What is the average customer rating for Technical category tickets?`
-   `Show me all Critical tickets not resolved within 12 hours.`
-   `Which agent has the lowest average customer rating?`
-   `Are there any anomalies in resolution times this week?`
-   `Show unresolved Critical tickets older than 24 hours.`
-   `Anything unusual in the data lately?`

## Safety and Design Notes

-   The LLM does not execute arbitrary generated code.
-   Query specifications are constrained, validated, and executed
    through the application's pandas logic.
-   Anomaly detection is deterministic and auditable.
-   The `/query` endpoint is stateless.
-   The application is designed as a take-home assessment prototype, not
    a production deployment.
-   `/admin/reload` is intentionally unauthenticated for the local
    prototype.

## Known Limitations

-   The query-spec vocabulary is fixed and complex multi-operation
    questions may need to be decomposed.
-   There is no conversation memory between `/query` calls.
-   Anomaly thresholds are fixed constants.
-   The ticket dataset is held in a single in-memory pandas DataFrame.
-   LLM calls do not currently implement retry/backoff.
-   The Streamlit UI is intentionally minimal.
-   `/admin/reload` has no authentication.

## Future Improvements

-   Add retry and backoff for LLM calls.
-   Add a small automated Q&A evaluation set.
-   Move larger datasets to DuckDB or PostgreSQL.
-   Add session support for follow-up questions.
-   Add authentication around administrative operations.
