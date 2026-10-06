# custom-agent

A local AI research agent powered by [Ollama](https://ollama.com). No cloud API keys required — everything runs on your machine.

## Requirements

- [Ollama](https://ollama.com) running locally with at least one model pulled
- [uv](https://docs.astral.sh/uv/) for Python dependency management

## Setup

```bash
# Pull a model if you haven't already
ollama pull qwen3.5

# Install dependencies
uv sync
```

## Usage

```bash
uv run python main.py
```

You'll get an interactive prompt:

```
Agent ready (model: qwen3.5:latest)
Commands: 'reset' to clear history, 'quit' to exit

You: What are the latest developments in fusion energy?
  [tool] web_search({'query': 'latest developments fusion energy 2025'})
  [tool] fetch_webpage({'url': 'https://...'})
Agent: ...
```

**Session commands:**

| Command | Effect |
|---------|--------|
| `reset` | Clear conversation history |
| `quit` / `exit` | Exit the agent |

## Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web via DuckDuckGo |
| `fetch_webpage` | Fetch and extract readable text from a URL |
| `parse_pdf` | Extract text from a local PDF file |
| `read_file` | Read a local text file |
| `list_directory` | List files in a directory |
| `calculate` | Evaluate math expressions (`sqrt`, `log`, etc.) |
| `get_current_datetime` | Return the current date and time |

## Stock news skill

`/news <TICKER>` archives the past month of news for a stock and summarizes it. One-time setup for screenshots:

```bash
uv run playwright install chromium
```

| Tool | Description |
|------|-------------|
| `fetch_stock_news` | Find recent articles (DuckDuckGo news), save original HTML + screenshot + `meta.json` under `~/.custom-agent/news/<TICKER>/` |
| `summarize_stock_news` | Per-article summaries (stored in `meta.json`) plus a combined dated overview |

`meta.json` records both `retrieved_at` (when we downloaded it) and `published_at` (extracted from the page's metadata, falling back to the search result date; `published_at_source` says which).

## REST API

A restricted HTTP API exposes selected skills as background jobs, for another app on the same machine to call. It does not expose the chat agent. The caller stores the results; the server keeps a SQLite history of every job (`~/.custom-agent/jobs.db`, override with `JOBS_DB`).

```bash
API_KEY=changeme uv run uvicorn server:get_app --factory --host 127.0.0.1 --port 8000
```

| Endpoint | Description |
|----------|-------------|
| `POST /news/{ticker}` | Queue a fetch-and-summarize job. Optional JSON body: `days` (default 30), `max_articles` (default 15). Returns `202` with the job; an identical queued/running job is reused (`200`) |
| `GET /jobs/{id}` | Job status (`queued`, `running`, `done`, `failed`), a `progress` block, and, when done, the result |
| `GET /jobs?status=&limit=` | Job history, newest first |

Send the key as an `X-API-Key` header. If `API_KEY` is unset the API is unauthenticated (a warning is logged), so keep it bound to `127.0.0.1`.

**How a job runs** (`news_pipeline.py`): list candidate articles from a news search, then for each one in turn fetch it (plain text only; no HTML or screenshots are kept), check it is primarily about the ticker, and summarize it; finally bundle the summaries into an overview. Each article's state is stored in the `job_articles` table (`pending` → `fetched` → `relevant` → `done`, or `skipped` with a `reason`). Jobs run one at a time, and a job interrupted by a restart is re-queued and resumes where it left off.

`progress` in `GET /jobs/{id}` has the current `stage` (`queued`, `listing`, `processing`, `bundling`, `done`, `failed`), the article `total`, per-status `counts`, and each article's `status` and `reason`.

| Env var | Default | Purpose |
|---------|---------|---------|
| `API_KEY` | unset | Required `X-API-Key` value |
| `JOBS_DB` | `~/.custom-agent/jobs.db` | SQLite job history |
| `NEWS_MODEL` | `qwen3.6:35b` | Ollama model for everything below |
| `NEWS_RELEVANCE_MODEL` | `NEWS_MODEL` | Model for the relevance check (a smaller one works) |
| `NEWS_SUMMARY_MODEL` | `NEWS_MODEL` | Model for article summaries and the overview |

A finished news job's `result` has `ticker`, `generated_at`, `overview`, and `articles` (each with `title`, `source`, `url`, `published_at`, `retrieved_at`, `summary`), plus `skipped`.

```bash
curl -X POST -H "X-API-Key: changeme" localhost:8000/news/AAPL
curl -H "X-API-Key: changeme" localhost:8000/jobs/<id>
```

## Changing the model

Edit the `MODEL` constant at the top of `main.py`:

```python
MODEL = "qwen3.5:latest"  # change to any model you have in Ollama
```

## Adding a tool

1. Write the function in `tools.py`
2. Add its JSON schema to `TOOL_DEFINITIONS`
3. Register it in `TOOL_FUNCTIONS`

```python
def my_tool(arg: str) -> str:
    return f"result for {arg}"

TOOL_FUNCTIONS = {
    ...
    "my_tool": my_tool,
}

TOOL_DEFINITIONS = [
    ...
    {
        "type": "function",
        "function": {
            "name": "my_tool",
            "description": "What this tool does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg": {"type": "string", "description": "The input argument."}
                },
                "required": ["arg"],
            },
        },
    },
]
```
