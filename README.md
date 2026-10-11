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
| `POST /news/{ticker}` | Queue a fetch-and-summarize job. Optional JSON body: `days` (default 30), `max_articles` (default 15), `refresh` (default `false`; see below). Returns `202` with the job; an identical queued/running job is reused (`200`) |
| `GET /jobs/{id}` | Job status (`queued`, `running`, `paused`, `done`, `failed`), a `progress` block, and, when done, the result |
| `GET /jobs?status=&limit=` | Job history, newest first |
| `POST /jobs/{id}/pause` | Pause a job. A queued job pauses at once; a running one at its next step (`stage` shows `pausing` meanwhile) |
| `POST /jobs/{id}/resume` | Re-queue a paused job; it continues where it left off |
| `POST /jobs/{id}/cancel` | Fail a job with error `cancelled by user` (a running job at its next step, `stage` `cancelling`) |

Send the key as an `X-API-Key` header. If `API_KEY` is unset the API is unauthenticated (a warning is logged), so keep it bound to `127.0.0.1`.

**How a job runs** (`news_pipeline.py`): list candidate articles from a news search (at most 2× `max_articles`), then for each one in turn fetch it (plain text only; no HTML or screenshots are kept), check it is primarily about the ticker, and summarize it; finally bundle the summaries into an overview. Each candidate's state is stored in `job_articles` (`pending` → `fetched` → `relevant` → `done`, or `skipped` with a `reason`). Jobs run one at a time, and a job interrupted by a restart is re-queued and resumes where it left off (paused jobs stay paused). Pause and cancel take effect between steps, so an article already being fetched or summarized finishes first. Pause/resume/cancel on a job in the wrong state returns `409`; a new request identical to a paused job returns that job.

**Relevance.** Before an article is summarized, the model is asked to name the one company or topic it is mainly about and then answer YES or NO to whether that is the requested ticker's company, with explicit rules that market wraps, premarket roundups, stock lists and passing mentions are NO. It runs at temperature 0 so a stored verdict is repeatable. On 25 labeled negatives (market wraps, roundups, articles about other companies) and 44 labeled Apple articles, the earlier one-word prompt let two roundups through (92% specificity), while this one let none through, at the same speed and with no Apple article lost; sending twice as much article text did not help. Verdicts are stored per article and ticker, so stored verdicts from before this change are kept until the article is refreshed (`"refresh": true`).

**Duplicates.** After an article is fetched and before any model call, it is compared with the articles the job has already accepted. If it is a copy (a syndicated or excerpted version of the same story), it is skipped with the reason `duplicate of article <id>`, where the id is the copy that was kept; the first one in search order wins. Two articles count as copies when at least 70% of the shorter one's 5-word phrases appear in the other, or at least 25% if their titles also match. Different stories on the same topic, or from one outlet sharing site boilerplate, are kept. Reworded versions of a story are not caught.

**Articles are shared across jobs.** Jobs and articles are many-to-many: `articles` holds each retrieval under its own `article_id` (URL, title, source, `published_at`, `retrieved_at`, plain text), `article_analysis` holds the per-ticker relevance verdict and summary, and `job_articles` links a job to the articles it considered. A job still searches for candidates every time, but a candidate whose URL (ignoring tracking parameters and fragments) was already retrieved is reused with its stored text, verdict and summary, so only new articles are downloaded and sent to the model. A reused article keeps its original `retrieved_at` and `article_id`, so the caller can recognize it. The `days` window is applied per job to the article's published date, so a stored article may be skipped by one job and used by another.

With `"refresh": true` every candidate is downloaded again and stored as a new article entry (new `article_id`; earlier entries and earlier jobs' results are untouched). A later non-refresh job reuses the latest entry for each URL. Download failures are never stored, so they are retried next time.

**Speed and model choice.** Qwen3-family models "think" before answering unless told not to, which generates thousands of hidden tokens; the pipeline turns thinking off (`NEWS_LLM_THINK=1` re-enables it). That alone cut a 15-article AAPL job from about 15 minutes to under 3, with summaries of the same length and no unsupported figures. Smaller models did not help in a benchmark on stored articles (thinking off, per article for relevance plus summary): `qwen3.6:35b` took 8.2s because it is a mixture-of-experts model, `qwen3.5:latest` (9.7B, dense) took 14.6s and was less accurate (it accepted an off-topic article and invented a figure), and using the small model for relevance and the large one for summaries took 21.4s because the two do not fit in memory together and reload every article. Keep one model for both steps.

**Ollama failures.** A dropped connection, timeout or 5xx from Ollama is retried after 5, 20 and 60 seconds. If it is still failing, the job is **paused** (not failed) with an `error` explaining why; all progress is already saved, so `POST /jobs/{id}/resume` continues it once Ollama is back. Other errors, such as an unknown model, fail the job. In jobs, a relevance check never guesses when Ollama is down, so no unverified verdict is ever stored.

An older database is migrated automatically on startup (a `.bak` copy is written first).

`progress` in `GET /jobs/{id}` has the current `stage` (`queued`, `listing`, `processing`, `bundling`, `pausing`, `cancelling`, `paused`, `done`, `failed`), the article `total`, per-status `counts`, and each article's `status`, `reason`, `article_id` and `reused`. `target` is the number of articles the job aims to keep (`max_articles`), while `total` is the larger candidate count.

| Env var | Default | Purpose |
|---------|---------|---------|
| `API_KEY` | unset | Required `X-API-Key` value |
| `JOBS_DB` | `~/.custom-agent/jobs.db` | SQLite job history |
| `NEWS_MODEL` | `qwen3.6:35b` | Ollama model for everything below |
| `NEWS_RELEVANCE_MODEL` | `NEWS_MODEL` | Model for the relevance check (a smaller one works) |
| `NEWS_SUMMARY_MODEL` | `NEWS_MODEL` | Model for article summaries and the overview |
| `NEWS_LLM_TIMEOUT` | `300` | Seconds to wait for one Ollama response |
| `NEWS_LLM_THINK` | off | Set to `1` to enable the model's thinking mode (see below) |

A finished news job's `result` has `ticker`, `generated_at`, `overview`, and `articles` (each with `article_id`, `title`, `source`, `url`, `published_at`, `retrieved_at`, `summary`), plus `skipped`.

```bash
curl -X POST -H "X-API-Key: changeme" localhost:8000/news/AAPL
curl -H "X-API-Key: changeme" localhost:8000/jobs/<id>
```

## Tests

```bash
uv run pytest
```

The suite covers the REST API, job store, article cache, migration, news pipeline and the `/news` skill. It runs in about two seconds and makes no network calls: Ollama, web search and page downloads are stubbed, and every test uses a temporary database (it never touches `~/.custom-agent/`). It doesn't cover the older tools, the research skill or the SEC 10-K skill.

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
