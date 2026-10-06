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
