import datetime
import math
import os

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from pypdf import PdfReader

# Max characters returned per tool to keep context manageable
_WEBPAGE_LIMIT = 6000
_PDF_LIMIT = 8000


def get_current_datetime() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["__builtins__"] = {}
        result = eval(expression, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def read_file(path: str) -> str:
    try:
        with open(os.path.expanduser(path)) as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


def list_directory(path: str = ".") -> str:
    try:
        entries = os.listdir(os.path.expanduser(path))
        return "\n".join(sorted(entries))
    except Exception as e:
        return f"Error: {e}"


def web_search(query: str, num_results: int = 6) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n   URL: {r['href']}\n   {r['body']}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def fetch_webpage(url: str) -> str:
    try:
        response = httpx.get(
            url,
            follow_redirects=True,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; research-agent/1.0)"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Collapse excessive blank lines
        lines = [l for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)
        if len(text) > _WEBPAGE_LIMIT:
            text = text[:_WEBPAGE_LIMIT] + f"\n\n[truncated — {len(text)} total chars]"
        return text
    except Exception as e:
        return f"Error: {e}"


def parse_pdf(path: str) -> str:
    try:
        reader = PdfReader(os.path.expanduser(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
        if len(text) > _PDF_LIMIT:
            text = text[:_PDF_LIMIT] + f"\n\n[truncated — {len(text)} total chars]"
        return text
    except Exception as e:
        return f"Error: {e}"


TOOL_FUNCTIONS = {
    "get_current_datetime": get_current_datetime,
    "calculate": calculate,
    "read_file": read_file,
    "list_directory": list_directory,
    "web_search": web_search,
    "fetch_webpage": fetch_webpage,
    "parse_pdf": parse_pdf,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Supports standard math functions like sqrt, sin, cos, log, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '2 ** 10' or 'sqrt(144)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the contents of a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the files and folders in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list. Defaults to current directory.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using DuckDuckGo. Returns titles, URLs, and snippets "
                "for the top results. Use this to find current information or discover sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 6, max 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": (
                "Fetch a webpage by URL and return its readable text content. "
                "Use this after web_search to read the full content of a promising result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the page to fetch.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parse_pdf",
            "description": "Extract and return the text content of a local PDF file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the PDF file.",
                    }
                },
                "required": ["path"],
            },
        },
    },
]
