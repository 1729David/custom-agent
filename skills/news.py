import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from news_pipeline import (
    _ARTICLE_TEXT_LIMIT,
    _HEADERS,
    _TS_FORMAT,
    NEWS_MODEL,
    _chat,
    _extract_published,
    _now,
    _page_text,
    _search,
    check_relevance,
)

NAME = "news"
DESCRIPTION = "Archive the past month of news for a stock ticker, then summarize it"

NEWS_DIR = Path(os.path.expanduser(os.environ.get("NEWS_DIR", "~/.custom-agent/news")))

SYSTEM_PROMPT_ADDITION = (
    "You are archiving and summarizing recent news about a public company's stock. "
    "Use fetch_stock_news to download the articles, then summarize_stock_news to "
    "summarize them. Always report both the published date and the retrieval date."
)

WORKFLOW_PROMPT = """\
Gather and summarize the past month of news for the stock ticker: {query}

Step 1 — Fetch: call fetch_stock_news with the ticker. This saves each article's HTML and a screenshot, and records when it was retrieved and when it was published.

Step 2 — Summarize: call summarize_stock_news with the same ticker.

Step 3 — Report: show the overview returned by summarize_stock_news, then list each article with its title, source, published date, retrieval date, and the folder where the original is stored. Mention any articles that had no screenshot or no published date."""


def _screenshot(page, url: str, path: Path) -> bool:
    try:
        page.goto(url, timeout=20_000, wait_until="domcontentloaded")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(path), full_page=True)
        return True
    except Exception:
        return False


def fetch_news(ticker: str, days: int = 30, max_articles: int = 15) -> dict:
    """Search for and archive recent articles. Returns {"ticker", "saved", "skipped"}.

    Each saved entry carries an "id" (its archive folder name) so a later
    summarize_news call can be scoped to exactly these articles.
    """
    ticker = ticker.strip().upper()
    cutoff = _now() - timedelta(days=int(days))
    found = _search(ticker, int(max_articles) * 2)
    if not found:
        return {"ticker": ticker, "saved": [], "skipped": []}

    folder = NEWS_DIR / ticker
    folder.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright

        pw = sync_playwright().start()
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
    except Exception:
        pw = browser = page = None

    saved, skipped = [], []
    try:
        for r in found:
            if len(saved) >= int(max_articles):
                break
            url = r["url"]
            try:
                resp = httpx.get(
                    url, follow_redirects=True, timeout=15, headers=_HEADERS
                )
                resp.raise_for_status()
            except Exception as e:
                skipped.append({"url": url, "reason": f"download failed: {e}"})
                continue

            retrieved = _now()
            soup = BeautifulSoup(resp.text, "html.parser")
            published, pub_source = _extract_published(soup, r.get("date"))
            if published and published < cutoff:
                skipped.append({"url": url, "reason": "older than window"})
                continue

            title = r.get("title") or (soup.title.string if soup.title else "")
            text = _page_text(resp.text)
            if len(text) < 200:
                text = r.get("body") or text
            if not check_relevance(ticker, title, text):
                skipped.append({"url": url, "title": title, "reason": f"not primarily about {ticker}"})
                continue

            slug = hashlib.sha1(url.encode()).hexdigest()[:10]
            stamp = (published or retrieved).strftime("%Y%m%d")
            art_dir = folder / f"{stamp}_{slug}"
            art_dir.mkdir(exist_ok=True)
            (art_dir / "article.html").write_bytes(resp.content)

            shot_ok = page is not None and _screenshot(
                page, url, art_dir / "screenshot.png"
            )
            meta = {
                "url": url,
                "title": title,
                "source": r.get("source"),
                "ticker": ticker,
                "retrieved_at": retrieved.strftime(_TS_FORMAT),
                "published_at": published.strftime(_TS_FORMAT) if published else None,
                "published_at_source": pub_source,
                "screenshot": "screenshot.png" if shot_ok else None,
                "summary": None,
            }
            (art_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            saved.append({**meta, "id": art_dir.name, "path": str(art_dir)})
    finally:
        if browser:
            browser.close()
        if pw:
            pw.stop()

    return {"ticker": ticker, "saved": saved, "skipped": skipped}


def fetch_stock_news(ticker: str, days: int = 30, max_articles: int = 15) -> str:
    try:
        result = fetch_news(ticker, days, max_articles)
        if not result["saved"] and not result["skipped"]:
            return f"No news found for {result['ticker']}."
        return json.dumps(
            {"saved": result["saved"], "skipped": result["skipped"]}, indent=2
        )
    except Exception as e:
        return f"Error: {e}"


def summarize_news(
    ticker: str, model: str | None = None, article_ids: list[str] | None = None
) -> dict:
    """Summarize archived articles. Raises LookupError if there are none.

    article_ids restricts the summary to specific archive folders (the "id"
    values returned by fetch_news); None means everything archived for the ticker.
    """
    ticker = ticker.strip().upper()
    model = model or NEWS_MODEL
    metas = sorted(
        (
            p
            for p in (NEWS_DIR / ticker).glob("*/meta.json")
            if article_ids is None or p.parent.name in article_ids
        ),
        key=lambda p: json.loads(p.read_text()).get("published_at") or "",
    )
    if not metas:
        raise LookupError(f"No archived articles for {ticker}.")

    entries = []
    for path in metas:
        meta = json.loads(path.read_text())
        if not meta.get("summary"):
            text = _page_text((path.parent / "article.html").read_text(errors="replace"))
            if len(text) < 200:
                meta["summary"] = "(not enough article text to summarize)"
            else:
                meta["summary"] = _chat(
                    model,
                    f"Summarize this news article about {ticker} in 3-4 sentences. "
                    "Focus on facts relevant to the stock (results, guidance, "
                    "products, analyst actions, risks). No preamble.\n\n"
                    f"Title: {meta['title']}\n\n{text[:_ARTICLE_TEXT_LIMIT]}",
                )
            path.write_text(json.dumps(meta, indent=2))
        entries.append(meta)

    digest = "\n\n".join(
        f"[{m['published_at'] or 'date unknown'}] {m['title']} ({m['source']})\n{m['summary']}"
        for m in entries
    )
    overview = _chat(
        model,
        f"Below are per-article summaries of the past month of news on {ticker}, "
        "oldest first. Write a concise overview: key themes, notable events "
        "with their dates, and overall tone. Only use facts from the summaries.\n\n"
        + digest,
    )
    return {
        "ticker": ticker,
        "generated_at": _now().strftime(_TS_FORMAT),
        "article_count": len(entries),
        "overview": overview,
        "articles": [
            {k: m[k] for k in ("title", "source", "url", "published_at", "retrieved_at", "summary")}
            for m in entries
        ],
    }


def summarize_stock_news(ticker: str, model: str | None = None) -> str:
    try:
        return json.dumps(summarize_news(ticker, model), indent=2)
    except LookupError:
        return f"No archived articles for {ticker.strip().upper()}. Run fetch_stock_news first."
    except Exception as e:
        return f"Error: {e}"


TOOL_FUNCTIONS = {
    "fetch_stock_news": fetch_stock_news,
    "summarize_stock_news": summarize_stock_news,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_stock_news",
            "description": (
                "Search for news about a stock ticker from the past month and archive each "
                "article (original HTML, screenshot, and metadata with published and "
                "retrieval dates) under ~/.custom-agent/news/<TICKER>/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
                    "days": {"type": "integer", "description": "Lookback window in days (default 30)"},
                    "max_articles": {"type": "integer", "description": "Maximum articles to archive (default 15)"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_stock_news",
            "description": (
                "Summarize the archived news articles for a ticker: one summary per article "
                "(stored with the article) plus a combined dated overview."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
                    "model": {"type": "string", "description": "Optional Ollama model override"},
                },
                "required": ["ticker"],
            },
        },
    },
]
