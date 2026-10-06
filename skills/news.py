import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import ollama
from bs4 import BeautifulSoup
from ddgs import DDGS

NAME = "news"
DESCRIPTION = "Archive the past month of news for a stock ticker, then summarize it"

NEWS_MODEL = "qwen3.6:35b"
NEWS_DIR = Path(os.path.expanduser("~/.custom-agent/news"))
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-agent/1.0)"}
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_ARTICLE_TEXT_LIMIT = 8000

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


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _json_ld_dates(soup: BeautifulSoup):
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except ValueError:
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                if node.get("datePublished"):
                    yield node["datePublished"]
                stack.extend(node.get("@graph", []))


def _extract_published(soup: BeautifulSoup, fallback: str | None):
    """Return (datetime, source) using the most reliable date found in the page."""
    for attrs in (
        {"property": "article:published_time"},
        {"name": "article:published_time"},
        {"itemprop": "datePublished"},
        {"name": "date"},
    ):
        tag = soup.find("meta", attrs=attrs)
        dt = _parse_date(tag.get("content") if tag else None)
        if dt:
            return dt, "meta_tag"
    for value in _json_ld_dates(soup):
        dt = _parse_date(value)
        if dt:
            return dt, "json_ld"
    time_tag = soup.find("time", attrs={"datetime": True})
    dt = _parse_date(time_tag["datetime"] if time_tag else None)
    if dt:
        return dt, "time_tag"
    dt = _parse_date(fallback)
    if dt:
        return dt, "search_result"
    return None, None


def _page_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))


def _screenshot(page, url: str, path: Path) -> bool:
    try:
        page.goto(url, timeout=20_000, wait_until="domcontentloaded")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(path), full_page=True)
        return True
    except Exception:
        return False


def _search(ticker: str, max_results: int) -> list[dict]:
    seen: dict[str, dict] = {}
    for query in (f"{ticker} stock", f"{ticker} shares earnings"):
        try:
            results = DDGS().news(query, timelimit="m", max_results=max_results)
        except Exception:
            continue
        for r in results:
            url = r.get("url")
            if url and url not in seen:
                seen[url] = r
    return list(seen.values())


def _is_about_ticker(ticker: str, title: str, text: str, model: str) -> bool:
    """Ask the model whether the article is primarily about the ticker's company.

    Fails open (returns True) if the model can't be reached, so an Ollama outage
    doesn't silently discard every article.
    """
    prompt = (
        f"Stock ticker: {ticker}\n\n"
        f"Title: {title}\n\n{text[:3000]}\n\n"
        f"Is this article primarily about the company with ticker {ticker}? "
        "Answer NO if it is mainly about a different company, or only mentions "
        f"{ticker} in passing (e.g. in a list of stocks or a comparison). "
        "Reply with exactly one word: YES or NO."
    )
    try:
        answer = _chat(model, prompt)
    except Exception:
        return True
    return answer.upper().lstrip("*`\" ").startswith("YES")


def fetch_stock_news(ticker: str, days: int = 30, max_articles: int = 15) -> str:
    try:
        ticker = ticker.strip().upper()
        cutoff = _now() - timedelta(days=int(days))
        found = _search(ticker, int(max_articles) * 2)
        if not found:
            return f"No news found for {ticker}."

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
                if not _is_about_ticker(ticker, title, text, NEWS_MODEL):
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
                saved.append({**meta, "path": str(art_dir)})
        finally:
            if browser:
                browser.close()
            if pw:
                pw.stop()

        return json.dumps({"saved": saved, "skipped": skipped}, indent=2)
    except Exception as e:
        return f"Error: {e}"


def _chat(model: str, prompt: str) -> str:
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"].strip()


def summarize_stock_news(ticker: str, model: str | None = None) -> str:
    try:
        ticker = ticker.strip().upper()
        model = model or NEWS_MODEL
        metas = sorted(
            (NEWS_DIR / ticker).glob("*/meta.json"),
            key=lambda p: json.loads(p.read_text()).get("published_at") or "",
        )
        if not metas:
            return f"No archived articles for {ticker}. Run fetch_stock_news first."

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
        return json.dumps(
            {
                "ticker": ticker,
                "article_count": len(entries),
                "overview": overview,
                "articles": [
                    {k: m[k] for k in ("title", "source", "url", "published_at", "retrieved_at", "summary")}
                    for m in entries
                ],
            },
            indent=2,
        )
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
