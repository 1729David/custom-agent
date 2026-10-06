"""Stock-news pipeline, split into small stages so a job can report progress.

Stages: list_articles -> fetch_article -> check_relevance -> summarize_article
-> build_report. run_news_job drives them one article at a time through a job
context, which persists each article's state so an interrupted job can resume.
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone

import httpx
import ollama
from bs4 import BeautifulSoup
from ddgs import DDGS

NEWS_MODEL = os.environ.get("NEWS_MODEL", "qwen3.6:35b")
RELEVANCE_MODEL = os.environ.get("NEWS_RELEVANCE_MODEL", NEWS_MODEL)
SUMMARY_MODEL = os.environ.get("NEWS_SUMMARY_MODEL", NEWS_MODEL)

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-agent/1.0)"}
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_ARTICLE_TEXT_LIMIT = 8000
_MIN_TEXT_CHARS = 200


class ArticleSkipped(Exception):
    """An article was rejected for an expected reason (carried as the message)."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(dt: datetime | None) -> str | None:
    return dt.strftime(_TS_FORMAT) if dt else None


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


def _chat(model: str, prompt: str) -> str:
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"].strip()


# --- Stage 1: list -----------------------------------------------------------


def list_articles(ticker: str, max_articles: int = 15) -> list[dict]:
    """Candidate articles from a news search: url, title, source, search_date, snippet.

    Returns up to twice max_articles, since some will fail later stages.
    """
    return [
        {
            "url": r["url"],
            "title": r.get("title"),
            "source": r.get("source"),
            "search_date": r.get("date"),
            "snippet": r.get("body"),
        }
        for r in _search(ticker.strip().upper(), int(max_articles) * 2)
    ]


# --- Stage 2: fetch ----------------------------------------------------------


def fetch_article(candidate: dict, cutoff: datetime) -> dict:
    """Download one article and return its plain text and dates.

    Raises ArticleSkipped if it can't be downloaded, is older than cutoff, or has
    too little text.
    """
    url = candidate["url"]
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=15, headers=_HEADERS)
        resp.raise_for_status()
    except Exception as e:
        raise ArticleSkipped(f"download failed: {e}") from e

    retrieved = _now()
    soup = BeautifulSoup(resp.text, "html.parser")
    published, pub_source = _extract_published(soup, candidate.get("search_date"))
    if published and published < cutoff:
        raise ArticleSkipped("older than window")

    text = _page_text(resp.text)
    if len(text) < _MIN_TEXT_CHARS:
        text = candidate.get("snippet") or text
    if len(text) < _MIN_TEXT_CHARS:
        raise ArticleSkipped("not enough article text")

    title = candidate.get("title") or (soup.title.string if soup.title else None)
    return {
        "title": title,
        "text": text,
        "published_at": _stamp(published),
        "published_at_source": pub_source,
        "retrieved_at": _stamp(retrieved),
    }


# --- Stage 3: relevance ------------------------------------------------------


def check_relevance(
    ticker: str, title: str | None, text: str, model: str | None = None
) -> bool:
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
        answer = _chat(model or RELEVANCE_MODEL, prompt)
    except Exception:
        return True
    return answer.upper().lstrip("*`\" ").startswith("YES")


# --- Stage 4: summarize ------------------------------------------------------


def summarize_article(
    ticker: str, title: str | None, text: str, model: str | None = None
) -> str:
    return _chat(
        model or SUMMARY_MODEL,
        f"Summarize this news article about {ticker} in 3-4 sentences. "
        "Focus on facts relevant to the stock (results, guidance, "
        "products, analyst actions, risks). No preamble.\n\n"
        f"Title: {title}\n\n{text[:_ARTICLE_TEXT_LIMIT]}",
    )


# --- Stage 5: bundle ---------------------------------------------------------

_REPORT_FIELDS = ("title", "source", "url", "published_at", "retrieved_at", "summary")


def build_report(ticker: str, articles: list[dict], model: str | None = None) -> dict:
    """Combine per-article summaries into the final response, with an overview."""
    ordered = sorted(articles, key=lambda a: a.get("published_at") or "")
    digest = "\n\n".join(
        f"[{a['published_at'] or 'date unknown'}] {a['title']} ({a['source']})\n{a['summary']}"
        for a in ordered
    )
    overview = _chat(
        model or SUMMARY_MODEL,
        f"Below are per-article summaries of the past month of news on {ticker}, "
        "oldest first. Write a concise overview: key themes, notable events "
        "with their dates, and overall tone. Only use facts from the summaries.\n\n"
        + digest,
    )
    return {
        "ticker": ticker,
        "generated_at": _stamp(_now()),
        "article_count": len(ordered),
        "overview": overview,
        "articles": [{k: a[k] for k in _REPORT_FIELDS} for a in ordered],
    }


# --- Job driver --------------------------------------------------------------


def run_news_job(ctx, ticker: str, days: int = 30, max_articles: int = 15) -> dict:
    """Run the pipeline one article at a time, persisting state through ctx.

    ctx provides articles(), add_articles(rows), update_article(url, **fields) and
    checkpoint(), which raises to pause or cancel the job between steps.
    Article status advances pending -> fetched -> relevant -> done (or skipped), so
    calling this again on the same ctx resumes where the previous run stopped.
    Raises LookupError if no usable articles were found.
    """
    ticker = ticker.strip().upper()
    cutoff = _now() - timedelta(days=int(days))

    ctx.checkpoint()
    if not ctx.articles():
        candidates = list_articles(ticker, max_articles)
        if not candidates:
            raise LookupError(f"No news found for {ticker}.")
        ctx.add_articles(candidates)

    articles = ctx.articles()
    kept = sum(a["status"] in ("relevant", "done") for a in articles)

    for art in articles:
        url = art["url"]

        def advance(**fields):
            art.update(fields)
            ctx.update_article(url, **fields)

        if art["status"] in ("done", "skipped"):
            continue
        if art["status"] in ("pending", "fetched") and kept >= int(max_articles):
            advance(status="skipped", reason="max articles reached")
            continue

        if art["status"] == "pending":
            ctx.checkpoint()
            try:
                info = fetch_article(art, cutoff)
            except ArticleSkipped as e:
                advance(status="skipped", reason=str(e))
                continue
            advance(status="fetched", **info)

        if art["status"] == "fetched":
            ctx.checkpoint()
            if not check_relevance(ticker, art["title"], art["text"]):
                advance(status="skipped", reason=f"not primarily about {ticker}")
                continue
            kept += 1
            advance(status="relevant")

        if art["status"] == "relevant":
            ctx.checkpoint()
            summary = summarize_article(ticker, art["title"], art["text"])
            advance(status="done", summary=summary)

    ctx.checkpoint()
    done = [a for a in ctx.articles() if a["status"] == "done"]
    if not done:
        raise LookupError(f"No usable news found for {ticker}.")
    report = build_report(ticker, done)
    report["skipped"] = [
        {"url": a["url"], "title": a["title"], "reason": a["reason"]}
        for a in ctx.articles()
        if a["status"] == "skipped"
    ]
    return report
