"""Stock-news pipeline, split into small stages so a job can report progress.

Stages: list_articles -> fetch_article -> check_relevance -> summarize_article
-> build_report. run_news_job drives them one article at a time through a job
context, which persists each article's state so an interrupted job can resume.
"""

import json
import os
import re
import time
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

# Ollama calls are retried on transient failures (dropped connection, timeout, 5xx)
# after these delays in seconds; once they run out the job pauses so it can be resumed.
# Duplicate detection (syndicated copies of one story). Articles are compared by the
# share of the shorter one's 5-word phrases that appear in the other ("containment").
# Calibrated on real articles: a true copy scored 1.00, while two different stories
# from one outlet sharing site boilerplate scored 0.52 and everything else <= 0.45.
# The relevance reply is a short subject line plus YES/NO. Temperature 0 keeps the
# verdict repeatable, since it is stored and reused by later jobs.
_RELEVANCE_MAX_TOKENS = 60
_RELEVANCE_TEMPERATURE = 0
_RELEVANCE_ANSWER = re.compile(r"ANSWER:\s*[*`\"' ]*(YES|NO)", re.IGNORECASE)

# Naming the main subject before answering is what separates a real article about the
# company from a market wrap or a list that merely includes it. Measured on 25 labeled
# negatives and 44 positives: the one-word prompt let two roundups through (92%
# specificity), this one let none through, at the same speed.
_RELEVANCE_PROMPT = """\
You are screening news for a sentiment feed about the company with stock ticker {ticker}. Decide whether this article is primarily about that company, meaning the company (its business, products, executives, stock, earnings, legal matters or analyst views) is the main subject.
Answer NO if any of these apply:
- the article is mainly about another company or topic, and {ticker} is only a comparison, example, supplier, customer or background;
- it is a market wrap, premarket or futures roundup, or a list of several stocks, where {ticker} is one of many;
- {ticker} appears only in the title or a ticker list while the body is about something else.
Answer YES only if you could summarize the article as being about {ticker}'s company.

Title: {title}

{text}

Reply in exactly this format:
MAIN SUBJECT: <the one company or topic the article is mainly about, or 'several companies'>
ANSWER: YES or NO"""

_DUP_SHINGLE_WORDS = 5
_DUP_MAX_WORDS = 3000
_DUP_MIN_SHINGLES = 15  # shorter texts are too small to compare
_DUP_CONTAINMENT = 0.70
_DUP_CONTAINMENT_SAME_TITLE = 0.25  # lower bar when the titles also match

_LLM_TIMEOUT = float(os.environ.get("NEWS_LLM_TIMEOUT", "300"))
# Qwen3-family models "think" by default: thousands of hidden reasoning tokens before
# each answer. Off, a summary takes ~6s instead of ~57s and the summaries are just as
# good (measured on stored articles). Set NEWS_LLM_THINK=1 to turn it back on.
_LLM_THINK = os.environ.get("NEWS_LLM_THINK", "0").lower() in ("1", "true", "yes")
_LLM_RETRY_DELAYS = (5, 20, 60)


class ArticleSkipped(Exception):
    """An article was rejected for an expected reason (carried as the message)."""


class LLMUnavailable(Exception):
    """Ollama kept failing with transient errors after all retries."""


def _is_transient(error: Exception) -> bool:
    if isinstance(error, ollama.ResponseError):
        return error.status_code >= 500
    return isinstance(error, (httpx.TransportError, ConnectionError, TimeoutError))


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


def _chat(
    model: str,
    prompt: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """One Ollama chat call, retrying transient failures.

    max_tokens caps the reply length and temperature fixes the sampling; both are
    ignored when thinking is enabled, since the reasoning tokens would use the budget
    up before any answer and greedy decoding is discouraged for thinking models.
    Raises LLMUnavailable once the retries are used up; other errors (such as an
    unknown model) are not retried and propagate unchanged.
    """
    options = {}
    if not _LLM_THINK:
        if max_tokens:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
    options = options or None
    delays = iter(_LLM_RETRY_DELAYS)
    while True:
        try:
            resp = ollama.Client(timeout=_LLM_TIMEOUT).chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                think=_LLM_THINK,
                options=options,
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            if not _is_transient(e):
                raise
            delay = next(delays, None)
            if delay is None:
                raise LLMUnavailable(f"{type(e).__name__}: {e}") from e
            time.sleep(delay)


# --- Stage 1: list -----------------------------------------------------------


def list_articles(ticker: str, max_articles: int = 15) -> list[dict]:
    """Candidate articles from a news search: url, title, source, search_date, snippet.

    Returns at most twice max_articles, since some will fail later stages.
    """
    limit = int(max_articles) * 2
    return [
        {
            "url": r["url"],
            "title": r.get("title"),
            "source": r.get("source"),
            "search_date": r.get("date"),
            "snippet": r.get("body"),
        }
        for r in _search(ticker.strip().upper(), limit)[:limit]
    ]


# --- Stage 2: fetch ----------------------------------------------------------


def fetch_article(candidate: dict) -> dict:
    """Download one article and return its plain text and dates.

    Raises ArticleSkipped if it can't be downloaded. Whether the article is too
    old or too short is judged separately (see _reject_reason) so that a stored
    article can be re-judged by later jobs with different windows.
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

    text = _page_text(resp.text)
    if len(text) < _MIN_TEXT_CHARS:
        text = candidate.get("snippet") or text

    title = candidate.get("title") or (soup.title.string if soup.title else None)
    return {
        "title": title,
        "text": text,
        "published_at": _stamp(published),
        "published_at_source": pub_source,
        "retrieved_at": _stamp(retrieved),
    }


# --- Duplicate detection -----------------------------------------------------


def _shingles(text: str) -> set[str]:
    words = re.findall(r"\w+", text.lower())[:_DUP_MAX_WORDS]
    n = _DUP_SHINGLE_WORDS
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _title_key(title: str | None) -> str:
    return re.sub(r"\W+", " ", (title or "").lower()).strip()


def _dup_entry(art: dict) -> dict:
    """What find_duplicate needs to remember about an accepted article."""
    return {
        "article_id": art["article_id"],
        "title_key": _title_key(art.get("title")),
        "shingles": _shingles(art.get("text") or ""),
    }


def find_duplicate(art: dict, accepted: list[dict]) -> int | None:
    """The article_id of an accepted article that art is a copy of, or None.

    accepted holds _dup_entry() results. Catches syndicated copies and excerpts of
    the same story; a different story on the same topic is not a duplicate.
    """
    shingles = _shingles(art.get("text") or "")
    title_key = _title_key(art.get("title"))
    for other in accepted:
        smaller = min(len(shingles), len(other["shingles"]))
        if smaller < _DUP_MIN_SHINGLES:
            continue
        containment = len(shingles & other["shingles"]) / smaller
        same_title = bool(title_key) and title_key == other["title_key"]
        if containment >= _DUP_CONTAINMENT or (
            same_title and containment >= _DUP_CONTAINMENT_SAME_TITLE
        ):
            return other["article_id"]
    return None


# --- Stage 3: relevance ------------------------------------------------------


def check_relevance(
    ticker: str,
    title: str | None,
    text: str,
    model: str | None = None,
    fail_open: bool = True,
) -> bool:
    """Ask the model whether the article is primarily about the ticker's company.

    If the model can't be reached this returns True (fail open) so an outage doesn't
    silently discard every article, unless fail_open is false, in which case the
    error propagates; jobs that store the verdict must not record a guess.
    """
    prompt = _RELEVANCE_PROMPT.format(ticker=ticker, title=title, text=text[:3000])
    try:
        answer = _chat(
            model or RELEVANCE_MODEL,
            prompt,
            max_tokens=_RELEVANCE_MAX_TOKENS,
            temperature=_RELEVANCE_TEMPERATURE,
        )
    except Exception:
        if not fail_open:
            raise
        return True
    return _parse_relevance(answer)


def _parse_relevance(answer: str) -> bool:
    """YES/NO from a reply in the requested format, or a bare YES/NO as a fallback."""
    match = _RELEVANCE_ANSWER.search(answer)
    if match:
        return match.group(1).upper() == "YES"
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

_REPORT_FIELDS = (
    "article_id", "title", "source", "url", "published_at", "retrieved_at", "summary",
)


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


def _reject_reason(art: dict, cutoff: datetime) -> str | None:
    """Why a retrieved article can't be used by this job, or None if it can."""
    published = _parse_date(art.get("published_at"))
    if published and published < cutoff:
        return "older than window"
    if len(art.get("text") or "") < _MIN_TEXT_CHARS:
        return "not enough article text"
    return None


def run_news_job(ctx, *args, **kwargs) -> dict:
    """Run the news pipeline; pause the job if Ollama stays unavailable.

    See _run_news_job. When Ollama keeps failing past the retries, all progress is
    already stored per article, so the job is paused (resume it once Ollama is back)
    and not failed.
    """
    try:
        return _run_news_job(ctx, *args, **kwargs)
    except LLMUnavailable as e:
        ctx.pause_job(f"Ollama unavailable, job paused; resume it once Ollama is back ({e})")


def _run_news_job(
    ctx, ticker: str, days: int = 30, max_articles: int = 15, refresh: bool = False
) -> dict:
    """Run the pipeline one article at a time, persisting state through ctx.

    ctx provides articles(), add_articles(rows), update_article(url, **fields),
    cached_article(url), store_article(url, info), store_analysis(article_id, ...),
    checkpoint(), which raises to pause or cancel the job between steps, and
    pause_job(message), which pauses it with an explanation.

    Retrieved articles and their per-ticker analysis are shared across jobs: unless
    refresh is true, a candidate that an earlier job already retrieved is reused
    (its stored text, relevance verdict and summary), and only new articles are
    downloaded and sent to the model. With refresh, every candidate is downloaded
    again and stored as a new article.

    An article that is a copy of one this job already accepted is skipped before any
    model call. The first copy in search order wins; the skip reason names its
    article_id.

    Job-article status advances pending -> fetched -> relevant -> done (or
    skipped), so calling this again on the same ctx resumes where the previous run
    stopped. Raises LookupError if no usable articles were found.
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
    accepted = [_dup_entry(a) for a in articles if a["status"] in ("relevant", "done")]

    for art in articles:
        url = art["url"]

        def advance(status, reason=None, **job_fields):
            art.update(status=status, reason=reason, **job_fields)
            ctx.update_article(url, status=status, reason=reason, **job_fields)

        if art["status"] in ("done", "skipped"):
            continue
        if art["status"] in ("pending", "fetched") and kept >= int(max_articles):
            advance("skipped", "max articles reached")
            continue

        if art["status"] == "pending":
            ctx.checkpoint()
            found = None if refresh else ctx.cached_article(url)
            if found:
                found["reused"] = True
            else:
                try:
                    info = fetch_article(art)
                except ArticleSkipped as e:
                    advance("skipped", str(e))
                    continue
                article_id = ctx.store_article(url, {**info, "source": art.get("source")})
                found = {**info, "article_id": article_id, "relevant": None, "summary": None}
                found["reused"] = False
            art.update(found)
            link = {"article_id": found["article_id"], "reused": found["reused"]}
            reason = _reject_reason(art, cutoff)
            if reason:
                advance("skipped", reason, **link)
                continue
            advance("fetched", **link)

        if art["status"] == "fetched":
            ctx.checkpoint()
            original = find_duplicate(art, accepted)
            if original is not None:
                advance("skipped", f"duplicate of article {original}")
                continue
            if art.get("relevant") is None:
                relevant = check_relevance(
                    ticker, art["title"], art["text"], fail_open=False
                )
                ctx.store_analysis(art["article_id"], relevant=relevant)
                art["relevant"] = relevant
            if not art["relevant"]:
                advance("skipped", f"not primarily about {ticker}")
                continue
            kept += 1
            accepted.append(_dup_entry(art))
            advance("relevant")

        if art["status"] == "relevant":
            ctx.checkpoint()
            if not art.get("summary"):
                art["summary"] = summarize_article(ticker, art["title"], art["text"])
                ctx.store_analysis(art["article_id"], summary=art["summary"])
            advance("done")

    ctx.checkpoint()
    final = ctx.articles()
    done = [a for a in final if a["status"] == "done"]
    if not done:
        raise LookupError(f"No usable news found for {ticker}.")
    report = build_report(ticker, done)
    report["skipped"] = [
        {"url": a["url"], "title": a["title"], "reason": a["reason"]}
        for a in final
        if a["status"] == "skipped"
    ]
    return report
