import importlib.util
import json
from pathlib import Path

import pytest

from skills import load_skills

SKILL_PATH = Path(__file__).parents[1] / "skills" / "news.py"


@pytest.fixture
def news(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("news_skill_under_test", SKILL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "NEWS_DIR", tmp_path / "news")
    return mod


def archive(news, ticker, name, published, text="word " * 80, summary=None, title=None):
    folder = news.NEWS_DIR / ticker / name
    folder.mkdir(parents=True)
    (folder / "article.html").write_text(f"<html><body><p>{text}</p></body></html>")
    meta = {"url": f"http://x/{name}", "title": title or f"Title {name}", "source": "Src",
            "ticker": ticker, "retrieved_at": "2026-10-01T00:00:00Z",
            "published_at": published, "summary": summary}
    (folder / "meta.json").write_text(json.dumps(meta))
    return folder


# --- registration ------------------------------------------------------------


def test_skill_is_registered_with_matching_tools():
    skill = load_skills(str(SKILL_PATH.parent))["news"]
    assert skill.description
    defined = {d["function"]["name"] for d in skill.tool_definitions}
    assert defined == set(skill.tool_functions) == {"fetch_stock_news", "summarize_stock_news"}


# --- summarize_news ----------------------------------------------------------


def test_summarize_all_orders_by_date_and_caches_summaries(news, monkeypatch):
    archive(news, "AAPL", "b", "2026-09-20T00:00:00Z")
    archive(news, "AAPL", "a", "2026-09-10T00:00:00Z")
    prompts = []
    monkeypatch.setattr(news, "_chat", lambda model, prompt: prompts.append(prompt) or "SUMMARY")

    report = news.summarize_news("aapl")

    assert report["ticker"] == "AAPL" and report["article_count"] == 2
    assert [a["title"] for a in report["articles"]] == ["Title a", "Title b"]
    assert all(a["summary"] == "SUMMARY" for a in report["articles"])
    assert report["overview"] == "SUMMARY" and report["generated_at"].endswith("Z")
    assert len(prompts) == 3  # two articles and the overview

    prompts.clear()
    news.summarize_news("AAPL")
    assert len(prompts) == 1  # summaries were saved; only the overview is redone
    meta = json.loads((news.NEWS_DIR / "AAPL" / "a" / "meta.json").read_text())
    assert meta["summary"] == "SUMMARY"


def test_summarize_can_be_scoped_to_article_ids(news, monkeypatch):
    for name, day in (("a", "10"), ("b", "11"), ("c", "12")):
        archive(news, "AAPL", name, f"2026-09-{day}T00:00:00Z")
    monkeypatch.setattr(news, "_chat", lambda model, prompt: "S")

    scoped = news.summarize_news("AAPL", article_ids=["a", "c"])
    assert [a["title"] for a in scoped["articles"]] == ["Title a", "Title c"]
    assert news.summarize_news("AAPL")["article_count"] == 3  # None means all


def test_articles_with_too_little_text_get_a_placeholder(news, monkeypatch):
    archive(news, "AAPL", "a", "2026-09-10T00:00:00Z", text="tiny")
    monkeypatch.setattr(news, "_chat", lambda model, prompt: "S")
    report = news.summarize_news("AAPL")
    assert report["articles"][0]["summary"] == "(not enough article text to summarize)"


def test_summarize_with_nothing_archived_raises_lookup_error(news):
    with pytest.raises(LookupError, match="No archived articles for ZZZ"):
        news.summarize_news("zzz")


# --- tool wrappers -----------------------------------------------------------


def test_summarize_tool_returns_json_or_a_readable_message(news, monkeypatch):
    archive(news, "AAPL", "a", "2026-09-10T00:00:00Z")
    monkeypatch.setattr(news, "_chat", lambda model, prompt: "S")
    assert json.loads(news.summarize_stock_news("AAPL"))["article_count"] == 1

    message = news.summarize_stock_news("zzz")
    assert "No archived articles for ZZZ" in message and "fetch_stock_news" in message

    monkeypatch.setattr(news, "summarize_news", lambda *a, **k: 1 / 0)
    assert news.summarize_stock_news("AAPL").startswith("Error: ")


def test_fetch_tool_wraps_results_and_errors(news, monkeypatch):
    monkeypatch.setattr(news, "fetch_news", lambda *a: {"ticker": "AAPL", "saved": [], "skipped": []})
    assert news.fetch_stock_news("aapl") == "No news found for AAPL."

    result = {"ticker": "AAPL", "saved": [{"id": "x"}], "skipped": [{"url": "u", "reason": "r"}]}
    monkeypatch.setattr(news, "fetch_news", lambda *a: result)
    assert json.loads(news.fetch_stock_news("AAPL")) == {
        "saved": [{"id": "x"}], "skipped": [{"url": "u", "reason": "r"}]}

    def boom(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(news, "fetch_news", boom)
    assert news.fetch_stock_news("AAPL") == "Error: boom"


# --- fetch_news --------------------------------------------------------------


class FakeResponse:
    def __init__(self, html, status=200):
        self.text, self.content, self.status_code = html, html.encode(), status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def page(published, body="word " * 80):
    return (f'<html><head><meta property="article:published_time" content="{published}">'
            f"</head><body><p>{body}</p></body></html>")


def test_fetch_news_archives_articles_and_reports_skips(news, monkeypatch):
    from datetime import datetime, timedelta, timezone

    def days_ago(n):
        return (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%SZ")

    pages = {
        "http://x/good": FakeResponse(page(days_ago(2))),
        "http://x/old": FakeResponse(page(days_ago(60))),
        "http://x/offtopic": FakeResponse(page(days_ago(3))),
        "http://x/forbidden": FakeResponse("no", status=403),
    }
    monkeypatch.setattr(news, "_search", lambda ticker, n: [
        {"url": u, "title": u.rsplit("/", 1)[1], "source": "Src"} for u in pages])
    monkeypatch.setattr(news.httpx, "get", lambda url, **kw: pages[url])
    monkeypatch.setattr(news, "check_relevance", lambda t, title, text: title != "offtopic")
    import playwright.sync_api as pw
    monkeypatch.setattr(pw, "sync_playwright", lambda: (_ for _ in ()).throw(RuntimeError("no browser")))

    result = news.fetch_news("aapl")

    assert result["ticker"] == "AAPL"
    assert [s["title"] for s in result["saved"]] == ["good"]
    saved = result["saved"][0]
    folder = news.NEWS_DIR / "AAPL" / saved["id"]
    assert (folder / "article.html").exists() and (folder / "meta.json").exists()
    assert saved["screenshot"] is None and saved["summary"] is None
    assert saved["published_at_source"] == "meta_tag" and saved["retrieved_at"].endswith("Z")
    reasons = {s["url"].rsplit("/", 1)[1]: s["reason"] for s in result["skipped"]}
    assert reasons["old"] == "older than window"
    assert reasons["offtopic"] == "not primarily about AAPL"
    assert reasons["forbidden"].startswith("download failed")


def test_fetch_news_with_no_search_results(news, monkeypatch):
    monkeypatch.setattr(news, "_search", lambda ticker, n: [])
    assert news.fetch_news("aapl") == {"ticker": "AAPL", "saved": [], "skipped": []}
