from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import news_pipeline as np
from jobs import JobStore

_TS = "%Y-%m-%dT%H:%M:%SZ"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "jobs.db"


@pytest.fixture
def make_store(db_path):
    """Build JobStores on a temp DB; started ones are stopped after the test."""
    made = []

    def make(runners=None, path=None, start=False):
        store = JobStore(runners or {}, path or db_path)
        if start:
            store.start()
        made.append(store)
        return store

    yield make
    for store in made:
        store.stop()


@pytest.fixture
def sleeps(monkeypatch):
    """Replace news_pipeline's sleep so retries are instant; records the delays."""
    recorded = []
    monkeypatch.setattr(np, "time", SimpleNamespace(sleep=recorded.append))
    return recorded


class World:
    """Stand-in for the search engine, web pages and Ollama, with call logs."""

    def __init__(self):
        self.candidates = []
        self.pages = {}  # url -> info dict, or an exception to raise
        self.irrelevant = set()  # (ticker, title) pairs the model answers NO to
        self.summary_error = None
        self.calls = {"list": 0, "fetch": [], "relevance": [], "summary": []}

    def url(self, name):
        return f"http://example.com/{name}"

    def add(self, name, days_old=3, text=None, error=None, published=True, title=None):
        url = self.url(name)
        title = title or f"Title {name}"
        if text is None:  # distinct per article, so unrelated articles aren't "copies"
            text = " ".join(f"{name}{i}" for i in range(80))
        self.candidates.append(
            {"url": url, "title": title, "source": "Src",
             "search_date": None, "snippet": None}
        )
        if error:
            self.pages[url] = error
        else:
            when = datetime.now(timezone.utc) - timedelta(days=days_old)
            self.pages[url] = {
                "title": title,
                "text": text,
                "published_at": when.strftime(_TS) if published else None,
                "published_at_source": "meta_tag" if published else None,
                "retrieved_at": datetime.now(timezone.utc).strftime(_TS),
            }
        return url

    # patched-in implementations
    def list_articles(self, ticker, max_articles=15):
        self.calls["list"] += 1
        return [dict(c) for c in self.candidates[: int(max_articles) * 2]]

    def fetch_article(self, candidate):
        self.calls["fetch"].append(candidate["url"])
        page = self.pages[candidate["url"]]
        if isinstance(page, Exception):
            raise page
        return dict(page)

    def check_relevance(self, ticker, title, text, model=None, fail_open=True):
        self.calls["relevance"].append((ticker, title))
        return (ticker, title) not in self.irrelevant

    def summarize_article(self, ticker, title, text, model=None):
        self.calls["summary"].append((ticker, title))
        if self.summary_error:
            raise self.summary_error
        return f"{ticker} summary of {title}"


@pytest.fixture
def world(monkeypatch):
    w = World()
    monkeypatch.setattr(np, "list_articles", w.list_articles)
    monkeypatch.setattr(np, "fetch_article", w.fetch_article)
    monkeypatch.setattr(np, "check_relevance", w.check_relevance)
    monkeypatch.setattr(np, "summarize_article", w.summarize_article)
    monkeypatch.setattr(np, "_chat", lambda model, prompt: "OVERVIEW")
    return w


@pytest.fixture
def pipeline_store(make_store):
    """A JobStore whose "news" runner is the real pipeline (worker not started)."""
    return make_store({"news": np.run_news_job})
