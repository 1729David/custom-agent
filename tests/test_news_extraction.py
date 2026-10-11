import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from bs4 import BeautifulSoup

import news_pipeline as np


def soup(html):
    return BeautifulSoup(html, "html.parser")


# --- dates -------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026-09-17T17:11:38Z", datetime(2026, 9, 17, 17, 11, 38, tzinfo=timezone.utc)),
        ("2026-09-17T19:11:38+02:00", datetime(2026, 9, 17, 17, 11, 38, tzinfo=timezone.utc)),
        ("2026-09-17", datetime(2026, 9, 17, tzinfo=timezone.utc)),  # naive means UTC
        ("  2026-09-17T00:00:00Z  ", datetime(2026, 9, 17, tzinfo=timezone.utc)),
        ("yesterday", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_date(value, expected):
    assert np._parse_date(value) == expected


def test_meta_tag_is_the_most_trusted_date():
    html = """<meta property="article:published_time" content="2026-09-01T00:00:00Z">
    <script type="application/ld+json">{"datePublished": "2026-09-02T00:00:00Z"}</script>
    <time datetime="2026-09-03T00:00:00Z"></time>"""
    published, source = np._extract_published(soup(html), "2026-09-04T00:00:00Z")
    assert (published.day, source) == (1, "meta_tag")


@pytest.mark.parametrize(
    "attrs", ['name="article:published_time"', 'itemprop="datePublished"', 'name="date"']
)
def test_other_meta_variants(attrs):
    published, source = np._extract_published(
        soup(f'<meta {attrs} content="2026-09-05T00:00:00Z">'), None)
    assert (published.day, source) == (5, "meta_tag")


def test_json_ld_is_used_when_there_is_no_meta_tag():
    html = '<script type="application/ld+json">{"datePublished": "2026-09-02T00:00:00Z"}</script>'
    published, source = np._extract_published(soup(html), None)
    assert (published.day, source) == (2, "json_ld")


def test_json_ld_graph_and_lists_are_searched():
    graph = {"@graph": [{"@type": "WebSite"}, {"@type": "NewsArticle",
                                              "datePublished": "2026-09-06T00:00:00Z"}]}
    html = f'<script type="application/ld+json">{json.dumps([graph])}</script>'
    published, source = np._extract_published(soup(html), None)
    assert (published.day, source) == (6, "json_ld")


def test_broken_json_ld_is_ignored():
    html = '<script type="application/ld+json">{not json</script><time datetime="2026-09-03T00:00:00Z">'
    published, source = np._extract_published(soup(html), None)
    assert (published.day, source) == (3, "time_tag")


def test_search_result_date_is_the_last_resort():
    published, source = np._extract_published(soup("<p>nothing</p>"), "2026-09-04T00:00:00Z")
    assert (published.day, source) == (4, "search_result")
    assert np._extract_published(soup("<p>nothing</p>"), None) == (None, None)
    assert np._extract_published(soup("<p>nothing</p>"), "garbage") == (None, None)


# --- page text ---------------------------------------------------------------


def test_page_text_drops_markup_and_boilerplate():
    html = """<html><head><style>p{color:red}</style><script>var x=1;</script></head>
    <body><header>Site header</header><nav>Menu</nav>
    <article><h1>Headline</h1><p>First paragraph.</p>


    <p>Second paragraph.</p></article><aside>Related</aside><footer>Copyright</footer></body></html>"""
    text = np._page_text(html)
    assert "Headline" in text and "First paragraph." in text and "Second paragraph." in text
    for junk in ("<", "var x", "color:red", "Menu", "Site header", "Related", "Copyright"):
        assert junk not in text
    assert "\n\n\n" not in text


# --- fetch_article -----------------------------------------------------------


class FakeResponse:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"Client error '{self.status_code}'", request=None, response=None)


def article_html(date="2026-09-17T17:11:38Z", body="word " * 80):
    return (f'<html><head><title>Page title</title>'
            f'<meta property="article:published_time" content="{date}"></head>'
            f'<body><p>{body}</p></body></html>')


def stub_get(monkeypatch, response):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(np.httpx, "get", get)
    return calls


def test_fetch_article_returns_text_dates_and_retrieval_time(monkeypatch):
    calls = stub_get(monkeypatch, FakeResponse(article_html()))
    before = datetime.now(timezone.utc).replace(microsecond=0)

    info = np.fetch_article({"url": "http://x/a", "title": "Search title"})

    assert info["title"] == "Search title"
    assert "word word word" in info["text"]
    assert info["published_at"] == "2026-09-17T17:11:38Z"
    assert info["published_at_source"] == "meta_tag"
    retrieved = datetime.strptime(info["retrieved_at"], np._TS_FORMAT).replace(tzinfo=timezone.utc)
    assert before <= retrieved <= datetime.now(timezone.utc)
    assert calls[0][1]["follow_redirects"] is True and "User-Agent" in calls[0][1]["headers"]


def test_fetch_article_uses_page_title_when_search_gave_none(monkeypatch):
    stub_get(monkeypatch, FakeResponse(article_html()))
    assert np.fetch_article({"url": "http://x/a"})["title"] == "Page title"


def test_fetch_article_falls_back_to_the_search_snippet_for_thin_pages(monkeypatch):
    stub_get(monkeypatch, FakeResponse("<html><body><p>paywall</p></body></html>"))
    snippet = "snippet " * 40
    info = np.fetch_article({"url": "http://x/a", "title": "t", "snippet": snippet})
    assert info["text"] == snippet


def test_fetch_article_keeps_thin_text_for_the_job_to_judge(monkeypatch):
    stub_get(monkeypatch, FakeResponse("<html><body><p>paywall</p></body></html>"))
    info = np.fetch_article({"url": "http://x/a", "title": "t"})
    assert info["text"] == "paywall"  # rejected later by _reject_reason


def test_fetch_article_does_not_judge_the_age(monkeypatch):
    stub_get(monkeypatch, FakeResponse(article_html(date="2020-01-01T00:00:00Z")))
    assert np.fetch_article({"url": "http://x/a", "title": "t"})["published_at"].startswith("2020")


@pytest.mark.parametrize(
    "response",
    [FakeResponse("forbidden", 403), httpx.ConnectError("refused"), httpx.ReadTimeout("slow")],
)
def test_fetch_article_turns_download_errors_into_skips(monkeypatch, response):
    stub_get(monkeypatch, response)
    with pytest.raises(np.ArticleSkipped, match="download failed"):
        np.fetch_article({"url": "http://x/a", "title": "t"})


def test_reject_reason_orders_age_before_length():
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    old = (datetime.now(timezone.utc) - timedelta(days=40)).strftime(np._TS_FORMAT)
    assert np._reject_reason({"published_at": old, "text": "short"}, cutoff) == "older than window"
    assert np._reject_reason({"published_at": None, "text": "short"}, cutoff) == "not enough article text"
    assert np._reject_reason({"published_at": None, "text": None}, cutoff) == "not enough article text"
    assert np._reject_reason({"published_at": None, "text": "x" * 200}, cutoff) is None


# --- search ------------------------------------------------------------------


def test_search_merges_queries_and_dedupes_by_url(monkeypatch):
    queries = []

    class FakeDDGS:
        def news(self, query, timelimit, max_results):
            queries.append((query, timelimit, max_results))
            if "earnings" in query:
                return [{"url": "http://x/2", "title": "two"}, {"url": "http://x/3", "title": "three"}]
            return [{"url": "http://x/1", "title": "one"}, {"url": "http://x/2", "title": "dup"}]

    monkeypatch.setattr(np, "DDGS", FakeDDGS)
    results = np._search("AAPL", 10)
    assert [r["url"] for r in results] == ["http://x/1", "http://x/2", "http://x/3"]
    assert results[1]["title"] == "dup"  # first occurrence wins
    assert {q[1] for q in queries} == {"m"} and {q[2] for q in queries} == {10}


def test_search_survives_a_failing_query(monkeypatch):
    class FakeDDGS:
        def news(self, query, timelimit, max_results):
            if "earnings" in query:
                raise RuntimeError("rate limited")
            return [{"url": "http://x/1"}]

    monkeypatch.setattr(np, "DDGS", FakeDDGS)
    assert [r["url"] for r in np._search("AAPL", 5)] == ["http://x/1"]
