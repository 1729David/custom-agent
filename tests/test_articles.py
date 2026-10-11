import pytest

from jobs import JobContext, normalize_url


@pytest.mark.parametrize(
    "url, expected",
    [
        ("HTTP://Example.COM/Path", "http://example.com/Path"),  # host/scheme case only
        ("http://x.com/a/", "http://x.com/a"),  # trailing slash
        ("http://x.com", "http://x.com/"),  # bare host gets a root path
        ("http://x.com/a#section", "http://x.com/a"),  # fragment
        ("http://x.com/a?utm_source=tw&utm_medium=s", "http://x.com/a"),
        ("http://x.com/a?fbclid=1&gclid=2&ref=x", "http://x.com/a"),
        ("http://x.com/a?b=2&a=1", "http://x.com/a?a=1&b=2"),  # param order
        ("http://x.com/a?id=7&utm_campaign=c", "http://x.com/a?id=7"),  # meaningful kept
        ("  http://x.com/a  ", "http://x.com/a"),
    ],
)
def test_normalize_url(url, expected):
    assert normalize_url(url) == expected


def test_equivalent_urls_share_a_key():
    assert normalize_url("http://x.com/a?utm_source=tw") == normalize_url("http://X.com/a/")


@pytest.fixture
def ctx(make_store):
    store = make_store({"news": lambda c, **p: {}})
    job, _ = store.submit("news", {"ticker": "AAPL"})
    return JobContext(store, job["id"])


INFO = {
    "title": "A title",
    "source": "Src",
    "published_at": "2026-09-01T00:00:00Z",
    "published_at_source": "meta_tag",
    "retrieved_at": "2026-10-01T00:00:00Z",
    "text": "full text",
}


def test_add_articles_keeps_order_and_ignores_duplicate_urls(ctx):
    ctx.add_articles([{"url": "http://x/1"}, {"url": "http://x/2"}])
    ctx.add_articles([{"url": "http://x/2"}, {"url": "http://x/3"}])
    assert [a["url"] for a in ctx.articles()] == ["http://x/1", "http://x/2", "http://x/3"]
    assert {a["status"] for a in ctx.articles()} == {"pending"}


def test_update_article_only_accepts_job_fields(ctx):
    ctx.add_articles([{"url": "http://x/1"}])
    ctx.update_article("http://x/1", status="skipped", reason="why", reused=True)
    art = ctx.articles()[0]
    assert (art["status"], art["reason"], art["reused"]) == ("skipped", "why", True)
    with pytest.raises(ValueError, match="unknown job-article fields"):
        ctx.update_article("http://x/1", text="nope")


def test_store_article_always_creates_a_new_row(ctx):
    first = ctx.store_article("http://x/a", INFO)
    second = ctx.store_article("http://x/a", INFO)
    assert second != first


def test_cached_article_returns_latest_retrieval_and_matches_url_variants(ctx):
    assert ctx.cached_article("http://x/a") is None
    old = ctx.store_article("http://x/a", {**INFO, "text": "old"})
    new = ctx.store_article("http://x/a?utm_source=tw", {**INFO, "text": "new"})
    found = ctx.cached_article("http://X/a/")  # a different spelling of the same URL
    assert found["article_id"] == new and found["text"] == "new"
    assert found["article_id"] != old
    assert ctx.cached_article("http://x/other") is None


def test_analysis_is_per_ticker_and_accumulates(make_store):
    store = make_store({"news": lambda c, **p: {}})
    aapl = JobContext(store, store.submit("news", {"ticker": "AAPL"})[0]["id"])
    msft = JobContext(store, store.submit("news", {"ticker": "MSFT"})[0]["id"])
    article_id = aapl.store_article("http://x/a", INFO)

    assert aapl.cached_article("http://x/a")["relevant"] is None
    aapl.store_analysis(article_id, relevant=True)
    aapl.store_analysis(article_id, summary="AAPL summary")  # keeps the verdict
    found = aapl.cached_article("http://x/a")
    assert found["relevant"] is True and found["summary"] == "AAPL summary"

    other = msft.cached_article("http://x/a")  # same article, other ticker
    assert other["article_id"] == article_id
    assert other["relevant"] is None and other["summary"] is None
    msft.store_analysis(article_id, relevant=False)
    assert msft.cached_article("http://x/a")["relevant"] is False
    assert aapl.cached_article("http://x/a")["relevant"] is True


def test_false_verdict_is_stored_not_confused_with_unknown(ctx):
    article_id = ctx.store_article("http://x/a", INFO)
    ctx.store_analysis(article_id, relevant=False)
    assert ctx.cached_article("http://x/a")["relevant"] is False


def test_articles_joins_article_and_analysis(ctx):
    ctx.add_articles([{"url": "http://x/a", "title": "search title", "source": "Src"}])
    pending = ctx.articles()[0]
    assert pending["article_id"] is None and pending["text"] is None

    article_id = ctx.store_article("http://x/a", INFO)
    ctx.store_analysis(article_id, relevant=True, summary="sum")
    ctx.update_article("http://x/a", status="done", article_id=article_id, reused=False)
    art = ctx.articles()[0]
    assert art["title"] == "A title"  # the retrieved article's title wins
    assert (art["text"], art["summary"], art["relevant"]) == ("full text", "sum", True)
    assert art["retrieved_at"] == "2026-10-01T00:00:00Z"
