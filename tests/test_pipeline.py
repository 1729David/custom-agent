from datetime import datetime, timezone

import pytest

import news_pipeline as np
from jobs import JobInterrupted
from tests.helpers import make_ctx, run_job


def article_rows(ctx):
    return {a["url"].rsplit("/", 1)[1]: a for a in ctx.articles()}


# --- happy path and report ---------------------------------------------------


def test_report_has_summaries_overview_and_skipped(world, pipeline_store):
    world.add("old", days_old=3)
    world.add("new", days_old=1)
    world.add("blocked", error=np.ArticleSkipped("download failed: 403"))

    report, _ = run_job(pipeline_store)

    assert report["ticker"] == "AAPL"
    assert report["overview"] == "OVERVIEW"
    assert report["article_count"] == 2
    assert [a["title"] for a in report["articles"]] == ["Title old", "Title new"]  # oldest first
    first = report["articles"][0]
    assert set(first) == {
        "article_id", "title", "source", "url", "published_at", "retrieved_at", "summary",
    }
    assert first["summary"] == "AAPL summary of Title old"
    assert first["url"] == world.url("old")
    assert report["skipped"] == [
        {"url": world.url("blocked"), "title": "Title blocked", "reason": "download failed: 403"}
    ]


def test_ticker_is_normalized_in_the_report(world, pipeline_store):
    world.add("a")
    report, _ = run_job(pipeline_store, ticker="aapl")
    assert report["ticker"] == "AAPL"
    assert world.calls["summary"] == [("AAPL", "Title a")]


# --- skip reasons ------------------------------------------------------------


def test_skip_reasons(world, pipeline_store):
    world.add("good")
    world.add("ancient", days_old=45)
    world.add("short", text="too short")
    world.add("offtopic")
    world.irrelevant.add(("AAPL", "Title offtopic"))
    world.add("down", error=np.ArticleSkipped("download failed: 500"))

    report, ctx = run_job(pipeline_store)

    assert [a["title"] for a in report["articles"]] == ["Title good"]
    reasons = {s["url"].rsplit("/", 1)[1]: s["reason"] for s in report["skipped"]}
    assert reasons == {
        "ancient": "older than window",
        "short": "not enough article text",
        "offtopic": "not primarily about AAPL",
        "down": "download failed: 500",
    }
    rows = article_rows(ctx)
    assert rows["good"]["status"] == "done"
    assert all(rows[n]["status"] == "skipped" for n in reasons)


def test_article_without_a_published_date_is_kept(world, pipeline_store):
    world.add("undated", published=False)
    report, _ = run_job(pipeline_store)
    assert report["articles"][0]["published_at"] is None


def test_irrelevant_articles_are_not_summarized(world, pipeline_store):
    world.add("a")
    world.add("b")
    world.irrelevant.add(("AAPL", "Title b"))
    run_job(pipeline_store)
    assert world.calls["summary"] == [("AAPL", "Title a")]


# --- limits ------------------------------------------------------------------


def test_max_articles_stops_further_work(world, pipeline_store):
    for name in "abcdef":
        world.add(name)

    report, ctx = run_job(pipeline_store, max_articles=2)

    assert [a["title"] for a in report["articles"]] == ["Title a", "Title b"]
    assert world.calls["fetch"] == [world.url("a"), world.url("b")]  # later ones never fetched
    reasons = {s["url"].rsplit("/", 1)[1]: s["reason"] for s in report["skipped"]}
    # only 2 * max_articles candidates are listed, so e and f never appear
    assert reasons == {n: "max articles reached" for n in "cd"}


def test_rejected_articles_do_not_count_toward_max(world, pipeline_store):
    world.add("bad", days_old=60)
    world.add("a")
    world.add("b")
    report, _ = run_job(pipeline_store, max_articles=2)
    assert len(report["articles"]) == 2


def test_candidate_list_is_capped_at_twice_max_articles(monkeypatch):
    seen = {}

    def fake_search(ticker, max_results):
        seen["max_results"] = max_results
        return [{"url": f"http://x/{i}", "title": "t", "source": "s", "date": None, "body": "b"}
                for i in range(47)]

    monkeypatch.setattr(np, "_search", fake_search)
    candidates = np.list_articles("aapl", 15)
    assert len(candidates) == 30 and seen["max_results"] == 30
    assert candidates[0] == {"url": "http://x/0", "title": "t", "source": "s",
                             "search_date": None, "snippet": "b"}
    assert len(np.list_articles("AAPL", 5)) == 10


def test_no_candidates_raises_lookup_error(world, pipeline_store):
    with pytest.raises(LookupError, match="No news found for AAPL"):
        run_job(pipeline_store)


def test_nothing_usable_raises_lookup_error(world, pipeline_store):
    world.add("old", days_old=90)
    with pytest.raises(LookupError, match="No usable news found"):
        run_job(pipeline_store)


# --- reuse across jobs -------------------------------------------------------


def test_second_job_reuses_everything_already_processed(world, pipeline_store):
    for name in "abc":
        world.add(name)
    first, _ = run_job(pipeline_store)
    fetched, summarized = list(world.calls["fetch"]), list(world.calls["summary"])

    world.add("d")
    second, ctx = run_job(pipeline_store)

    assert world.calls["fetch"] == fetched + [world.url("d")]  # only the new article
    assert world.calls["summary"] == summarized + [("AAPL", "Title d")]
    ids = {a["title"]: a["article_id"] for a in first["articles"]}
    for article in second["articles"][:3]:
        assert article["article_id"] == ids[article["title"]]  # same entry
    reused = {n: a["reused"] for n, a in article_rows(ctx).items()}
    assert reused == {"a": True, "b": True, "c": True, "d": False}


def test_reused_article_keeps_its_original_retrieval_time(world, pipeline_store):
    world.add("a")
    first, _ = run_job(pipeline_store)
    world.pages[world.url("a")]["retrieved_at"] = "2099-01-01T00:00:00Z"
    second, _ = run_job(pipeline_store)
    assert second["articles"][0]["retrieved_at"] == first["articles"][0]["retrieved_at"]


def test_refresh_downloads_again_as_a_new_entry(world, pipeline_store):
    world.add("a")
    first, _ = run_job(pipeline_store)
    refreshed, _ = run_job(pipeline_store, refresh=True)

    assert world.calls["fetch"] == [world.url("a")] * 2
    assert refreshed["articles"][0]["article_id"] != first["articles"][0]["article_id"]
    assert len(world.calls["summary"]) == 2  # the new entry is analysed afresh

    reused, _ = run_job(pipeline_store)  # a later normal job uses the latest entry
    assert reused["articles"][0]["article_id"] == refreshed["articles"][0]["article_id"]
    assert len(world.calls["fetch"]) == 2


def test_other_ticker_reuses_text_but_recomputes_the_analysis(world, pipeline_store):
    world.add("a")
    world.add("b")
    run_job(pipeline_store, ticker="AAPL")
    world.irrelevant.add(("MSFT", "Title b"))

    report, _ = run_job(pipeline_store, ticker="MSFT")

    assert len(world.calls["fetch"]) == 2  # nothing downloaded again
    assert [c for c in world.calls["relevance"] if c[0] == "MSFT"] == [
        ("MSFT", "Title a"), ("MSFT", "Title b")]
    assert world.calls["summary"][-1] == ("MSFT", "Title a")
    assert [a["title"] for a in report["articles"]] == ["Title a"]


def test_window_is_judged_per_job_against_stored_articles(world, pipeline_store):
    world.add("recent")
    world.add("older", days_old=45)

    narrow, _ = run_job(pipeline_store, days=30)
    assert [a["title"] for a in narrow["articles"]] == ["Title recent"]
    assert narrow["skipped"][0]["reason"] == "older than window"

    wide, _ = run_job(pipeline_store, days=60)
    assert {a["title"] for a in wide["articles"]} == {"Title recent", "Title older"}
    assert world.calls["fetch"].count(world.url("older")) == 1  # not downloaded twice


def test_short_text_articles_are_cached_and_not_refetched(world, pipeline_store):
    world.add("short", text="tiny")
    for _ in range(2):  # two separate jobs
        with pytest.raises(LookupError):
            run_job(pipeline_store)
    assert world.calls["fetch"] == [world.url("short")]


def recovered_page():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"title": "Title flaky", "text": "word " * 80, "published_at": now,
            "published_at_source": "meta_tag", "retrieved_at": now}


def test_failed_downloads_are_not_stored_and_are_retried(world, pipeline_store):
    world.add("flaky", error=np.ArticleSkipped("download failed: 403"))
    with pytest.raises(LookupError):
        run_job(pipeline_store)

    world.pages[world.url("flaky")] = recovered_page()  # the site comes back
    report, _ = run_job(pipeline_store)
    assert [a["title"] for a in report["articles"]] == ["Title flaky"]
    assert world.calls["fetch"].count(world.url("flaky")) == 2


# --- resume ------------------------------------------------------------------


def test_job_resumes_without_repeating_finished_work(world, pipeline_store):
    for name in "abcd":
        world.add(name)
    world.summary_error = RuntimeError("model crashed")
    ctx, params = make_ctx(pipeline_store)

    with pytest.raises(RuntimeError):
        np.run_news_job(ctx, **params)
    assert world.calls["fetch"] == [world.url("a")]  # stopped at the first failure

    world.summary_error = None
    report = np.run_news_job(ctx, **params)  # same job, run again

    assert [a["title"] for a in report["articles"]] == [f"Title {n}" for n in "abcd"]
    assert world.calls["fetch"].count(world.url("a")) == 1  # never downloaded twice
    assert world.calls["relevance"].count(("AAPL", "Title a")) == 1  # verdict was saved
    assert world.calls["list"] == 1  # candidates are listed once


# --- interruption ------------------------------------------------------------


def test_ollama_outage_pauses_the_job_and_stores_no_guessed_verdict(
    world, pipeline_store, monkeypatch
):
    world.add("a")

    def down(ticker, title, text, model=None, fail_open=True):
        assert fail_open is False  # jobs must never guess
        raise np.LLMUnavailable("RemoteProtocolError: Server disconnected")

    monkeypatch.setattr(np, "check_relevance", down)
    ctx, params = make_ctx(pipeline_store)

    with pytest.raises(JobInterrupted) as exc:
        np.run_news_job(ctx, **params)

    assert exc.value.action == "pause"
    assert "Ollama unavailable" in exc.value.message and "Server disconnected" in exc.value.message
    assert ctx.cached_article(world.url("a"))["relevant"] is None  # nothing recorded

    monkeypatch.setattr(np, "check_relevance", world.check_relevance)
    report = np.run_news_job(ctx, **params)
    assert report["article_count"] == 1
    assert world.calls["fetch"] == [world.url("a")]  # resumed, not refetched


def test_checkpoint_interrupts_between_steps(world, pipeline_store, monkeypatch):
    for name in "abc":
        world.add(name)
    ctx, params = make_ctx(pipeline_store)

    original = world.summarize_article

    def summarize_then_cancel(ticker, title, text, model=None):
        result = original(ticker, title, text, model)
        pipeline_store._control[ctx.job_id] = "cancel"  # a cancel request arrives
        return result

    monkeypatch.setattr(np, "summarize_article", summarize_then_cancel)
    with pytest.raises(JobInterrupted) as exc:
        np.run_news_job(ctx, **params)
    assert exc.value.action == "cancel"
    assert world.calls["fetch"] == [world.url("a")]  # stopped before the next article


# --- duplicates --------------------------------------------------------------


def words(prefix, n):
    """n distinct words, so two texts built from different prefixes share no phrases."""
    return " ".join(f"{prefix}{i}" for i in range(n))


def skipped_reasons(report):
    return {s["url"].rsplit("/", 1)[1]: s["reason"] for s in report["skipped"]}


def test_a_copy_under_another_url_is_skipped_before_any_model_call(world, pipeline_store):
    body = words("story", 120)
    world.add("orig", text=body)
    world.add("copy", text=body)  # a different title and URL, but the same story

    report, ctx = run_job(pipeline_store)

    assert [a["title"] for a in report["articles"]] == ["Title orig"]
    original_id = report["articles"][0]["article_id"]
    assert skipped_reasons(report) == {"copy": f"duplicate of article {original_id}"}
    assert world.calls["relevance"] == [("AAPL", "Title orig")]  # no wasted LLM calls
    assert world.calls["summary"] == [("AAPL", "Title orig")]
    assert article_rows(ctx)["copy"]["status"] == "skipped"


def test_the_first_copy_in_search_order_wins(world, pipeline_store):
    body = words("story", 120)
    world.add("second_site", text=body)
    world.add("first_site", text=body)
    report, _ = run_job(pipeline_store)
    assert [a["title"] for a in report["articles"]] == ["Title second_site"]


def test_an_excerpt_of_a_longer_article_is_a_copy(world, pipeline_store):
    body = words("story", 200)
    world.add("full", text=body)
    world.add("excerpt", text=" ".join(body.split()[:60]))
    report, _ = run_job(pipeline_store)
    assert [a["title"] for a in report["articles"]] == ["Title full"]


def test_same_title_with_partial_overlap_is_a_copy(world, pipeline_store):
    shared = words("shared", 30)
    world.add("a", title="How to play AAPL stock", text=shared + " " + words("alpha", 70))
    world.add("b", title="How to Play AAPL Stock!", text=shared + " " + words("beta", 70))
    report, _ = run_job(pipeline_store)
    assert len(report["articles"]) == 1 and "b" in skipped_reasons(report)


def test_identical_generic_titles_over_different_content_are_not_copies(world, pipeline_store):
    world.add("site_one", title="AAPL Stock Price Prediction", text=words("one", 100))
    world.add("site_two", title="AAPL Stock Price Prediction", text=words("two", 100))
    report, _ = run_job(pipeline_store)
    assert len(report["articles"]) == 2 and report["skipped"] == []


def test_partial_overlap_with_different_titles_is_not_a_copy(world, pipeline_store):
    shared = words("shared", 30)
    world.add("a", text=shared + " " + words("alpha", 70))
    world.add("b", text=shared + " " + words("beta", 70))
    report, _ = run_job(pipeline_store)
    assert len(report["articles"]) == 2


def test_different_stories_sharing_site_boilerplate_are_kept(world, pipeline_store):
    """Two real articles from one outlet overlapped by ~52% (shared footer text)."""
    boilerplate = words("footer", 50)
    world.add("markets", text=words("markets", 50) + " " + boilerplate)
    world.add("okx", text=words("okx", 50) + " " + boilerplate)
    report, _ = run_job(pipeline_store)
    assert {a["title"] for a in report["articles"]} == {"Title markets", "Title okx"}
    assert report["skipped"] == []


def test_copies_do_not_count_toward_max_articles(world, pipeline_store):
    body = words("story", 120)
    world.add("a", text=body)
    world.add("a_copy", text=body)
    world.add("b")
    report, _ = run_job(pipeline_store, max_articles=2)
    assert {a["title"] for a in report["articles"]} == {"Title a", "Title b"}


def test_dedupe_survives_a_resume(world, pipeline_store):
    body = words("story", 120)
    world.add("orig", text=body)
    world.add("copy", text=body)
    world.summary_error = RuntimeError("model crashed")
    ctx, params = make_ctx(pipeline_store)
    with pytest.raises(RuntimeError):
        np.run_news_job(ctx, **params)  # stops while "orig" is accepted but unsummarized

    world.summary_error = None
    report = np.run_news_job(ctx, **params)

    assert [a["title"] for a in report["articles"]] == ["Title orig"]
    assert "copy" in skipped_reasons(report)
    assert world.calls["relevance"] == [("AAPL", "Title orig")]


def test_a_later_job_reaches_the_same_decision_without_new_work(world, pipeline_store):
    body = words("story", 120)
    world.add("orig", text=body)
    world.add("copy", text=body)
    first, _ = run_job(pipeline_store)
    keys = ("fetch", "relevance", "summary")
    calls_after_first = {k: list(world.calls[k]) for k in keys}

    second, _ = run_job(pipeline_store)

    assert [a["article_id"] for a in second["articles"]] == [a["article_id"] for a in first["articles"]]
    assert skipped_reasons(second) == skipped_reasons(first)
    for key in keys:
        assert world.calls[key] == calls_after_first[key]  # all reused


# --- duplicate detection helpers ---------------------------------------------


def entry(article_id, title, text):
    return np._dup_entry({"article_id": article_id, "title": title, "text": text})


def test_find_duplicate_ignores_texts_too_short_to_compare():
    short = words("tiny", 12)  # 8 phrases
    assert np.find_duplicate({"title": "t", "text": short}, [entry(1, "t", short)]) is None


def test_find_duplicate_thresholds():
    base = words("w", 100)  # 96 phrases
    accepted = [entry(7, "Some title", base)]
    identical = {"title": "x", "text": base}
    assert np.find_duplicate(identical, accepted) == 7

    # ~71% of the new text's phrases are in the original: just over the bar
    over = {"title": "x", "text": " ".join(base.split()[:75]) + " " + words("n", 25)}
    assert np.find_duplicate(over, accepted) == 7
    # ~48% overlap: a different story
    under = {"title": "x", "text": " ".join(base.split()[:50]) + " " + words("n", 50)}
    assert np.find_duplicate(under, accepted) is None
    # the same overlap with a matching title is enough
    assert np.find_duplicate({**under, "title": "SOME title!"}, accepted) == 7


def test_title_key_normalizes_case_and_punctuation():
    assert np._title_key("How to Play AAPL Stock — Now!") == np._title_key("how to play aapl stock now")
    assert np._title_key(None) == "" and np._title_key("") == ""


def test_empty_titles_never_match_each_other():
    base = words("w", 100)
    other = " ".join(base.split()[:50]) + " " + words("n", 50)
    assert np.find_duplicate({"title": "", "text": other}, [entry(1, "", base)]) is None
