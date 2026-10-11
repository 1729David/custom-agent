import httpx
import ollama
import pytest

import news_pipeline as np


class FakeClient:
    """Stands in for ollama.Client; each chat() call consumes the next scripted step."""

    script: list = []
    timeouts: list = []
    requests: list = []

    def __init__(self, timeout=None):
        FakeClient.timeouts.append(timeout)

    def chat(self, model, messages, think=None, options=None):
        FakeClient.requests.append({"model": model, "think": think, "options": options})
        step = FakeClient.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return {"message": {"content": step}}


@pytest.fixture(autouse=True)
def fake_ollama(monkeypatch, sleeps):
    FakeClient.script, FakeClient.timeouts, FakeClient.requests = [], [], []
    monkeypatch.setattr(np.ollama, "Client", FakeClient)
    return FakeClient


DROPPED = httpx.RemoteProtocolError("Server disconnected without sending a response.")


def test_success_needs_no_retry(fake_ollama, sleeps):
    fake_ollama.script = ["  hello \n"]
    assert np._chat("m", "p") == "hello"
    assert sleeps == []


def test_every_call_has_a_timeout(fake_ollama):
    fake_ollama.script = ["x"]
    np._chat("m", "p")
    assert fake_ollama.timeouts == [np._LLM_TIMEOUT]


def test_transient_errors_are_retried_with_backoff(fake_ollama, sleeps):
    fake_ollama.script = [DROPPED, DROPPED, "recovered"]
    assert np._chat("m", "p") == "recovered"
    assert sleeps == [5, 20]


@pytest.mark.parametrize(
    "error",
    [
        DROPPED,
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("timed out"),
        ConnectionError("reset"),
        TimeoutError("slow"),
        ollama.ResponseError("overloaded", 503),
    ],
)
def test_each_kind_of_transient_error_is_retried(fake_ollama, sleeps, error):
    fake_ollama.script = [error, "ok"]
    assert np._chat("m", "p") == "ok"
    assert sleeps == [5]


def test_gives_up_after_the_last_delay(fake_ollama, sleeps):
    fake_ollama.script = [DROPPED, DROPPED, DROPPED, DROPPED]
    with pytest.raises(np.LLMUnavailable, match="RemoteProtocolError"):
        np._chat("m", "p")
    assert sleeps == [5, 20, 60]
    assert fake_ollama.script == []  # 1 attempt + 3 retries, no more


@pytest.mark.parametrize(
    "error",
    [ollama.ResponseError("model not found", 404), ValueError("bad"), KeyError("x")],
)
def test_other_errors_are_not_retried(fake_ollama, sleeps, error):
    fake_ollama.script = [error]
    with pytest.raises(type(error)):
        np._chat("m", "p")
    assert sleeps == []


# --- relevance ---------------------------------------------------------------


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("MAIN SUBJECT: Apple Inc.\nANSWER: YES", True),
        ("MAIN SUBJECT: several companies\nANSWER: NO", False),
        ("main subject: OKX\nanswer: no", False),
        ("MAIN SUBJECT: Apple\nANSWER: **YES**", True),
        ("MAIN SUBJECT: Apple\nANSWER: `NO`", False),
        ("MAIN SUBJECT: Yes Bank\nANSWER: NO", False),  # a "yes" in the subject isn't the answer
        ("MAIN SUBJECT: Apple\nANSWER: YES\nANSWER: NO", True),  # the first answer counts
        # bare replies (a model ignoring the format) still work
        ("YES", True), ("yes", True), ("**YES**", True), ('"Yes."', True),
        ("NO", False), ("no.", False), ("**NO**", False), ("Not really", False),
    ],
)
def test_relevance_answer_parsing(fake_ollama, answer, expected):
    fake_ollama.script = [answer]
    assert np.check_relevance("AAPL", "t", "text") is expected


def test_relevance_fails_open_by_default_when_ollama_is_down(fake_ollama):
    fake_ollama.script = [DROPPED] * 4
    assert np.check_relevance("AAPL", "t", "text") is True


def test_relevance_can_fail_closed_to_avoid_guessing(fake_ollama):
    fake_ollama.script = [DROPPED] * 4
    with pytest.raises(np.LLMUnavailable):
        np.check_relevance("AAPL", "t", "text", fail_open=False)


def test_relevance_prompt_defines_the_rules_and_carries_the_article(monkeypatch):
    prompts = []
    monkeypatch.setattr(np, "_chat", lambda model, prompt, **kw: prompts.append(prompt) or "ANSWER: YES")
    np.check_relevance("AAPL", "A headline", "#" * 5000)
    prompt = prompts[0]
    assert "AAPL" in prompt and "A headline" in prompt
    assert "market wrap" in prompt and "MAIN SUBJECT" in prompt and "ANSWER: YES or NO" in prompt
    assert "stock quote, price chart, data table" in prompt  # quote pages are not news
    assert prompt.count("#") == 3000  # the article text is truncated


def test_models_are_configurable(monkeypatch):
    used = []
    monkeypatch.setattr(np, "_chat", lambda model, prompt, **kw: used.append(model) or "YES")
    monkeypatch.setattr(np, "RELEVANCE_MODEL", "small-model")
    monkeypatch.setattr(np, "SUMMARY_MODEL", "big-model")
    np.check_relevance("AAPL", "t", "text")
    np.summarize_article("AAPL", "t", "text")
    np.build_report("AAPL", [{"published_at": None, "title": "t", "source": "s",
                              "summary": "x", "article_id": 1, "url": "u", "retrieved_at": "r"}])
    assert used == ["small-model", "big-model", "big-model"]
    np.check_relevance("AAPL", "t", "text", model="explicit")
    assert used[-1] == "explicit"


def test_prompts_carry_the_ticker_and_truncate_long_text(monkeypatch):
    prompts = []
    monkeypatch.setattr(np, "_chat", lambda model, prompt, **kw: prompts.append(prompt) or "ok")
    np.summarize_article("AAPL", "A title", "x" * 20000)
    assert "AAPL" in prompts[0] and "A title" in prompts[0]
    assert prompts[0].count("x") == np._ARTICLE_TEXT_LIMIT


def test_report_is_ordered_oldest_first_with_undated_first(monkeypatch):
    monkeypatch.setattr(np, "_chat", lambda model, prompt, **kw: "OVERVIEW")

    def art(i, published):
        return {"article_id": i, "title": f"t{i}", "source": "s", "url": f"u{i}",
                "published_at": published, "retrieved_at": "r", "summary": f"s{i}"}

    report = np.build_report(
        "AAPL", [art(1, "2026-09-03T00:00:00Z"), art(2, None), art(3, "2026-09-01T00:00:00Z")]
    )
    assert [a["article_id"] for a in report["articles"]] == [2, 3, 1]
    assert report["article_count"] == 3 and report["overview"] == "OVERVIEW"
    assert report["generated_at"].endswith("Z")


# --- thinking mode -----------------------------------------------------------


def test_thinking_is_off_by_default(fake_ollama):
    fake_ollama.script = ["ok"]
    np._chat("m", "p")
    assert fake_ollama.requests == [{"model": "m", "think": False, "options": None}]


def test_reply_length_is_capped_only_when_thinking_is_off(fake_ollama, monkeypatch):
    fake_ollama.script = ["ok", "ok"]
    np._chat("m", "p", max_tokens=16)
    assert fake_ollama.requests[-1]["options"] == {"num_predict": 16}

    monkeypatch.setattr(np, "_LLM_THINK", True)  # reasoning would eat the budget
    np._chat("m", "p", max_tokens=16)
    assert fake_ollama.requests[-1] == {"model": "m", "think": True, "options": None}


def test_relevance_check_is_short_and_repeatable(monkeypatch):
    seen = {}
    monkeypatch.setattr(np, "_chat", lambda model, prompt, **kw: seen.update(kw) or "ANSWER: YES")
    np.check_relevance("AAPL", "t", "text")
    assert seen == {"max_tokens": np._RELEVANCE_MAX_TOKENS, "temperature": 0}


def test_temperature_is_sent_only_when_thinking_is_off(fake_ollama, monkeypatch):
    fake_ollama.script = ["ok", "ok"]
    np._chat("m", "p", temperature=0)
    assert fake_ollama.requests[-1]["options"] == {"temperature": 0}
    np._chat("m", "p", max_tokens=60, temperature=0)
    assert fake_ollama.requests[-1]["options"] == {"num_predict": 60, "temperature": 0}

    monkeypatch.setattr(np, "_LLM_THINK", True)
    fake_ollama.script = ["ok"]
    np._chat("m", "p", max_tokens=60, temperature=0)
    assert fake_ollama.requests[-1]["options"] is None


def test_thinking_can_be_enabled_from_the_environment(monkeypatch):
    import importlib

    monkeypatch.setenv("NEWS_LLM_THINK", "1")
    assert importlib.reload(np)._LLM_THINK is True
    monkeypatch.delenv("NEWS_LLM_THINK")
    assert importlib.reload(np)._LLM_THINK is False


# --- overview prompt ---------------------------------------------------------


def test_overview_prompt_warns_that_bracketed_dates_are_publication_dates(monkeypatch):
    prompts = []
    monkeypatch.setattr(np, "_chat", lambda model, prompt, **kw: prompts.append(prompt) or "OVERVIEW")

    def art(i, published):
        return {"article_id": i, "title": f"Headline {i}", "source": "Src", "url": f"u{i}",
                "published_at": published, "retrieved_at": "r", "summary": f"Summary {i}."}

    np.build_report("NVDA", [art(2, "2026-09-23T00:00:00Z"), art(1, None)])

    prompt = prompts[0]
    assert "NVDA" in prompt
    assert "published" in prompt and "later than the events" in prompt
    assert "date not stated" in prompt  # the wording to use when no summary gives a date
    assert "never describe such an event as happening" in prompt
    # the digest still carries each date and summary, undated articles first
    assert prompt.index("[date unknown] Headline 1") < prompt.index("[2026-09-23T00:00:00Z] Headline 2")
    assert "Summary 1." in prompt and "Summary 2." in prompt
