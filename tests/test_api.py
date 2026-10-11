import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import server
from tests.helpers import wait_until

NEWS_RESULT = {
    "ticker": "AAPL",
    "generated_at": "2026-10-06T00:00:00Z",
    "article_count": 1,
    "overview": "An overview.",
    "articles": [{
        "article_id": 1, "title": "A title", "source": "Src", "url": "http://x/a",
        "published_at": "2026-09-01T00:00:00Z", "retrieved_at": "2026-10-01T00:00:00Z",
        "summary": "A summary.",
    }],
    "skipped": [{"url": "http://x/b", "title": None, "reason": "older than window"}],
}


@pytest.fixture
def api(make_store, monkeypatch):
    """A client for the real app over a stub "news" runner that can be held open."""
    monkeypatch.setattr(server, "API_KEY", None)
    gate = threading.Event()
    gate.set()
    calls = []

    def runner(ctx, **params):
        calls.append(params)
        gate.wait(5)
        ctx.checkpoint()
        return NEWS_RESULT

    store = make_store({"news": runner})
    with TestClient(server.create_app(store)) as client:
        try:
            yield SimpleNamespace(client=client, store=store, gate=gate, calls=calls)
        finally:
            gate.set()  # let a held job finish so shutdown doesn't wait on it


def post(api, ticker="AAPL", **body):
    return api.client.post(f"/news/{ticker}", json=body or None)


def job_status(api, job_id):
    return api.client.get(f"/jobs/{job_id}").json()["status"]


# --- starting jobs -----------------------------------------------------------


def test_start_job_returns_202_with_location_and_defaults(api):
    api.gate.clear()
    response = post(api, "aapl")
    job = response.json()
    assert response.status_code == 202
    assert response.headers["location"] == f"/jobs/{job['id']}"
    assert job["kind"] == "news" and job["status"] in ("queued", "running")
    assert job["params"] == {"ticker": "AAPL", "days": 30, "max_articles": 15, "refresh": False}
    assert job["result"] is None and job["error"] is None


def test_request_body_overrides_defaults(api):
    job = post(api, "MSFT", days=7, max_articles=3, refresh=True).json()
    assert job["params"] == {"ticker": "MSFT", "days": 7, "max_articles": 3, "refresh": True}
    wait_until(lambda: api.calls)
    assert api.calls[0] == job["params"]  # the runner gets exactly these


def test_identical_active_request_returns_the_existing_job(api):
    api.gate.clear()
    first = post(api)
    second = post(api)
    assert first.status_code == 202 and second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert post(api, refresh=True).json()["id"] != first.json()["id"]  # differs by params


@pytest.mark.parametrize("ticker", ["BRK.B", "BF-B", "aapl", "A"])
def test_valid_tickers_are_accepted_and_uppercased(api, ticker):
    response = post(api, ticker)
    assert response.status_code == 202
    assert response.json()["params"]["ticker"] == ticker.upper()


@pytest.mark.parametrize("ticker", ["1AAPL", "AAPL$", "WAYTOOLONGTICKER", "A_B", "A%20B", "-X"])
def test_invalid_tickers_are_rejected(api, ticker):
    assert post(api, ticker).status_code == 422
    assert api.calls == []


@pytest.mark.parametrize(
    "body",
    [{"days": 0}, {"days": 91}, {"max_articles": 0}, {"max_articles": 31}, {"refresh": "maybe"},
     {"days": "many"}],
)
def test_invalid_bodies_are_rejected(api, body):
    assert post(api, **body).status_code == 422


# --- reading jobs ------------------------------------------------------------


def test_finished_job_has_result_and_progress(api):
    job = post(api).json()
    wait_until(lambda: job_status(api, job["id"]) == "done")
    done = api.client.get(f"/jobs/{job['id']}").json()
    assert done["result"] == NEWS_RESULT
    assert done["started_at"] and done["finished_at"]
    assert done["progress"]["stage"] == "done"
    assert done["progress"]["target"] == 15


def test_unknown_job_is_404(api):
    assert api.client.get("/jobs/nope").status_code == 404


def test_list_is_newest_first_filterable_and_limited(api):
    ids = [post(api, t).json()["id"] for t in ("AAA", "BBB", "CCC")]
    wait_until(lambda: all(job_status(api, i) == "done" for i in ids))
    api.store._set_status(ids[0], "failed", "x", finished=True)

    listing = api.client.get("/jobs").json()
    assert [j["id"] for j in listing] == ids[::-1]
    assert all(j["progress"] is None for j in listing)  # progress is per-job only
    assert [j["id"] for j in api.client.get("/jobs?status=failed").json()] == [ids[0]]
    assert len(api.client.get("/jobs?limit=2").json()) == 2
    assert api.client.get("/jobs?limit=0").status_code == 422
    assert api.client.get("/jobs?limit=501").status_code == 422
    assert api.client.get("/jobs?status=bogus").status_code == 422


def test_results_stored_before_article_ids_existed_still_list(api):
    """Regression: the list endpoint failed validation on old results."""
    job = post(api).json()
    wait_until(lambda: job_status(api, job["id"]) == "done")
    old = {**NEWS_RESULT, "articles": [
        {k: v for k, v in NEWS_RESULT["articles"][0].items() if k != "article_id"}]}
    import json
    with api.store._connect() as conn:
        conn.execute("UPDATE jobs SET result=? WHERE id=?", (json.dumps(old), job["id"]))
    response = api.client.get("/jobs")
    assert response.status_code == 200
    assert response.json()[0]["result"]["articles"][0]["article_id"] is None


def test_failed_job_reports_the_error(make_store, monkeypatch):
    monkeypatch.setattr(server, "API_KEY", None)

    def boom(ctx, **params):
        raise LookupError("No news found for AAPL.")

    with TestClient(server.create_app(make_store({"news": boom}))) as client:
        job = client.post("/news/AAPL").json()
        wait_until(lambda: client.get(f"/jobs/{job['id']}").json()["status"] == "failed")
        failed = client.get(f"/jobs/{job['id']}").json()
        assert failed["error"] == "LookupError: No news found for AAPL."
        assert failed["result"] is None


# --- pause / resume / cancel -------------------------------------------------


def test_pause_then_resume_over_http(api):
    api.gate.clear()
    job = post(api).json()
    wait_until(lambda: job_status(api, job["id"]) == "running")

    paused = api.client.post(f"/jobs/{job['id']}/pause")
    assert paused.status_code == 200
    assert paused.json()["progress"]["stage"] == "pausing"
    api.gate.set()
    wait_until(lambda: job_status(api, job["id"]) == "paused")
    assert api.client.get(f"/jobs/{job['id']}").json()["progress"]["stage"] == "paused"

    resumed = api.client.post(f"/jobs/{job['id']}/resume")
    assert resumed.status_code == 200 and resumed.json()["status"] == "queued"
    wait_until(lambda: job_status(api, job["id"]) == "done")


def test_cancel_over_http(api):
    api.gate.clear()
    job = post(api).json()
    wait_until(lambda: job_status(api, job["id"]) == "running")
    assert api.client.post(f"/jobs/{job['id']}/cancel").json()["progress"]["stage"] == "cancelling"
    api.gate.set()
    wait_until(lambda: job_status(api, job["id"]) == "failed")
    assert api.client.get(f"/jobs/{job['id']}").json()["error"] == "cancelled by user"


def test_control_errors_are_404_and_409(api):
    job = post(api).json()
    wait_until(lambda: job_status(api, job["id"]) == "done")
    for action in ("pause", "resume", "cancel"):
        assert api.client.post(f"/jobs/{job['id']}/{action}").status_code == 409
        assert api.client.post(f"/jobs/missing/{action}").status_code == 404


# --- authentication ----------------------------------------------------------

ENDPOINTS = [
    ("post", "/news/AAPL"), ("get", "/jobs"), ("get", "/jobs/x"),
    ("post", "/jobs/x/pause"), ("post", "/jobs/x/resume"), ("post", "/jobs/x/cancel"),
]


@pytest.mark.parametrize("method, path", ENDPOINTS)
def test_every_endpoint_requires_the_key_when_configured(api, monkeypatch, method, path):
    monkeypatch.setattr(server, "API_KEY", "secret")
    call = getattr(api.client, method)
    assert call(path).status_code == 401
    assert call(path, headers={"X-API-Key": "wrong"}).status_code == 401
    assert call(path, headers={"X-API-Key": "secret"}).status_code != 401


def test_api_is_open_when_no_key_is_configured(api):
    assert api.client.get("/jobs").status_code == 200


def test_docs_describe_the_endpoints(api):
    paths = api.client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/news/{ticker}", "/jobs", "/jobs/{job_id}",
                          "/jobs/{job_id}/pause", "/jobs/{job_id}/resume", "/jobs/{job_id}/cancel"}


# --- end to end with the real pipeline ---------------------------------------


def test_full_flow_through_the_real_pipeline(make_store, world, monkeypatch):
    import news_pipeline as np

    monkeypatch.setattr(server, "API_KEY", None)
    for name in "abc":
        world.add(name)
    world.add("old", days_old=60)
    store = make_store({"news": np.run_news_job})

    with TestClient(server.create_app(store)) as client:
        first = client.post("/news/AAPL", json={"max_articles": 5}).json()
        wait_until(lambda: client.get(f"/jobs/{first['id']}").json()["status"] == "done")
        first = client.get(f"/jobs/{first['id']}").json()
        assert first["result"]["article_count"] == 3
        assert [s["reason"] for s in first["result"]["skipped"]] == ["older than window"]
        assert first["progress"]["counts"] == {"done": 3, "skipped": 1}
        assert first["progress"]["total"] == 4 and first["progress"]["target"] == 5
        assert not any(a["reused"] for a in first["progress"]["articles"])

        world.add("d")
        second = client.post("/news/AAPL", json={"max_articles": 5}).json()
        wait_until(lambda: client.get(f"/jobs/{second['id']}").json()["status"] == "done")
        second = client.get(f"/jobs/{second['id']}").json()
        reused = {a["title"]: a["reused"] for a in second["progress"]["articles"]}
        assert reused["Title a"] and reused["Title b"] and not reused["Title d"]
        ids = {a["title"]: a["article_id"] for a in first["result"]["articles"]}
        for article in second["result"]["articles"]:
            if article["title"] in ids:
                assert article["article_id"] == ids[article["title"]]
