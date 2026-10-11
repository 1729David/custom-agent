import threading

import pytest

from jobs import InvalidTransition, JobContext, JobInterrupted, JobStore
from tests.helpers import wait_until


def status_of(store, job):
    return store.get(job["id"])["status"]


def echo(ctx, value=1):
    return {"value": value}


# --- lifecycle ---------------------------------------------------------------


def test_job_runs_and_stores_result(make_store):
    store = make_store({"echo": echo}, start=True)
    job, created = store.submit("echo", {"value": 3})
    assert created and job["status"] == "queued"

    done = wait_until(lambda: (j := store.get(job["id"]))["status"] == "done" and j)
    assert done["result"] == {"value": 3}
    assert done["error"] is None
    assert done["started_at"] and done["finished_at"]
    assert done["params"] == {"value": 3}


def test_runner_exception_fails_job_with_message(make_store):
    def boom(ctx):
        raise ValueError("boom")

    store = make_store({"boom": boom}, start=True)
    job, _ = store.submit("boom", {})
    failed = wait_until(lambda: (j := store.get(job["id"]))["status"] == "failed" and j)
    assert failed["error"] == "ValueError: boom"
    assert failed["result"] is None


def test_unknown_kind_is_rejected(make_store):
    store = make_store({"echo": echo})
    with pytest.raises(ValueError, match="unknown job kind"):
        store.submit("nope", {})


def test_jobs_run_one_at_a_time(make_store):
    lock = threading.Lock()
    state = {"active": 0, "max": 0}

    def tracked(ctx, n):
        with lock:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        threading.Event().wait(0.05)
        with lock:
            state["active"] -= 1
        return {}

    store = make_store({"t": tracked}, start=True)
    ids = [store.submit("t", {"n": i})[0]["id"] for i in range(4)]
    wait_until(lambda: all(store.get(i)["status"] == "done" for i in ids))
    assert state["max"] == 1


# --- duplicate handling ------------------------------------------------------


def test_identical_active_job_is_reused(make_store):
    gate = threading.Event()

    def slow(ctx, n):
        gate.wait(5)
        return {}

    store = make_store({"slow": slow}, start=True)
    first, created1 = store.submit("slow", {"n": 1})
    wait_until(lambda: status_of(store, first) == "running")
    again, created2 = store.submit("slow", {"n": 1})
    other, created3 = store.submit("slow", {"n": 2})
    assert not created2 and again["id"] == first["id"]
    assert created3 and other["id"] != first["id"]

    gate.set()
    wait_until(lambda: status_of(store, first) == "done")
    new, created4 = store.submit("slow", {"n": 1})
    assert created4 and new["id"] != first["id"]


def test_identical_paused_job_is_reused(make_store):
    store = make_store({"echo": echo})
    job, _ = store.submit("echo", {})
    store.pause(job["id"])
    again, created = store.submit("echo", {})
    assert not created and again["id"] == job["id"]
    assert again["status"] == "paused"


# --- listing -----------------------------------------------------------------


def test_list_is_newest_first_with_filter_and_limit(make_store):
    store = make_store({"echo": echo})
    ids = [store.submit("echo", {"value": i})[0]["id"] for i in range(3)]
    store._set_status(ids[1], "done", finished=True)

    assert [j["id"] for j in store.list()] == ids[::-1]
    assert [j["id"] for j in store.list(status="done")] == [ids[1]]
    assert len(store.list(limit=2)) == 2


# --- restart recovery --------------------------------------------------------


def test_restart_requeues_running_and_queued_but_not_paused(db_path):
    first = JobStore({"echo": echo}, db_path)
    running, _ = first.submit("echo", {"value": "running"})
    paused, _ = first.submit("echo", {"value": "paused"})
    queued, _ = first.submit("echo", {"value": "queued"})
    first._set_status(running["id"], "running")
    first._set_status(paused["id"], "paused")

    ran = []

    def record(ctx, value):
        ran.append(value)
        return {}

    second = JobStore({"echo": record}, db_path)
    second.start()
    try:
        wait_until(lambda: sorted(ran) == ["queued", "running"])
        assert second.get(paused["id"])["status"] == "paused"
        assert "paused" not in ran
    finally:
        second.stop()


# --- pause / resume / cancel -------------------------------------------------


def test_pause_queued_job_is_immediate_and_resume_runs_it(make_store):
    store = make_store({"echo": echo})
    job, _ = store.submit("echo", {})
    assert store.pause(job["id"])["status"] == "paused"

    store.start()
    sentinel, _ = store.submit("echo", {"value": "sentinel"})  # queued behind it
    wait_until(lambda: status_of(store, sentinel) == "done")
    assert status_of(store, job) == "paused"  # the worker passed over it

    assert store.resume(job["id"])["status"] == "queued"
    wait_until(lambda: status_of(store, job) == "done")


def make_stepper(step, done, starts):
    def runner(ctx, steps):
        starts.append(1)
        for i in range(steps):
            ctx.checkpoint()
            step.acquire(timeout=5)
            done.append(i)
        return {"steps": steps}

    return runner


def test_pause_running_job_stops_at_next_checkpoint_then_resumes(make_store):
    step, done, starts = threading.Semaphore(0), [], []
    store = make_store({"s": make_stepper(step, done, starts)}, start=True)
    job, _ = store.submit("s", {"steps": 3})

    step.release()
    wait_until(lambda: done == [0])
    paused = store.pause(job["id"])
    assert paused["status"] == "running"  # takes effect at the next checkpoint
    assert store.progress(store.get(job["id"]))["stage"] == "pausing"

    step.release()
    parked = wait_until(lambda: (j := store.get(job["id"]))["status"] == "paused" and j)
    assert done == [0, 1]
    assert parked["finished_at"] is None and parked["error"] is None
    assert store.progress(parked)["stage"] == "paused"

    started_at = parked["started_at"]
    step.release(10)
    store.resume(job["id"])
    finished = wait_until(lambda: (j := store.get(job["id"]))["status"] == "done" and j)
    assert finished["started_at"] == started_at  # first start time is kept
    assert len(starts) == 2


def test_cancel_running_job_fails_it(make_store):
    step, done, starts = threading.Semaphore(0), [], []
    store = make_store({"s": make_stepper(step, done, starts)}, start=True)
    job, _ = store.submit("s", {"steps": 3})

    step.release()
    wait_until(lambda: done == [0])
    store.cancel(job["id"])
    assert store.progress(store.get(job["id"]))["stage"] == "cancelling"
    step.release()
    failed = wait_until(lambda: (j := store.get(job["id"]))["status"] == "failed" and j)
    assert failed["error"] == "cancelled by user"
    assert failed["finished_at"]
    assert store._control == {}


@pytest.mark.parametrize("setup", ["queued", "paused"])
def test_cancel_idle_job_fails_it_immediately(make_store, setup):
    store = make_store({"echo": echo})
    job, _ = store.submit("echo", {})
    if setup == "paused":
        store.pause(job["id"])
    result = store.cancel(job["id"])
    assert result["status"] == "failed" and result["error"] == "cancelled by user"
    assert result["finished_at"]


def test_invalid_transitions(make_store):
    gate = threading.Event()

    def slow(ctx):
        gate.wait(5)
        return {}

    store = make_store({"slow": slow, "echo": echo}, start=True)
    done_job, _ = store.submit("echo", {})
    wait_until(lambda: status_of(store, done_job) == "done")
    for action in (store.pause, store.resume, store.cancel):
        with pytest.raises(InvalidTransition):
            action(done_job["id"])

    running, _ = store.submit("slow", {})
    wait_until(lambda: status_of(store, running) == "running")
    with pytest.raises(InvalidTransition):
        store.resume(running["id"])
    store.cancel(running["id"])
    with pytest.raises(InvalidTransition, match="being cancelled"):
        store.pause(running["id"])
    gate.set()

    for action in (store.pause, store.resume, store.cancel):
        with pytest.raises(KeyError):
            action("missing")


def test_runner_can_pause_itself_with_a_message(make_store):
    calls = []

    def flaky(ctx):
        calls.append(1)
        if len(calls) == 1:
            ctx.pause_job("Ollama unavailable")
        return {"ok": True}

    store = make_store({"f": flaky}, start=True)
    job, _ = store.submit("f", {})
    paused = wait_until(lambda: (j := store.get(job["id"]))["status"] == "paused" and j)
    assert paused["error"] == "Ollama unavailable"
    assert paused["finished_at"] is None

    resumed = store.resume(job["id"])
    assert resumed["error"] is None  # resume clears the explanation
    done = wait_until(lambda: (j := store.get(job["id"]))["status"] == "done" and j)
    assert done["result"] == {"ok": True}


def test_checkpoint_raises_only_when_requested(make_store):
    store = make_store({"echo": echo})
    job, _ = store.submit("echo", {})
    ctx = JobContext(store, job["id"])
    ctx.checkpoint()
    store._control[job["id"]] = "cancel"
    with pytest.raises(JobInterrupted) as exc:
        ctx.checkpoint()
    assert exc.value.action == "cancel"


# --- progress ----------------------------------------------------------------


def test_progress_stages_and_counts(make_store):
    store = make_store({"news": echo})
    job, _ = store.submit("news", {"ticker": "AAPL", "max_articles": 7})
    job_id = job["id"]
    ctx = JobContext(store, job_id)

    queued = store.progress(store.get(job_id))
    assert queued == {"stage": "queued", "target": 7, "total": 0, "counts": {}, "articles": []}

    store._set_status(job_id, "running")
    assert store.progress(store.get(job_id))["stage"] == "listing"

    ctx.add_articles([{"url": f"http://x/{i}", "title": f"T{i}"} for i in range(3)])
    progress = store.progress(store.get(job_id))
    assert progress["stage"] == "processing"
    assert progress["total"] == 3 and progress["counts"] == {"pending": 3}
    assert progress["target"] == 7

    for i, status in enumerate(("done", "skipped", "skipped")):
        ctx.update_article(f"http://x/{i}", status=status)
    progress = store.progress(store.get(job_id))
    assert progress["stage"] == "bundling"
    assert progress["counts"] == {"done": 1, "skipped": 2}

    store._set_status(job_id, "done", finished=True)
    assert store.progress(store.get(job_id))["stage"] == "done"
