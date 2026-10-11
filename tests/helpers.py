import time

import news_pipeline as np
from jobs import JobContext


def wait_until(predicate, timeout=5.0):
    """Poll until predicate() is truthy and return its value."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("timed out waiting for condition")


def default_params(ticker="AAPL", **overrides):
    params = {"ticker": ticker, "days": 30, "max_articles": 15, "refresh": False}
    params.update(overrides)
    return params


def make_ctx(store, ticker="AAPL", **overrides):
    """Create a queued job (the worker isn't running) and return (ctx, params)."""
    params = default_params(ticker, **overrides)
    job, _ = store.submit("news", params)
    return JobContext(store, job["id"]), params


def run_job(store, ticker="AAPL", **overrides):
    """Run the news pipeline synchronously as one job; mark it done like the worker."""
    ctx, params = make_ctx(store, ticker, **overrides)
    try:
        report = np.run_news_job(ctx, **params)
    except Exception as e:
        store._set_status(ctx.job_id, "failed", f"{type(e).__name__}: {e}", finished=True)
        raise
    store._set_status(ctx.job_id, "done", finished=True)
    return report, ctx
