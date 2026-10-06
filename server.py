import logging
import os
import re
import secrets
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field

from jobs import JobStore
from news_pipeline import run_news_job

log = logging.getLogger("server")

API_KEY = os.environ.get("API_KEY")
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

JobStatus = Literal["queued", "running", "done", "failed"]


class NewsRequest(BaseModel):
    days: int = Field(30, ge=1, le=90, description="Lookback window in days")
    max_articles: int = Field(15, ge=1, le=30, description="Maximum articles to archive")


class Article(BaseModel):
    title: str | None
    source: str | None
    url: str
    published_at: str | None
    retrieved_at: str
    summary: str


class SkippedArticle(BaseModel):
    url: str
    title: str | None = None
    reason: str


class NewsResult(BaseModel):
    ticker: str
    generated_at: str
    article_count: int
    overview: str
    articles: list[Article]
    skipped: list[SkippedArticle]


class ArticleProgress(BaseModel):
    url: str
    title: str | None
    source: str | None
    status: Literal["pending", "fetched", "relevant", "done", "skipped"]
    reason: str | None
    published_at: str | None
    retrieved_at: str | None


class Progress(BaseModel):
    stage: Literal["queued", "listing", "processing", "bundling", "done", "failed"]
    total: int
    counts: dict[str, int]
    articles: list[ArticleProgress]


class Job(BaseModel):
    id: str
    kind: str
    params: dict
    status: JobStatus
    created_at: str
    started_at: str | None
    finished_at: str | None
    result: NewsResult | None
    error: str | None
    progress: Progress | None = None


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def create_app(store: JobStore | None = None) -> FastAPI:
    if store is None:
        store = JobStore({"news": run_news_job})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if API_KEY is None:
            log.warning("API_KEY is not set; the API is unauthenticated")
        store.start()
        yield
        store.stop()

    app = FastAPI(title="custom-agent skills API", lifespan=lifespan)
    auth = [Depends(require_api_key)]

    @app.post("/news/{ticker}", status_code=202, response_model=Job, dependencies=auth)
    def start_news_job(ticker: str, response: Response, body: NewsRequest | None = None):
        """Queue a fetch-and-summarize job. An identical queued/running job is reused."""
        ticker = ticker.strip().upper()
        if not _TICKER_RE.match(ticker):
            raise HTTPException(status_code=422, detail="invalid ticker")
        body = body or NewsRequest()
        job, created = store.submit("news", {"ticker": ticker, **body.model_dump()})
        if not created:
            response.status_code = 200
        response.headers["Location"] = f"/jobs/{job['id']}"
        return job

    @app.get("/jobs/{job_id}", response_model=Job, dependencies=auth)
    def get_job(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        job["progress"] = store.progress(job)
        return job

    @app.get("/jobs", response_model=list[Job], dependencies=auth)
    def list_jobs(
        status: JobStatus | None = None, limit: int = Query(50, ge=1, le=500)
    ):
        """Job history, newest first (without progress; use GET /jobs/{id} for that)."""
        return store.list(limit=limit, status=status)

    return app


def get_app() -> FastAPI:
    return create_app()
