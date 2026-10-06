import json
import os
import queue
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

DB_PATH = Path(os.path.expanduser(os.environ.get("JOBS_DB", "~/.custom-agent/jobs.db")))
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    params      TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT,
    result      TEXT,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS jobs_created_at ON jobs (created_at);

CREATE TABLE IF NOT EXISTS job_articles (
    job_id              TEXT NOT NULL REFERENCES jobs (id),
    position            INTEGER NOT NULL,
    url                 TEXT NOT NULL,
    title               TEXT,
    source              TEXT,
    search_date         TEXT,
    snippet             TEXT,
    status              TEXT NOT NULL DEFAULT 'pending',
    reason              TEXT,
    published_at        TEXT,
    published_at_source TEXT,
    retrieved_at        TEXT,
    text                TEXT,
    summary             TEXT,
    PRIMARY KEY (job_id, url)
);
"""

# Article states; "done" and "skipped" are terminal.
_TERMINAL = ("done", "skipped")
_ARTICLE_FIELDS = {
    "title", "source", "search_date", "snippet", "status", "reason",
    "published_at", "published_at_source", "retrieved_at", "text", "summary",
}
_PROGRESS_COLUMNS = "url, title, source, status, reason, published_at, retrieved_at"

_JSON_COLUMNS = ("params", "result")


def _now() -> str:
    return datetime.now(timezone.utc).strftime(_TS_FORMAT)


def _row_to_dict(row: sqlite3.Row) -> dict:
    job = dict(row)
    for col in _JSON_COLUMNS:
        if job[col] is not None:
            job[col] = json.loads(job[col])
    return job


class JobContext:
    """What a runner gets: the job's per-article rows, readable and updatable."""

    def __init__(self, store: "JobStore", job_id: str):
        self._store = store
        self.job_id = job_id

    def articles(self) -> list[dict]:
        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM job_articles WHERE job_id=? ORDER BY position",
                (self.job_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def add_articles(self, rows: list[dict]) -> None:
        """Insert candidate articles (duplicate URLs are ignored), in order."""
        with self._store._connect() as conn:
            start = conn.execute(
                "SELECT COUNT(*) FROM job_articles WHERE job_id=?", (self.job_id,)
            ).fetchone()[0]
            for i, row in enumerate(rows, start=start):
                conn.execute(
                    "INSERT OR IGNORE INTO job_articles "
                    "(job_id, position, url, title, source, search_date, snippet) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (self.job_id, i, row["url"], row.get("title"), row.get("source"),
                     row.get("search_date"), row.get("snippet")),
                )

    def update_article(self, url: str, **fields) -> None:
        unknown = set(fields) - _ARTICLE_FIELDS
        if unknown:
            raise ValueError(f"unknown article fields: {sorted(unknown)}")
        assignments = ", ".join(f"{k}=?" for k in fields)
        with self._store._connect() as conn:
            conn.execute(
                f"UPDATE job_articles SET {assignments} WHERE job_id=? AND url=?",
                (*fields.values(), self.job_id, url),
            )


class JobStore:
    """SQLite-backed job history plus a single worker thread that runs the jobs.

    One worker means one job at a time, which protects Ollama. Jobs are never
    purged, so the table doubles as a record of what ran when. A runner is called
    as runner(ctx, **params) with a JobContext; because it persists its progress
    through ctx, a job interrupted by a restart is simply run again and resumes.
    """

    def __init__(
        self,
        runners: dict[str, Callable[..., dict]],
        db_path: Path = DB_PATH,
    ):
        self.runners = runners
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_lock = threading.Lock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        self._recover()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _recover(self) -> None:
        """After a restart: re-queue jobs that were mid-run or never started."""
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET status='queued' WHERE status='running'")
            queued = conn.execute(
                "SELECT id FROM jobs WHERE status='queued' ORDER BY created_at"
            ).fetchall()
        for row in queued:
            self._queue.put(row["id"])

    def start(self) -> None:
        self._thread = threading.Thread(target=self._work, name="job-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, kind: str, params: dict) -> tuple[dict, bool]:
        """Queue a job. Returns (job, created); an identical active job is reused."""
        if kind not in self.runners:
            raise ValueError(f"unknown job kind: {kind}")
        params_json = json.dumps(params, sort_keys=True)
        with self._create_lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM jobs WHERE kind=? AND params=? "
                "AND status IN ('queued','running') ORDER BY created_at LIMIT 1",
                (kind, params_json),
            ).fetchone()
            if existing:
                return _row_to_dict(existing), False
            job_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO jobs (id, kind, params, status, created_at) "
                "VALUES (?, ?, ?, 'queued', ?)",
                (job_id, kind, params_json, _now()),
            )
        self._queue.put(job_id)
        return self.get(job_id), True

    def get(self, job_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def progress(self, job: dict) -> dict | None:
        """Stage, per-status article counts and per-article state for a job."""
        if job["status"] == "queued":
            return {"stage": "queued", "total": 0, "counts": {}, "articles": []}
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_PROGRESS_COLUMNS} FROM job_articles "
                "WHERE job_id=? ORDER BY position",
                (job["id"],),
            ).fetchall()
        articles = [dict(r) for r in rows]
        counts: dict[str, int] = {}
        for a in articles:
            counts[a["status"]] = counts.get(a["status"], 0) + 1
        if job["status"] in ("done", "failed"):
            stage = job["status"]
        elif not articles:
            stage = "listing"
        elif any(a["status"] not in _TERMINAL for a in articles):
            stage = "processing"
        else:
            stage = "bundling"
        return {"stage": stage, "total": len(articles), "counts": counts, "articles": articles}

    def list(self, limit: int = 50, status: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM jobs", []
        if status:
            sql += " WHERE status=?"
            args.append(status)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(limit)
        with self._connect() as conn:
            return [_row_to_dict(r) for r in conn.execute(sql, args).fetchall()]

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            job = self.get(job_id)
            if not job or job["status"] != "queued":
                continue
            with self._connect() as conn:
                conn.execute(
                    "UPDATE jobs SET status='running', started_at=? WHERE id=?",
                    (_now(), job_id),
                )
            try:
                result = self.runners[job["kind"]](JobContext(self, job_id), **job["params"])
                status, result_json, error = "done", json.dumps(result), None
            except Exception as e:
                status, result_json, error = "failed", None, f"{type(e).__name__}: {e}"
            with self._connect() as conn:
                conn.execute(
                    "UPDATE jobs SET status=?, finished_at=?, result=?, error=? WHERE id=?",
                    (status, _now(), result_json, error, job_id),
                )
