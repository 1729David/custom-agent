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
"""

_JSON_COLUMNS = ("params", "result")


def _now() -> str:
    return datetime.now(timezone.utc).strftime(_TS_FORMAT)


def _row_to_dict(row: sqlite3.Row) -> dict:
    job = dict(row)
    for col in _JSON_COLUMNS:
        if job[col] is not None:
            job[col] = json.loads(job[col])
    return job


class JobStore:
    """SQLite-backed job history plus a single worker thread that runs the jobs.

    One worker means one job at a time, which protects Ollama and Playwright.
    Jobs are never purged, so the table doubles as a record of what ran when.
    """

    def __init__(self, runners: dict[str, Callable[..., dict]], db_path: Path = DB_PATH):
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
        """After a restart: fail jobs that were mid-run, re-queue ones never started."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='failed', finished_at=?, "
                "error='interrupted by server restart' WHERE status='running'",
                (_now(),),
            )
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
                result = self.runners[job["kind"]](**job["params"])
                status, result_json, error = "done", json.dumps(result), None
            except Exception as e:
                status, result_json, error = "failed", None, f"{type(e).__name__}: {e}"
            with self._connect() as conn:
                conn.execute(
                    "UPDATE jobs SET status=?, finished_at=?, result=?, error=? WHERE id=?",
                    (status, _now(), result_json, error, job_id),
                )
