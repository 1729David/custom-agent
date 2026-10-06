import json
import os
import queue
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

DB_PATH = Path(os.path.expanduser(os.environ.get("JOBS_DB", "~/.custom-agent/jobs.db")))
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# jobs and articles are many-to-many through job_articles, which also carries the
# per-job state of each candidate. An article row is one retrieval of a URL; a
# forced refresh adds a new row. article_analysis holds the ticker-specific
# results (relevance verdict, summary) so later jobs can reuse them.
_SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS jobs (
        id          TEXT PRIMARY KEY,
        kind        TEXT NOT NULL,
        params      TEXT NOT NULL,
        status      TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        started_at  TEXT,
        finished_at TEXT,
        result      TEXT,
        error       TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS jobs_created_at ON jobs (created_at)",
    """CREATE TABLE IF NOT EXISTS articles (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        url                 TEXT NOT NULL,
        url_key             TEXT NOT NULL,
        title               TEXT,
        source              TEXT,
        published_at        TEXT,
        published_at_source TEXT,
        retrieved_at        TEXT NOT NULL,
        text                TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS articles_url_key ON articles (url_key, id)",
    """CREATE TABLE IF NOT EXISTS article_analysis (
        article_id INTEGER NOT NULL REFERENCES articles (id),
        ticker     TEXT NOT NULL,
        relevant   INTEGER,
        summary    TEXT,
        PRIMARY KEY (article_id, ticker)
    )""",
    """CREATE TABLE IF NOT EXISTS job_articles (
        job_id      TEXT NOT NULL REFERENCES jobs (id),
        position    INTEGER NOT NULL,
        url         TEXT NOT NULL,
        title       TEXT,
        source      TEXT,
        search_date TEXT,
        snippet     TEXT,
        status      TEXT NOT NULL DEFAULT 'pending',
        reason      TEXT,
        article_id  INTEGER REFERENCES articles (id),
        reused      INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (job_id, url)
    )""",
)

# Job-article states; "done" and "skipped" are terminal.
_TERMINAL = ("done", "skipped")
_JOB_ARTICLE_FIELDS = {"status", "reason", "article_id", "reused"}
_ARTICLE_COLUMNS = (
    "title", "source", "published_at", "published_at_source", "retrieved_at", "text",
)

_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ocid", "cmpid", "ref", "ref_src"}

_JSON_COLUMNS = ("params", "result")


def _now() -> str:
    return datetime.now(timezone.utc).strftime(_TS_FORMAT)


def _row_to_dict(row: sqlite3.Row) -> dict:
    job = dict(row)
    for col in _JSON_COLUMNS:
        if job[col] is not None:
            job[col] = json.loads(job[col])
    return job


def _article_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d.get("relevant") is not None:
        d["relevant"] = bool(d["relevant"])
    if "reused" in d:
        d["reused"] = bool(d["reused"])
    return d


def normalize_url(url: str) -> str:
    """Canonical form of a URL, used to recognize the same article across jobs."""
    parts = urlsplit(url.strip())
    query = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
    )
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


class JobInterrupted(Exception):
    """Raised by JobContext.checkpoint() when the job was asked to pause or cancel."""

    def __init__(self, action: str):
        super().__init__(action)
        self.action = action  # "pause" or "cancel"


class InvalidTransition(ValueError):
    """The requested pause/resume/cancel doesn't apply to the job's current state."""


class JobContext:
    """What a runner gets: this job's candidate articles, plus the shared article store.

    articles() returns each candidate joined with its retrieved article (text,
    dates) and its analysis for this job's ticker, when those exist.
    """

    def __init__(self, store: "JobStore", job_id: str):
        self._store = store
        self.job_id = job_id
        job = store.get(job_id)
        self.ticker = (job["params"] or {}).get("ticker") if job else None

    def checkpoint(self) -> None:
        """Call between steps; raises JobInterrupted if a pause/cancel was requested."""
        action = self._store._control.get(self.job_id)
        if action:
            raise JobInterrupted(action)

    def articles(self) -> list[dict]:
        with self._store._connect() as conn:
            rows = conn.execute(
                """SELECT j.url, COALESCE(a.title, j.title) AS title, j.source,
                          j.search_date, j.snippet, j.status, j.reason, j.article_id,
                          j.reused, a.published_at, a.published_at_source,
                          a.retrieved_at, a.text, an.relevant, an.summary
                   FROM job_articles j
                   LEFT JOIN articles a ON a.id = j.article_id
                   LEFT JOIN article_analysis an
                          ON an.article_id = a.id AND an.ticker = ?
                   WHERE j.job_id = ? ORDER BY j.position""",
                (self.ticker, self.job_id),
            ).fetchall()
        return [_article_row(r) for r in rows]

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
        """Update this job's state for a candidate (status, reason, article_id, reused)."""
        unknown = set(fields) - _JOB_ARTICLE_FIELDS
        if unknown:
            raise ValueError(f"unknown job-article fields: {sorted(unknown)}")
        assignments = ", ".join(f"{k}=?" for k in fields)
        with self._store._connect() as conn:
            conn.execute(
                f"UPDATE job_articles SET {assignments} WHERE job_id=? AND url=?",
                (*fields.values(), self.job_id, url),
            )

    def cached_article(self, url: str) -> dict | None:
        """The most recent retrieval of this URL by any job, with its analysis."""
        with self._store._connect() as conn:
            row = conn.execute(
                """SELECT a.id AS article_id, a.title, a.published_at,
                          a.published_at_source, a.retrieved_at, a.text,
                          an.relevant, an.summary
                   FROM articles a
                   LEFT JOIN article_analysis an
                          ON an.article_id = a.id AND an.ticker = ?
                   WHERE a.url_key = ? ORDER BY a.id DESC LIMIT 1""",
                (self.ticker, normalize_url(url)),
            ).fetchone()
        return _article_row(row) if row else None

    def store_article(self, url: str, info: dict) -> int:
        """Save a newly retrieved article (always a new row) and return its id."""
        values = [info.get(c) for c in _ARTICLE_COLUMNS]
        with self._store._connect() as conn:
            cur = conn.execute(
                "INSERT INTO articles (url, url_key, title, source, published_at, "
                "published_at_source, retrieved_at, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (url, normalize_url(url), *values),
            )
            return cur.lastrowid

    def store_analysis(
        self, article_id: int, relevant: bool | None = None, summary: str | None = None
    ) -> None:
        """Record this ticker's relevance verdict and/or summary for an article."""
        with self._store._connect() as conn:
            conn.execute(
                "INSERT INTO article_analysis (article_id, ticker, relevant, summary) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (article_id, ticker) DO UPDATE SET "
                "relevant=COALESCE(excluded.relevant, relevant), "
                "summary=COALESCE(excluded.summary, summary)",
                (article_id, self.ticker, None if relevant is None else int(relevant), summary),
            )


class JobStore:
    """SQLite-backed job history plus a single worker thread that runs the jobs.

    One worker means one job at a time, which protects Ollama. Jobs are never
    purged, so the table doubles as a record of what ran when. A runner is called
    as runner(ctx, **params) with a JobContext; because it persists its progress
    through ctx, a job interrupted by a restart is simply run again and resumes.

    Pause and cancel are cooperative: a request on a running job takes effect at
    the runner's next ctx.checkpoint(), so the step in flight finishes first.
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
        self._control_lock = threading.Lock()
        self._control: dict[str, str] = {}  # job_id -> "pause" | "cancel"
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._migrate()
        with self._connect() as conn:
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)
        self._recover()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate(self) -> None:
        """Move an old job_articles table (article data inline) to the shared schema.

        Each old row that has article text becomes an articles row (plus analysis),
        so articles retrieved by earlier jobs can be reused. The database file is
        copied aside first, and the whole change runs in one transaction.
        """
        with self._connect() as conn:
            columns = [r["name"] for r in conn.execute("PRAGMA table_info(job_articles)")]
        if not columns or "article_id" in columns:
            return
        shutil.copy2(self.db_path, self.db_path.with_name(self.db_path.name + ".bak"))
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN")
            conn.execute("ALTER TABLE job_articles RENAME TO job_articles_old")
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)
            old = conn.execute(
                "SELECT o.*, j.params FROM job_articles_old o JOIN jobs j ON j.id = o.job_id "
                "ORDER BY j.created_at, o.position"
            ).fetchall()
            for r in old:
                article_id = None
                if r["text"]:
                    article_id = conn.execute(
                        "INSERT INTO articles (url, url_key, title, source, published_at, "
                        "published_at_source, retrieved_at, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (r["url"], normalize_url(r["url"]), r["title"], r["source"],
                         r["published_at"], r["published_at_source"],
                         r["retrieved_at"] or _now(), r["text"]),
                    ).lastrowid
                    ticker = (json.loads(r["params"]) or {}).get("ticker")
                    if r["status"] in ("relevant", "done"):
                        relevant = 1
                    elif (r["reason"] or "").startswith("not primarily about"):
                        relevant = 0
                    else:
                        relevant = None
                    if ticker and (relevant is not None or r["summary"]):
                        conn.execute(
                            "INSERT OR IGNORE INTO article_analysis "
                            "(article_id, ticker, relevant, summary) VALUES (?, ?, ?, ?)",
                            (article_id, ticker, relevant, r["summary"]),
                        )
                conn.execute(
                    "INSERT INTO job_articles (job_id, position, url, title, source, "
                    "search_date, snippet, status, reason, article_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (r["job_id"], r["position"], r["url"], r["title"], r["source"],
                     r["search_date"], r["snippet"], r["status"], r["reason"], article_id),
                )
            conn.execute("DROP TABLE job_articles_old")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

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
        """Queue a job. Returns (job, created); an identical active or paused job is reused."""
        if kind not in self.runners:
            raise ValueError(f"unknown job kind: {kind}")
        params_json = json.dumps(params, sort_keys=True)
        with self._create_lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM jobs WHERE kind=? AND params=? "
                "AND status IN ('queued','running','paused') ORDER BY created_at LIMIT 1",
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

    def _set_status(self, job_id: str, status: str, error: str | None = None, finished: bool = False):
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, error=?, finished_at=? WHERE id=?",
                (status, error, _now() if finished else None, job_id),
            )

    def pause(self, job_id: str) -> dict:
        """Pause a queued job now, or a running one at its next checkpoint."""
        with self._control_lock:
            job = self._require(job_id)
            if job["status"] == "queued":
                self._set_status(job_id, "paused")
            elif job["status"] == "running":
                if self._control.get(job_id) == "cancel":
                    raise InvalidTransition("job is being cancelled")
                self._control[job_id] = "pause"
            else:
                raise InvalidTransition(f"cannot pause a {job['status']} job")
        return self.get(job_id)

    def resume(self, job_id: str) -> dict:
        with self._control_lock:
            job = self._require(job_id)
            if job["status"] != "paused":
                raise InvalidTransition(f"cannot resume a {job['status']} job")
            self._set_status(job_id, "queued")
        self._queue.put(job_id)
        return self.get(job_id)

    def cancel(self, job_id: str) -> dict:
        """Fail a queued or paused job now, or a running one at its next checkpoint."""
        with self._control_lock:
            job = self._require(job_id)
            if job["status"] in ("queued", "paused"):
                self._set_status(job_id, "failed", "cancelled by user", finished=True)
            elif job["status"] == "running":
                self._control[job_id] = "cancel"
            else:
                raise InvalidTransition(f"cannot cancel a {job['status']} job")
        return self.get(job_id)

    def _require(self, job_id: str) -> dict:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def progress(self, job: dict) -> dict | None:
        """Stage, per-status article counts and per-article state for a job.

        target is how many articles the job aims to keep (its max_articles); total
        is the candidate count, which is larger because some candidates are rejected.
        """
        target = (job["params"] or {}).get("max_articles")
        if job["status"] == "queued":
            return {"stage": "queued", "target": target, "total": 0, "counts": {}, "articles": []}
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT j.url, COALESCE(a.title, j.title) AS title, j.source,
                          j.status, j.reason, j.article_id, j.reused,
                          a.published_at, a.retrieved_at
                   FROM job_articles j LEFT JOIN articles a ON a.id = j.article_id
                   WHERE j.job_id=? ORDER BY j.position""",
                (job["id"],),
            ).fetchall()
        articles = [_article_row(r) for r in rows]
        counts: dict[str, int] = {}
        for a in articles:
            counts[a["status"]] = counts.get(a["status"], 0) + 1
        pending_action = self._control.get(job["id"]) if job["status"] == "running" else None
        if job["status"] in ("done", "failed", "paused"):
            stage = job["status"]
        elif pending_action:
            stage = "pausing" if pending_action == "pause" else "cancelling"
        elif not articles:
            stage = "listing"
        elif any(a["status"] not in _TERMINAL for a in articles):
            stage = "processing"
        else:
            stage = "bundling"
        return {
            "stage": stage,
            "target": target,
            "total": len(articles),
            "counts": counts,
            "articles": articles,
        }

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
                    "UPDATE jobs SET status='running', started_at=COALESCE(started_at, ?) WHERE id=?",
                    (_now(), job_id),
                )
            try:
                result = self.runners[job["kind"]](JobContext(self, job_id), **job["params"])
                status, result_json, error = "done", json.dumps(result), None
            except JobInterrupted as e:
                if e.action == "pause":
                    status, result_json, error = "paused", None, None
                else:
                    status, result_json, error = "failed", None, "cancelled by user"
            except Exception as e:
                status, result_json, error = "failed", None, f"{type(e).__name__}: {e}"
            with self._control_lock, self._connect() as conn:
                conn.execute(
                    "UPDATE jobs SET status=?, finished_at=?, result=?, error=? WHERE id=?",
                    (status, None if status == "paused" else _now(), result_json, error, job_id),
                )
                self._control.pop(job_id, None)
