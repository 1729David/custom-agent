import json
import sqlite3

import pytest

from jobs import JobStore, normalize_url

OLD_SCHEMA = """
CREATE TABLE jobs (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, params TEXT NOT NULL, status TEXT NOT NULL,
    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, result TEXT, error TEXT
);
CREATE TABLE job_articles (
    job_id TEXT NOT NULL REFERENCES jobs (id), position INTEGER NOT NULL, url TEXT NOT NULL,
    title TEXT, source TEXT, search_date TEXT, snippet TEXT,
    status TEXT NOT NULL DEFAULT 'pending', reason TEXT, published_at TEXT,
    published_at_source TEXT, retrieved_at TEXT, text TEXT, summary TEXT,
    PRIMARY KEY (job_id, url)
);
"""


def old_row(job_id, position, url, status, **extra):
    row = {"title": f"T{position}", "source": "S", "search_date": None, "snippet": None,
           "reason": None, "published_at": None, "published_at_source": None,
           "retrieved_at": None, "text": None, "summary": None}
    row.update(extra)
    cols = ["job_id", "position", "url", "status", *row]
    return cols, [job_id, position, url, status, *row.values()]


@pytest.fixture
def old_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript(OLD_SCHEMA)
    conn.execute(
        "INSERT INTO jobs (id, kind, params, status, created_at) VALUES (?, 'news', ?, 'done', ?)",
        ("job1", json.dumps({"ticker": "AAPL"}), "2026-10-01T00:00:00Z"),
    )
    rows = [
        old_row("job1", 0, "http://x/a?utm_source=t", "done", text="body a", summary="sum a",
                published_at="2026-09-20T00:00:00Z", retrieved_at="2026-10-01T01:00:00Z"),
        old_row("job1", 1, "http://x/b", "skipped", text="body b",
                reason="not primarily about AAPL", retrieved_at="2026-10-01T02:00:00Z"),
        old_row("job1", 2, "http://x/c", "skipped", reason="download failed: 403"),
        old_row("job1", 3, "http://x/d", "fetched", text="body d",
                retrieved_at="2026-10-01T03:00:00Z"),
    ]
    for cols, values in rows:
        conn.execute(
            f"INSERT INTO job_articles ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            values,
        )
    conn.commit()
    conn.close()
    return db_path


def rows(db_path, sql):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql)]
    finally:
        conn.close()


def test_old_schema_is_migrated_with_a_backup(old_db):
    JobStore({}, old_db)

    columns = [r["name"] for r in rows(old_db, "PRAGMA table_info(job_articles)")]
    assert "article_id" in columns and "text" not in columns and "summary" not in columns
    assert not rows(old_db, "SELECT 1 FROM sqlite_master WHERE name='job_articles_old'")
    assert old_db.with_name(old_db.name + ".bak").exists()

    articles = rows(old_db, "SELECT * FROM articles ORDER BY id")
    assert [a["text"] for a in articles] == ["body a", "body b", "body d"]  # c had none
    assert articles[0]["url"] == "http://x/a?utm_source=t"
    assert articles[0]["url_key"] == normalize_url("http://x/a")
    assert articles[0]["retrieved_at"] == "2026-10-01T01:00:00Z"

    analysis = {a["article_id"]: a for a in rows(old_db, "SELECT * FROM article_analysis")}
    assert analysis[1]["relevant"] == 1 and analysis[1]["summary"] == "sum a"
    assert analysis[2]["relevant"] == 0  # rejected as not about the ticker
    assert 3 not in analysis  # only fetched: nothing known yet

    links = {r["url"]: r for r in rows(old_db, "SELECT * FROM job_articles")}
    assert links["http://x/c"]["article_id"] is None  # nothing was retrieved
    assert links["http://x/a?utm_source=t"]["article_id"] == 1
    assert links["http://x/b"]["status"] == "skipped"
    assert links["http://x/b"]["reason"] == "not primarily about AAPL"


def test_migration_is_idempotent(old_db):
    JobStore({}, old_db)
    before = rows(old_db, "SELECT COUNT(*) AS n FROM articles")
    JobStore({}, old_db)
    assert rows(old_db, "SELECT COUNT(*) AS n FROM articles") == before


def test_migrated_articles_are_reusable(old_db, make_store):
    from jobs import JobContext

    store = make_store({"news": lambda c, **p: {}}, path=old_db)
    job, _ = store.submit("news", {"ticker": "AAPL"})
    cached = JobContext(store, job["id"]).cached_article("http://x/a")
    assert cached["text"] == "body a" and cached["summary"] == "sum a"
    assert cached["relevant"] is True


def test_fresh_database_needs_no_migration_or_backup(db_path):
    JobStore({}, db_path)
    assert not db_path.with_name(db_path.name + ".bak").exists()
    assert rows(db_path, "SELECT COUNT(*) AS n FROM articles") == [{"n": 0}]


def test_failed_migration_rolls_back_and_keeps_old_data(old_db):
    conn = sqlite3.connect(old_db)
    conn.execute("UPDATE jobs SET params='not json' WHERE id='job1'")
    conn.commit()
    conn.close()

    with pytest.raises(json.JSONDecodeError):
        JobStore({}, old_db)

    columns = [r["name"] for r in rows(old_db, "PRAGMA table_info(job_articles)")]
    assert "text" in columns and "article_id" not in columns  # untouched
    assert rows(old_db, "SELECT COUNT(*) AS n FROM job_articles") == [{"n": 4}]
    assert not rows(old_db, "SELECT 1 FROM sqlite_master WHERE name='articles'")
