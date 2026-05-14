import json
import math
import os
import sqlite3
from datetime import datetime, timezone

import ollama

DB_DIR = os.path.expanduser("~/.custom-agent")
DB_PATH = os.path.join(DB_DIR, "memory.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content    TEXT    NOT NULL,
    tags       TEXT    NOT NULL DEFAULT '',
    source     TEXT    NOT NULL DEFAULT '',
    created_at TEXT    NOT NULL,
    embedding  TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    tags,
    content='memories',
    content_rowid='id',
    tokenize='porter ascii'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags)
    VALUES (new.id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES ('delete', old.id, old.content, old.tags);
    INSERT INTO memories_fts(rowid, content, tags)
    VALUES (new.id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES ('delete', old.id, old.content, old.tags);
END;
"""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class MemoryStore:
    def __init__(self, db_path: str = DB_PATH, embed_model: str = ""):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._embed_model = embed_model
        self._conn.executescript(_SCHEMA)

    def _get_embedding(self, text: str) -> list[float] | None:
        if not self._embed_model:
            return None
        try:
            response = ollama.embed(model=self._embed_model, input=text)
            return response.embeddings[0]
        except Exception:
            return None

    def save(self, content: str, tags: str = "", source: str = "") -> int:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        embedding = self._get_embedding(content)
        embedding_json = json.dumps(embedding) if embedding is not None else None
        cur = self._conn.execute(
            "INSERT INTO memories (content, tags, source, created_at, embedding) VALUES (?, ?, ?, ?, ?)",
            (content, tags, source, now, embedding_json),
        )
        self._conn.commit()
        return cur.lastrowid

    def search(self, query: str, limit: int = 5) -> list[sqlite3.Row]:
        limit = min(limit, 20)
        candidates_limit = limit * 3
        try:
            rows = self._conn.execute(
                """
                SELECT m.id, m.content, m.tags, m.source, m.created_at, m.embedding
                FROM memories_fts f
                JOIN memories m ON m.id = f.rowid
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, candidates_limit),
            ).fetchall()
        except Exception:
            # FTS match syntax error — fall back to recent
            rows = self.recent(candidates_limit)

        if not rows:
            return []

        query_embedding = self._get_embedding(query)
        if query_embedding is not None:
            scored = []
            for row in rows:
                if row["embedding"]:
                    stored = json.loads(row["embedding"])
                    score = _cosine_similarity(query_embedding, stored)
                else:
                    score = 0.0
                scored.append((score, row))
            scored.sort(key=lambda x: x[0], reverse=True)
            rows = [r for _, r in scored]

        return rows[:limit]

    def recent(self, limit: int = 5) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT id, content, tags, source, created_at, embedding FROM memories ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def close(self):
        self._conn.close()
