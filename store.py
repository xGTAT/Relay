"""SQLite persistence for Relay: chat memory, reminders/deadlines, course documents.

One small file-backed database (default relay.db). Safe to call from the event
loop and from worker threads: a single connection guarded by a lock.
"""

import re
import sqlite3
import threading
import time
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    fire_at REAL NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status, fire_at);

CREATE TABLE IF NOT EXISTS deadlines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    title TEXT NOT NULL,
    due_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    reminder_id INTEGER,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_deadlines_user ON deadlines(user_id, status, due_at);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    pages INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
    text, user_id UNINDEXED, doc_id UNINDEXED, doc_name UNINDEXED, page UNINDEXED
);
"""


class Store:
    def __init__(self, path: str = "relay.db"):
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    # ----- chat memory -----

    def add_message(self, user_id: str, role: str, content: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO messages (user_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (user_id, role, content, time.time()),
            )
            self._db.commit()

    def get_messages(self, user_id: str, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def clear_messages(self, user_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
            self._db.commit()

    # ----- reminders / deadlines -----

    def add_reminder(self, user_id: str, channel_id: str, fire_at: float, text: str) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO reminders (user_id, channel_id, fire_at, text, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, channel_id, fire_at, text, time.time()),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def pending_reminders(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM reminders WHERE status = 'pending' ORDER BY fire_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def list_reminders(
        self, user_id: str, until: Optional[float] = None, now: Optional[float] = None
    ) -> list[dict]:
        """Pending reminders for a user, soonest first; optionally only those due before `until`."""
        sql = "SELECT * FROM reminders WHERE user_id = ? AND status = 'pending'"
        args: list = [user_id]
        if until is not None:
            sql += " AND fire_at <= ?"
            args.append(until)
        sql += " ORDER BY fire_at"
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def reminder_status(self, reminder_id: int) -> Optional[str]:
        with self._lock:
            row = self._db.execute("SELECT status FROM reminders WHERE id = ?", (reminder_id,)).fetchone()
        return row["status"] if row else None

    def mark_reminder(self, reminder_id: int, status: str) -> None:
        with self._lock:
            self._db.execute("UPDATE reminders SET status = ? WHERE id = ?", (status, reminder_id))
            self._db.commit()

    def cancel_reminder(self, user_id: str, reminder_id: int) -> bool:
        with self._lock:
            cur = self._db.execute(
                "UPDATE reminders SET status = 'cancelled' "
                "WHERE id = ? AND user_id = ? AND status = 'pending'",
                (reminder_id, user_id),
            )
            self._db.commit()
            return cur.rowcount > 0

    # ----- course documents (PDF text, searched with SQLite FTS5) -----

    def add_document(self, user_id: str, name: str, pages: int, chunks: list[tuple[int, str]]) -> int:
        """Store a document and its (page, text) chunks. Re-uploading a name replaces the old copy."""
        with self._lock:
            old = self._db.execute(
                "SELECT id FROM documents WHERE user_id = ? AND name = ?", (user_id, name)
            ).fetchall()
            for row in old:
                self._db.execute("DELETE FROM chunks WHERE doc_id = ?", (row["id"],))
                self._db.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
            cur = self._db.execute(
                "INSERT INTO documents (user_id, name, pages, created_at) VALUES (?, ?, ?, ?)",
                (user_id, name, pages, time.time()),
            )
            doc_id = int(cur.lastrowid)
            self._db.executemany(
                "INSERT INTO chunks (text, user_id, doc_id, doc_name, page) VALUES (?, ?, ?, ?, ?)",
                [(text, user_id, doc_id, name, page) for page, text in chunks],
            )
            self._db.commit()
            return doc_id

    def list_documents(self, user_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, name, pages FROM documents WHERE user_id = ? ORDER BY id", (user_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _fts_query(query: str) -> str:
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))

    def search_chunks(
        self, user_id: str, query: str, limit: int = 5, doc_name: Optional[str] = None
    ) -> list[dict]:
        """Best-matching chunks for a user (BM25 ranking). Empty query returns nothing."""
        match = self._fts_query(query)
        if not match:
            return []
        sql = (
            "SELECT doc_name, page, text FROM chunks "
            "WHERE chunks MATCH ? AND user_id = ?"
        )
        args: list = [match, user_id]
        if doc_name:
            sql += " AND doc_name LIKE ?"
            args.append(f"%{doc_name}%")
        sql += " ORDER BY bm25(chunks) LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def sample_chunks(self, user_id: str, limit: int = 6, doc_name: Optional[str] = None) -> list[dict]:
        """Evenly spaced chunks across a user's documents, for quizzes with no specific topic."""
        sql = "SELECT rowid, doc_name, page, text FROM chunks WHERE user_id = ?"
        args: list = [user_id]
        if doc_name:
            sql += " AND doc_name LIKE ?"
            args.append(f"%{doc_name}%")
        sql += " ORDER BY rowid"
        with self._lock:
            rows = [dict(r) for r in self._db.execute(sql, args).fetchall()]
        if len(rows) <= limit:
            return rows
        step = len(rows) / limit
        return [rows[int(i * step)] for i in range(limit)]

    # ----- deadlines -----

    def add_deadline(
        self, user_id: str, channel_id: str, title: str, due_at: float, reminder_id: Optional[int] = None
    ) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO deadlines (user_id, channel_id, title, due_at, reminder_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, channel_id, title, due_at, reminder_id, time.time()),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def list_deadlines(self, user_id: str, until: Optional[float] = None) -> list[dict]:
        """Open deadlines, soonest first. Overdue ones are always included."""
        sql = "SELECT * FROM deadlines WHERE user_id = ? AND status = 'open'"
        args: list = [user_id]
        if until is not None:
            sql += " AND due_at <= ?"
            args.append(until)
        sql += " ORDER BY due_at"
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def complete_deadline(self, user_id: str, deadline_id: int) -> Optional[dict]:
        """Mark one of the user's open deadlines done and cancel its reminder. Returns the row or None."""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM deadlines WHERE id = ? AND user_id = ? AND status = 'open'",
                (deadline_id, user_id),
            ).fetchone()
            if not row:
                return None
            self._db.execute("UPDATE deadlines SET status = 'done' WHERE id = ?", (deadline_id,))
            if row["reminder_id"]:
                self._db.execute(
                    "UPDATE reminders SET status = 'cancelled' WHERE id = ? AND status = 'pending'",
                    (row["reminder_id"],),
                )
            self._db.commit()
        return dict(row)
