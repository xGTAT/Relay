"""SQLite persistence for Relay: chat memory, reminders/deadlines, course documents.

One small file-backed database (default relay.db). Safe to call from the event
loop and from worker threads: a single connection guarded by a lock.
"""

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
