"""SQLite database layer for agent tracking."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "agent_tracker.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agents (
            name TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            subagent_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'idle',
            max_turns INTEGER NOT NULL DEFAULT 10,
            turns_used INTEGER NOT NULL DEFAULT 0,
            current_task TEXT,
            spawned_at TEXT NOT NULL,
            last_activity_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            assigned_to TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            depends_on INTEGER,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            FOREIGN KEY (assigned_to) REFERENCES agents(name),
            FOREIGN KEY (depends_on) REFERENCES tasks(id)
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_agent TEXT NOT NULL,
            to_agent TEXT NOT NULL,
            summary TEXT,
            timestamp TEXT NOT NULL,
            read_by TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS pipeline_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pipeline_name TEXT NOT NULL,
            step_order INTEGER NOT NULL,
            task_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            FOREIGN KEY (task_id) REFERENCES tasks(id)
        );
    """)
    conn.commit()
    conn.close()


# ── Agent operations ─────────────────────────────────────────────────────

def upsert_agent(name: str, role: str, subagent_type: str,
                 status: str = "idle", max_turns: int = 10) -> dict:
    conn = get_conn()
    now = _now()
    conn.execute("""
        INSERT INTO agents (name, role, subagent_type, status, max_turns, spawned_at, last_activity_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            role=excluded.role, subagent_type=excluded.subagent_type,
            status=excluded.status, max_turns=excluded.max_turns,
            last_activity_at=excluded.last_activity_at
    """, (name, role, subagent_type, status, max_turns, now, now))
    conn.commit()
    row = conn.execute("SELECT * FROM agents WHERE name=?", (name,)).fetchone()
    conn.close()
    return dict(row)


def update_agent(name: str, **kwargs: Any) -> dict | None:
    conn = get_conn()
    kwargs["last_activity_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in kwargs)
    vals = list(kwargs.values()) + [name]
    conn.execute(f"UPDATE agents SET {sets} WHERE name=?", vals)
    conn.commit()
    row = conn.execute("SELECT * FROM agents WHERE name=?", (name,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_agent(name: str) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM agents WHERE name=?", (name,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_agents() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM agents ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Task operations ──────────────────────────────────────────────────────

def create_task(description: str, assigned_to: str | None = None,
                depends_on: int | None = None) -> dict:
    conn = get_conn()
    now = _now()
    cur = conn.execute("""
        INSERT INTO tasks (description, assigned_to, status, depends_on, created_at)
        VALUES (?, ?, 'pending', ?, ?)
    """, (description, assigned_to, depends_on, now))
    conn.commit()
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (cur.lastrowid,)).fetchone()
    conn.close()
    return dict(row)


def update_task(task_id: int, **kwargs: Any) -> dict | None:
    conn = get_conn()
    if kwargs.get("status") == "completed":
        kwargs["completed_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in kwargs)
    vals = list(kwargs.values()) + [task_id]
    conn.execute(f"UPDATE tasks SET {sets} WHERE id=?", vals)
    conn.commit()
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_tasks(status: str | None = None, assigned_to: str | None = None) -> list[dict]:
    conn = get_conn()
    query = "SELECT * FROM tasks WHERE 1=1"
    params: list[Any] = []
    if status:
        query += " AND status=?"
        params.append(status)
    if assigned_to:
        query += " AND assigned_to=?"
        params.append(assigned_to)
    query += " ORDER BY id"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Message log ──────────────────────────────────────────────────────────

def log_message(from_agent: str, to_agent: str, summary: str) -> dict:
    conn = get_conn()
    now = _now()
    cur = conn.execute("""
        INSERT INTO messages (from_agent, to_agent, summary, timestamp, read_by)
        VALUES (?, ?, ?, ?, '')
    """, (from_agent, to_agent, summary, now))
    conn.commit()
    row = conn.execute("SELECT * FROM messages WHERE id=?", (cur.lastrowid,)).fetchone()
    conn.close()
    return dict(row)


def get_messages(limit: int = 20, unread_for: str | None = None) -> list[dict]:
    conn = get_conn()
    if unread_for:
        rows = conn.execute(
            """SELECT * FROM messages
               WHERE (to_agent=? OR to_agent='all')
               AND (read_by NOT LIKE ?)
               ORDER BY id DESC LIMIT ?""",
            (unread_for, f"%{unread_for}%", limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_messages(message_ids: list[int]) -> int:
    """Delete messages by ID (call after marking read and actioning)."""
    conn = get_conn()
    placeholders = ",".join("?" * len(message_ids))
    conn.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", message_ids)
    count = conn.total_changes
    conn.commit()
    conn.close()
    return count


def mark_messages_read(agent_name: str, message_ids: list[int] | None = None) -> int:
    """Mark messages as read by agent_name. Pass message_ids=None to mark all addressed to them."""
    conn = get_conn()
    if message_ids:
        placeholders = ",".join("?" * len(message_ids))
        rows = conn.execute(
            f"SELECT id, read_by FROM messages WHERE id IN ({placeholders})", message_ids
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, read_by FROM messages WHERE to_agent=? OR to_agent='all'",
            (agent_name,)
        ).fetchall()
    count = 0
    for row in rows:
        current = row["read_by"] or ""
        if agent_name not in current.split(","):
            new_read_by = f"{current},{agent_name}".strip(",")
            conn.execute("UPDATE messages SET read_by=? WHERE id=?", (new_read_by, row["id"]))
            count += 1
    conn.commit()
    conn.close()
    return count


# ── Pipeline operations ──────────────────────────────────────────────────

def create_pipeline(name: str, task_ids: list[int]) -> list[dict]:
    conn = get_conn()
    steps = []
    for order, tid in enumerate(task_ids):
        cur = conn.execute("""
            INSERT INTO pipeline_steps (pipeline_name, step_order, task_id, status)
            VALUES (?, ?, ?, 'pending')
        """, (name, order, tid))
        steps.append(cur.lastrowid)
    conn.commit()
    rows = conn.execute(
        "SELECT * FROM pipeline_steps WHERE pipeline_name=? ORDER BY step_order",
        (name,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_pipeline(name: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute("""
        SELECT ps.*, t.description, t.assigned_to, t.status as task_status
        FROM pipeline_steps ps
        JOIN tasks t ON ps.task_id = t.id
        WHERE ps.pipeline_name=?
        ORDER BY ps.step_order
    """, (name,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_pipeline_step(step_id: int, status: str) -> dict | None:
    conn = get_conn()
    conn.execute("UPDATE pipeline_steps SET status=? WHERE id=?", (status, step_id))
    conn.commit()
    row = conn.execute("SELECT * FROM pipeline_steps WHERE id=?", (step_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
