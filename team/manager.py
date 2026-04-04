"""
Team manager: tracks agent assignments, status, and task history.

Usage as module:
    from team.manager import TeamManager
    tm = TeamManager()
    tm.set_status("adam", status="working", task="Fixing AMP NaN")
    print(tm.dashboard())

Usage as CLI:
    python -m team.manager status
    python -m team.manager set adam working "Fixing training bugs"
    python -m team.manager idle
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants -- no magic strings
# ---------------------------------------------------------------------------

AGENT_ROSTER: list[dict[str, str]] = [
    {"id": "jason",    "name": "Jason",    "role": "system-engineer",      "short": "system-eng"},
    {"id": "lara",     "name": "Lara",     "role": "ai-team-lead",         "short": "ai-lead"},
    {"id": "adam",     "name": "Adam",     "role": "ai-engineer-training", "short": "training"},
    {"id": "kyle",     "name": "Kyle",     "role": "ai-engineer-gans",     "short": "gans"},
    {"id": "cain",     "name": "Cain",     "role": "data-engineer",        "short": "data"},
    {"id": "rona",     "name": "Rona",     "role": "frontend-server",      "short": "fe-server"},
    {"id": "pierce",   "name": "Pierce",   "role": "frontend-analyzer",    "short": "fe-analyz"},
    {"id": "anton",    "name": "Anton",    "role": "backend-analyzer",     "short": "be-analyz"},
    {"id": "marina",   "name": "Marina",   "role": "devops-engineer",      "short": "devops"},
    {"id": "perla",    "name": "Perla",    "role": "docs-manager",         "short": "docs"},
    {"id": "florence", "name": "Florence", "role": "git-expert",           "short": "git"},
    {"id": "jack",     "name": "Jack",     "role": "qa-expert",            "short": "qa"},
]

VALID_STATUSES = ("idle", "working", "done", "on-call", "blocked", "supervising")

DB_FILENAME = "status.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pretty_time(iso: str) -> str:
    """Return 'YYYY-MM-DD HH:MM' from an ISO timestamp."""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        return iso or ""


def _status_display(status: str) -> str:
    """Format status for the dashboard column."""
    symbols = {
        "done": "done",
        "working": "working",
        "idle": "idle",
        "on-call": "on-call",
        "blocked": "BLOCKED",
        "supervising": "supervising",
    }
    return symbols.get(status, status)


# ---------------------------------------------------------------------------
# TeamManager
# ---------------------------------------------------------------------------

class TeamManager:
    """Lightweight agent team tracker backed by a JSON file."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path(__file__).resolve().parent / DB_FILENAME
        self._path = Path(db_path)
        self._data: dict[str, Any] = self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        return self._blank_db()

    def _save(self) -> None:
        self._data["last_updated"] = _now()
        # Atomic write: write to temp file then rename
        fd, tmp = tempfile.mkstemp(
            dir=self._path.parent, suffix=".tmp", prefix=".status_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp, self._path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _blank_db(self) -> dict[str, Any]:
        agents: dict[str, Any] = {}
        for a in AGENT_ROSTER:
            agents[a["id"]] = {
                "name": a["name"],
                "role": a["role"],
                "status": "idle",
                "current_task": None,
                "assigned_at": None,
                "completed_tasks": [],
                "blocked_by": None,
                "notes": "",
            }
        return {
            "last_updated": _now(),
            "agents": agents,
            "active_tasks": [],
            "completed_tasks": [],
        }

    # -- queries ------------------------------------------------------------

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        agent_id = agent_id.lower()
        if agent_id not in self._data["agents"]:
            raise KeyError(f"Unknown agent: {agent_id}")
        return dict(self._data["agents"][agent_id])

    def get_idle(self) -> list[str]:
        return [
            aid for aid, info in self._data["agents"].items()
            if info["status"] == "idle"
        ]

    def get_by_status(self, status: str) -> list[str]:
        return [
            aid for aid, info in self._data["agents"].items()
            if info["status"] == status
        ]

    # -- mutations (auto-save) ----------------------------------------------

    def set_status(
        self,
        agent_id: str,
        *,
        status: str,
        task: str | None = None,
        blocked_by: str | None = None,
    ) -> None:
        agent_id = agent_id.lower()
        if agent_id not in self._data["agents"]:
            raise KeyError(f"Unknown agent: {agent_id}")
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Valid: {', '.join(VALID_STATUSES)}"
            )
        agent = self._data["agents"][agent_id]
        agent["status"] = status
        if task is not None:
            agent["current_task"] = task
            agent["assigned_at"] = _now()
        if status == "idle":
            agent["current_task"] = None
            agent["assigned_at"] = None
            agent["blocked_by"] = None
        if blocked_by is not None:
            agent["blocked_by"] = blocked_by
        self._save()

    def assign_task(self, agent_id: str, task: str) -> None:
        self.set_status(agent_id, status="working", task=task)

    def complete_task(self, agent_id: str, summary: str | None = None) -> None:
        agent_id = agent_id.lower()
        agent = self._data["agents"][agent_id]
        desc = summary or agent.get("current_task") or "Unspecified task"
        agent["completed_tasks"].append(
            {"task": desc, "completed_at": _now()}
        )
        agent["status"] = "done"
        agent["current_task"] = desc
        agent["assigned_at"] = None
        agent["blocked_by"] = None
        self._save()

    def add_note(self, agent_id: str, note: str) -> None:
        agent_id = agent_id.lower()
        if agent_id not in self._data["agents"]:
            raise KeyError(f"Unknown agent: {agent_id}")
        existing = self._data["agents"][agent_id]["notes"]
        if existing:
            self._data["agents"][agent_id]["notes"] = existing + "\n" + note
        else:
            self._data["agents"][agent_id]["notes"] = note
        self._save()

    def add_active_task(
        self,
        task_id: str,
        description: str,
        assigned_to: list[str],
        details: str = "",
    ) -> None:
        self._data["active_tasks"].append({
            "id": task_id,
            "description": description,
            "assigned_to": assigned_to,
            "status": "running",
            "started": _now(),
            "details": details,
        })
        self._save()

    def finish_active_task(self, task_id: str) -> None:
        remaining = []
        for t in self._data["active_tasks"]:
            if t["id"] == task_id:
                t["status"] = "completed"
                t["finished"] = _now()
                self._data["completed_tasks"].append(t)
            else:
                remaining.append(t)
        self._data["active_tasks"] = remaining
        self._save()

    # -- display ------------------------------------------------------------

    def dashboard(self) -> str:
        updated = _pretty_time(self._data.get("last_updated", ""))
        agents = self._data["agents"]
        roster_map = {a["id"]: a for a in AGENT_ROSTER}

        # Column widths
        w_short = 10
        w_name = 8
        w_status = 12
        w_task = 36

        border_w = w_short + w_name + w_status + w_task + 7  # separators

        lines: list[str] = []
        top = f"+{'-' * (border_w)}+"
        lines.append(top)
        title = f"  Audio Enhancer -- Team Status (updated {updated})"
        lines.append(f"|{title:<{border_w}}|")
        lines.append(
            f"+{'-' * (w_short + 1)}+{'-' * (w_name + 1)}"
            f"+{'-' * (w_status + 1)}+{'-' * (w_task + 1)}+"
        )
        header = (
            f"| {'Agent':<{w_short}}"
            f"| {'Name':<{w_name}}"
            f"| {'Status':<{w_status}}"
            f"| {'Current Task':<{w_task}}|"
        )
        lines.append(header)
        lines.append(
            f"+{'-' * (w_short + 1)}+{'-' * (w_name + 1)}"
            f"+{'-' * (w_status + 1)}+{'-' * (w_task + 1)}+"
        )

        counts: dict[str, int] = {}
        for aid in roster_map:
            info = agents.get(aid, {})
            short = roster_map[aid]["short"]
            name = info.get("name", roster_map[aid]["name"])
            status = info.get("status", "idle")
            task = info.get("current_task") or "--"
            if len(task) > w_task:
                task = task[: w_task - 2] + ".."
            counts[status] = counts.get(status, 0) + 1
            row = (
                f"| {short:<{w_short}}"
                f"| {name:<{w_name}}"
                f"| {_status_display(status):<{w_status}}"
                f"| {task:<{w_task}}|"
            )
            lines.append(row)

        lines.append(
            f"+{'-' * (w_short + 1)}+{'-' * (w_name + 1)}"
            f"+{'-' * (w_status + 1)}+{'-' * (w_task + 1)}+"
        )

        parts = []
        for s in ("working", "done", "idle", "on-call", "supervising", "blocked"):
            c = counts.get(s, 0)
            if c:
                parts.append(f"{s.capitalize()}: {c}")
        lines.append("  ".join(parts))

        return "\n".join(lines)

    def history(self, agent_id: str) -> str:
        agent_id = agent_id.lower()
        info = self.get_agent(agent_id)
        lines = [f"Task history for {info['name']} ({agent_id}):"]
        tasks = info.get("completed_tasks", [])
        if not tasks:
            lines.append("  (no completed tasks)")
        for i, t in enumerate(tasks, 1):
            if isinstance(t, dict):
                lines.append(f"  {i}. {t['task']}  ({_pretty_time(t.get('completed_at', ''))})")
            else:
                lines.append(f"  {i}. {t}")
        if info.get("notes"):
            lines.append(f"\nNotes:\n  {info['notes']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_USAGE = """\
Usage: python -m team.manager <command> [args]

Commands:
  status                           Show full team dashboard
  set <agent> <status> [task]      Set agent status (and optional task)
  complete <agent> [summary]       Mark agent's current task as done
  assign <agent> <task>            Assign a task (sets status to working)
  idle                             List idle agents
  idle-all                         Set all non-supervising agents to idle
  note <agent> <text>              Add a note to an agent
  history <agent>                  Show agent's completed task history
  info <agent>                     Show full agent record
"""


def _cli(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(_USAGE)
        sys.exit(0)

    tm = TeamManager()
    cmd = args[0].lower()

    if cmd == "status":
        print(tm.dashboard())

    elif cmd == "set":
        if len(args) < 3:
            print("Usage: set <agent> <status> [task]")
            sys.exit(1)
        agent, status = args[1], args[2]
        task = args[3] if len(args) > 3 else None
        tm.set_status(agent, status=status, task=task)
        print(f"{agent} -> {status}" + (f" ({task})" if task else ""))

    elif cmd == "complete":
        if len(args) < 2:
            print("Usage: complete <agent> [summary]")
            sys.exit(1)
        agent = args[1]
        summary = args[2] if len(args) > 2 else None
        tm.complete_task(agent, summary)
        print(f"{agent} -> done")

    elif cmd == "assign":
        if len(args) < 3:
            print("Usage: assign <agent> <task>")
            sys.exit(1)
        agent, task = args[1], args[2]
        tm.assign_task(agent, task)
        print(f"{agent} -> working ({task})")

    elif cmd == "idle":
        idle = tm.get_idle()
        if idle:
            print("Idle agents: " + ", ".join(idle))
        else:
            print("No idle agents.")

    elif cmd == "idle-all":
        for a in AGENT_ROSTER:
            info = tm.get_agent(a["id"])
            if info["status"] != "supervising":
                tm.set_status(a["id"], status="idle")
        print("All agents set to idle (except supervising).")

    elif cmd == "note":
        if len(args) < 3:
            print("Usage: note <agent> <text>")
            sys.exit(1)
        tm.add_note(args[1], args[2])
        print(f"Note added to {args[1]}.")

    elif cmd == "history":
        if len(args) < 2:
            print("Usage: history <agent>")
            sys.exit(1)
        print(tm.history(args[1]))

    elif cmd == "info":
        if len(args) < 2:
            print("Usage: info <agent>")
            sys.exit(1)
        info = tm.get_agent(args[1])
        print(json.dumps(info, indent=2, ensure_ascii=False))

    else:
        print(f"Unknown command: {cmd}")
        print(_USAGE)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
