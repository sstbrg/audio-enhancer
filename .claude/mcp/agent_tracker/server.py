"""MCP server for tracking agent team state."""

import json
from mcp.server.fastmcp import FastMCP

from db import init_db, upsert_agent, update_agent, get_agent, get_all_agents
from db import create_task, update_task, get_tasks
from db import log_message, get_messages, mark_messages_read, delete_messages
from db import create_pipeline, get_pipeline, update_pipeline_step

init_db()

mcp = FastMCP("agent-tracker")


# ── Agent tools ──────────────────────────────────────────────────────────

@mcp.tool()
def agent_register(name: str, role: str, subagent_type: str,
                   status: str = "working", max_turns: int = 10) -> str:
    """Register or update an agent in the tracker."""
    result = upsert_agent(name, role, subagent_type, status, max_turns)
    return json.dumps(result, indent=2)


@mcp.tool()
def agent_update(name: str, status: str | None = None,
                 turns_used: int | None = None,
                 current_task: str | None = None) -> str:
    """Update an agent's status, turn count, or current task."""
    kwargs = {}
    if status is not None:
        kwargs["status"] = status
    if turns_used is not None:
        kwargs["turns_used"] = turns_used
    if current_task is not None:
        kwargs["current_task"] = current_task
    if not kwargs:
        return "No fields to update."
    result = update_agent(name, **kwargs)
    return json.dumps(result, indent=2) if result else f"Agent '{name}' not found."


@mcp.tool()
def agent_status(name: str | None = None) -> str:
    """Get status of one agent (by name) or all agents if name is omitted."""
    if name:
        result = get_agent(name)
        return json.dumps(result, indent=2) if result else f"Agent '{name}' not found."
    agents = get_all_agents()
    if not agents:
        return "No agents registered."
    return json.dumps(agents, indent=2)


@mcp.tool()
def agent_dashboard() -> str:
    """Return a formatted markdown table of all agents and their current state."""
    agents = get_all_agents()
    if not agents:
        return "No agents registered."

    lines = ["| # | Name | Role | Status | Turns | Current Task |",
             "|---|------|------|--------|-------|-------------|"]
    for i, a in enumerate(agents, 1):
        turns = f"{a['turns_used']}/{a['max_turns']}"
        task = a.get("current_task") or "-"
        status = a["status"]
        if status == "exhausted":
            status = "EXHAUSTED"
        elif status == "dead":
            status = "DEAD"
        lines.append(f"| {i} | {a['name']} | {a['role']} | {status} | {turns} | {task} |")
    return "\n".join(lines)


# ── Task tools ───────────────────────────────────────────────────────────

@mcp.tool()
def task_create(description: str, assigned_to: str | None = None,
                depends_on: int | None = None) -> str:
    """Create a task, optionally assigned to an agent with a dependency."""
    result = create_task(description, assigned_to, depends_on)
    return json.dumps(result, indent=2)


@mcp.tool()
def task_update(task_id: int, status: str | None = None,
                assigned_to: str | None = None) -> str:
    """Update a task's status or assignment."""
    kwargs = {}
    if status is not None:
        kwargs["status"] = status
    if assigned_to is not None:
        kwargs["assigned_to"] = assigned_to
    if not kwargs:
        return "No fields to update."
    result = update_task(task_id, **kwargs)
    return json.dumps(result, indent=2) if result else f"Task {task_id} not found."


@mcp.tool()
def task_list(status: str | None = None, assigned_to: str | None = None) -> str:
    """List tasks, optionally filtered by status or assignee."""
    tasks = get_tasks(status, assigned_to)
    if not tasks:
        return "No tasks found."
    return json.dumps(tasks, indent=2)


# ── Message log tools ────────────────────────────────────────────────────

@mcp.tool()
def message_log(from_agent: str, to_agent: str, summary: str) -> str:
    """Log a message between agents for audit trail."""
    result = log_message(from_agent, to_agent, summary)
    return json.dumps(result, indent=2)


@mcp.tool()
def message_history(limit: int = 20) -> str:
    """Get recent message history between agents."""
    msgs = get_messages(limit)
    if not msgs:
        return "No messages logged."
    return json.dumps(msgs, indent=2)


@mcp.tool()
def message_unread(agent_name: str, limit: int = 20) -> str:
    """Get unread messages for a specific agent (addressed to them or 'all')."""
    msgs = get_messages(limit=limit, unread_for=agent_name)
    if not msgs:
        return f"No unread messages for {agent_name}."
    return json.dumps(msgs, indent=2)


@mcp.tool()
def message_mark_read(agent_name: str, message_ids: list[int] | None = None) -> str:
    """Mark messages as read and delete them. Call after reading and actioning a message (task created or issue resolved)."""
    marked = mark_messages_read(agent_name, message_ids)
    # Determine which IDs to delete
    if message_ids:
        ids_to_delete = message_ids
    else:
        from db import get_conn
        conn = get_conn()
        rows = conn.execute(
            "SELECT id FROM messages WHERE to_agent=? OR to_agent='all'", (agent_name,)
        ).fetchall()
        conn.close()
        ids_to_delete = [r["id"] for r in rows]
    deleted = delete_messages(ids_to_delete) if ids_to_delete else 0
    return f"Marked {marked} messages read and deleted {deleted} for {agent_name}."


@mcp.tool()
def message_delete(message_ids: list[int]) -> str:
    """Delete messages by ID."""
    count = delete_messages(message_ids)
    return f"Deleted {count} messages."


# ── Pipeline tools ───────────────────────────────────────────────────────

@mcp.tool()
def pipeline_create(name: str, task_ids: list[int]) -> str:
    """Create a pipeline — an ordered sequence of tasks with dependencies."""
    steps = create_pipeline(name, task_ids)
    return json.dumps(steps, indent=2)


@mcp.tool()
def pipeline_status(name: str) -> str:
    """Show pipeline progress — which steps are done, current, blocked."""
    steps = get_pipeline(name)
    if not steps:
        return f"Pipeline '{name}' not found."

    lines = [f"Pipeline: {name}", ""]
    for s in steps:
        icon = {"completed": "done", "in_progress": ">>>", "pending": "...", "blocked": "BLOCKED"}.get(s["task_status"], "?")
        assignee = s.get("assigned_to") or "unassigned"
        lines.append(f"  {s['step_order']+1}. [{icon}] {s['description']} ({assignee})")
    return "\n".join(lines)


@mcp.tool()
def pipeline_step_update(step_id: int, status: str) -> str:
    """Update a pipeline step's status."""
    result = update_pipeline_step(step_id, status)
    return json.dumps(result, indent=2) if result else f"Step {step_id} not found."


if __name__ == "__main__":
    mcp.run(transport="stdio")
