#!/usr/bin/env python3
"""Multi-session agent launcher for the audio-enhancer project.

Launches multiple Claude Code sessions, each acting as a specific agent from the
team roster. Agents are fully autonomous — they read CLAUDE.md, check the project
roadmap, see what others are working on, and self-assign work. Coordination
happens through the agent tracker MCP database.

Two modes:
  - VS Code (primary): generates .vscode/tasks.json with terminal tasks and
    compound tasks for agent groups. Each agent gets a named terminal tab.
  - tmux (fallback): creates a tmux session with one pane per agent, suitable
    for headless/SSH environments.

Usage:
    python infra/launch_agents.py                           -- launch all agents
    python infra/launch_agents.py launch adam kyle jack      -- launch specific agents
    python infra/launch_agents.py launch --mode tmux        -- force tmux mode
    python infra/launch_agents.py launch --mode vscode      -- force VS Code mode
    python infra/launch_agents.py launch --directive "focus on Phase 1" adam kyle
    python infra/launch_agents.py status                    -- show agent status
    python infra/launch_agents.py stop                      -- kill tmux session
    python infra/launch_agents.py send adam "fix the loss"  -- send message to agent
    python infra/launch_agents.py logs                      -- show tracker dashboard
    python infra/launch_agents.py list                      -- list available agents
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = PROJECT_DIR / ".claude" / "agents"
VSCODE_DIR = PROJECT_DIR / ".vscode"
TASKS_JSON = VSCODE_DIR / "tasks.json"
TMUX_SESSION = "agents"

# Agent groups for compound tasks
AGENT_GROUPS: dict[str, list[str]] = {
    "ai-team": ["ai-team-lead", "ai-engineer-training", "ai-engineer-gans"],
    "frontend": ["frontend-analyzer", "frontend-server"],
    "infra": ["devops-engineer", "data-engineer"],
    "review": ["git-expert", "qa-expert", "docs-manager"],
    "core": ["system-engineer", "ai-team-lead", "ai-engineer-training",
             "ai-engineer-gans", "data-engineer"],
}


# ── Agent definition parsing ────────────────────────────────────────────────

@dataclass
class AgentDef:
    """An agent definition parsed from .claude/agents/*.md frontmatter."""
    file_stem: str       # e.g. "ai-engineer-training"
    name: str            # same as file_stem (the --agent value)
    human_name: str      # e.g. "Adam"
    description: str
    model: str           # e.g. "opus", "sonnet"
    max_turns: int

    @property
    def display(self) -> str:
        return f"{self.human_name} ({self.file_stem})"

    @property
    def task_label(self) -> str:
        """Label for VS Code task."""
        return f"Agent: {self.human_name}"

    @property
    def task_id(self) -> str:
        """Unique identifier for VS Code task."""
        return f"agent-{self.file_stem}"


def parse_agent_frontmatter(md_path: Path) -> AgentDef | None:
    """Parse YAML frontmatter from an agent .md file.

    Frontmatter is delimited by --- lines. We parse it manually to avoid
    requiring PyYAML as a dependency.
    """
    text = md_path.read_text()
    lines = text.splitlines()

    if not lines or lines[0].strip() != "---":
        return None

    # Find closing ---
    end_idx = None
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return None

    # Parse key: value pairs from frontmatter
    fm: dict[str, str] = {}
    for line in lines[1:end_idx]:
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip().strip('"').strip("'")

    # Extract human name from description
    description = fm.get("description", "")
    human_name = _extract_human_name(description, md_path.stem)

    return AgentDef(
        file_stem=md_path.stem,
        name=fm.get("name", md_path.stem),
        human_name=human_name,
        description=description,
        model=fm.get("model", "sonnet"),
        max_turns=int(fm.get("maxTurns", "20")),
    )


def _extract_human_name(description: str, fallback: str) -> str:
    """Extract human name from agent description string.

    Examples:
        "AI engineer Adam. Specializes in..." -> "Adam"
        "System engineer Jason. Supervises..." -> "Jason"
        "QA expert Jack. Specializes in..." -> "Jack"
    """
    first_sentence = description.split(".")[0] if description else ""
    words = first_sentence.split()
    if len(words) >= 2:
        candidate = words[-1].strip(".,;:")
        if candidate and candidate[0].isupper() and candidate.isalpha():
            return candidate
    return fallback


def load_all_agents() -> dict[str, AgentDef]:
    """Load all agent definitions from .claude/agents/*.md.

    Returns a dict keyed by file_stem (e.g. "ai-engineer-training").
    """
    agents: dict[str, AgentDef] = {}
    if not AGENTS_DIR.is_dir():
        print(f"Error: agents directory not found: {AGENTS_DIR}", file=sys.stderr)
        sys.exit(1)

    for md_file in sorted(AGENTS_DIR.glob("*.md")):
        agent = parse_agent_frontmatter(md_file)
        if agent:
            agents[agent.file_stem] = agent

    return agents


def resolve_agent_names(requested: list[str], all_agents: dict[str, AgentDef]) -> list[AgentDef]:
    """Resolve user-provided names to AgentDef objects.

    Accepts: file_stem ("ai-engineer-training"), human name ("adam", "Adam"),
    partial match ("training", "gans"), or group name ("ai-team", "frontend").
    """
    # Expand group names first
    expanded: list[str] = []
    for name in requested:
        key = name.lower().strip()
        if key in AGENT_GROUPS:
            expanded.extend(AGENT_GROUPS[key])
        else:
            expanded.append(name)

    # Build lookup index
    lookup: dict[str, AgentDef] = {}
    for agent in all_agents.values():
        lookup[agent.file_stem] = agent
        lookup[agent.human_name.lower()] = agent
        # Also index by last part of slug for convenience
        parts = agent.file_stem.split("-")
        if len(parts) > 1:
            lookup[parts[-1]] = agent

    resolved: list[AgentDef] = []
    seen: set[str] = set()

    for name in expanded:
        key = name.lower().strip()
        if key in lookup:
            agent = lookup[key]
            if agent.file_stem not in seen:
                resolved.append(agent)
                seen.add(agent.file_stem)
        else:
            # Try substring match
            matches = [a for a in all_agents.values()
                       if key in a.file_stem or key in a.human_name.lower()]
            if len(matches) == 1:
                agent = matches[0]
                if agent.file_stem not in seen:
                    resolved.append(agent)
                    seen.add(agent.file_stem)
            elif len(matches) > 1:
                names = ", ".join(m.display for m in matches)
                print(f"Error: '{name}' is ambiguous, matches: {names}", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"Error: no agent found matching '{name}'", file=sys.stderr)
                available = ", ".join(
                    f"{a.human_name} ({a.file_stem})" for a in all_agents.values()
                )
                print(f"Available agents: {available}", file=sys.stderr)
                sys.exit(1)

    return resolved


# ── Environment detection ────────────────────────────────────────────────────

def detect_mode() -> str:
    """Detect launch mode.

    Defaults to tmux — it actually launches agents directly. VS Code mode
    only generates tasks.json config (requires manual clicks to start).
    Use --mode vscode explicitly if you want that.
    """
    return "tmux"


# ── Bootstrap prompt ─────────────────────────────────────────────────────────

def build_bootstrap_prompt(agent: AgentDef, directive: str | None = None) -> str:
    """Build the autonomous bootstrap prompt for an agent.

    Agents are self-directed: they read the project state, check what others
    are doing, and decide what to work on based on their domain expertise.
    The agent tracker MCP is the coordination layer between all agents.
    """
    parts = [
        f"You are {agent.human_name}. You have just been launched as part of the "
        f"autonomous multi-agent team for the audio-enhancer project.",
        "",
        "## Startup sequence",
        "",
        "1. Register with the agent tracker MCP (agent_register)",
        "2. Set your status to 'working' (agent_update)",
        "3. Check for unread messages (message_unread) — act on any pending requests first",
        "4. Check for assigned tasks (task_list) — finish existing tasks before picking new work",
        "",
        "## Self-directed work",
        "",
        "If you have no pending messages or assigned tasks, you are NOT idle — you are a "
        "self-directed member of a startup team. Figure out what needs doing:",
        "",
        "5. Read CLAUDE.md — it has the project roadmap, current status, and next steps",
        "6. Check the agent tracker dashboard (agent_dashboard) — see what other agents "
        "are already working on so you don't duplicate effort",
        "7. Check recent git log to understand what was done recently",
        "8. Based on YOUR domain expertise and the project's needs, identify the highest-impact "
        "work you can do right now",
        "9. Create a task for yourself (task_create) describing what you'll do",
        "10. Announce your plan via message_log so the team knows",
        "11. Do the work. Commit to the develop branch when done.",
        "",
        "## Coordination",
        "",
        "- If you need something from another agent, send them a message (message_log)",
        "- If you find a bug or issue outside your domain, create a task and assign it "
        "to the right person",
        "- When you finish a task, check for new messages and pick up the next most "
        "impactful thing — don't wait to be told",
        "- Only go idle if you have genuinely exhausted all useful work in your domain "
        "AND verified via message_history and task_list that nothing is pending",
        "",
        "The project directory is /home/stas/audio-enhancer. "
        "Always work on the develop branch.",
    ]

    if directive:
        parts.extend([
            "",
            "## Team directive from the founder",
            "",
            f"{directive}",
            "",
            "This directive takes priority. Align your self-directed work with it.",
        ])

    return "\n".join(parts)


# ── VS Code mode ────────────────────────────────────────────────────────────

def build_vscode_tasks(agents: list[AgentDef], directive: str | None = None) -> dict:
    """Build a VS Code tasks.json structure with one task per agent,
    plus compound tasks for agent groups.

    Each agent task opens a dedicated VS Code terminal tab running
    `claude --agent <name>` in interactive mode.
    """
    tasks_list: list[dict] = []

    for agent in agents:
        prompt = build_bootstrap_prompt(agent, directive)
        agent_task: dict = {
            "label": agent.task_label,
            "type": "shell",
            "command": "claude",
            "args": [
                "--agent", agent.name,
                "--name", agent.human_name,
                "--append-system-prompt",
                prompt,
            ],
            "options": {
                "cwd": str(PROJECT_DIR),
            },
            "isBackground": True,
            "presentation": {
                "reveal": "always",
                "panel": "dedicated",
                "group": "agents",
                "close": False,
            },
            "problemMatcher": [],
        }
        tasks_list.append(agent_task)

    # Build compound tasks for groups
    all_agent_task_labels = [a.task_label for a in agents]

    # "Launch All Agents" compound
    compound_all: dict = {
        "label": "Launch All Agents",
        "dependsOn": all_agent_task_labels,
        "dependsOrder": "parallel",
        "problemMatcher": [],
    }
    tasks_list.append(compound_all)

    # Build compound tasks for predefined groups (only if agents in the group
    # are part of the current launch set)
    all_stems = {a.file_stem for a in agents}
    all_agents_map = {a.file_stem: a for a in agents}

    for group_name, group_stems in AGENT_GROUPS.items():
        group_agents = [all_agents_map[s] for s in group_stems if s in all_stems]
        if len(group_agents) >= 2:
            pretty_name = group_name.replace("-", " ").title()
            compound: dict = {
                "label": f"Launch {pretty_name}",
                "dependsOn": [a.task_label for a in group_agents],
                "dependsOrder": "parallel",
                "problemMatcher": [],
            }
            tasks_list.append(compound)

    return {
        "version": "2.0.0",
        "tasks": tasks_list,
    }


def write_vscode_tasks(tasks_json_data: dict) -> Path:
    """Write or merge tasks into .vscode/tasks.json.

    If the file already exists, we preserve non-agent tasks and replace
    only the agent-related entries (those whose labels start with "Agent:"
    or "Launch ").
    """
    VSCODE_DIR.mkdir(parents=True, exist_ok=True)

    existing_tasks: list[dict] = []
    if TASKS_JSON.exists():
        try:
            existing = json.loads(TASKS_JSON.read_text())
            existing_tasks = existing.get("tasks", [])
        except (json.JSONDecodeError, KeyError):
            pass

    # Separate agent tasks from user tasks
    agent_prefixes = ("Agent: ", "Launch ")
    user_tasks = [
        t for t in existing_tasks
        if not any(t.get("label", "").startswith(p) for p in agent_prefixes)
    ]

    # Merge: user tasks first, then new agent tasks
    merged = {
        "version": "2.0.0",
        "tasks": user_tasks + tasks_json_data["tasks"],
    }

    TASKS_JSON.write_text(json.dumps(merged, indent=2) + "\n")
    return TASKS_JSON


def launch_vscode(agents: list[AgentDef], directive: str | None = None) -> None:
    """Launch agents in VS Code terminals via tasks.json.

    Generates .vscode/tasks.json with per-agent terminal tasks, then instructs
    the user how to run them. We also attempt to trigger the task directly via
    the VS Code CLI if available.
    """
    print(f"Generating VS Code tasks for {len(agents)} agent(s)...")
    print()

    tasks_data = build_vscode_tasks(agents, directive)
    tasks_path = write_vscode_tasks(tasks_data)

    for agent in agents:
        print(f"  [{agent.human_name:>10}] {agent.file_stem} ({agent.model})")

    print()
    print(f"Tasks written to: {tasks_path}")
    print()

    # List available compound tasks
    compound_labels = [
        t["label"] for t in tasks_data["tasks"]
        if "dependsOn" in t
    ]
    if compound_labels:
        print("Compound tasks (launch groups):")
        for label in compound_labels:
            print(f"  - {label}")
        print()

    print("To launch agents in VS Code:")
    print()
    print("  1. Open Command Palette (Ctrl+Shift+P)")
    print("  2. Type 'Tasks: Run Task'")
    print("  3. Select 'Launch All Agents' (or a specific agent/group)")
    print()
    print("Each agent will open in its own named terminal tab in the")
    print("VS Code terminal panel. Switch tabs to interact with each agent.")
    print()
    print("Alternative: launch a single agent directly:")
    for agent in agents[:3]:
        print(f"    claude --agent {agent.name} --name '{agent.human_name}'")
    if len(agents) > 3:
        print(f"    ... and {len(agents) - 3} more")
    print()
    print("Status:  python infra/launch_agents.py status")
    print("Logs:    python infra/launch_agents.py logs")


# ── tmux mode ────────────────────────────────────────────────────────────────

def check_tmux() -> str:
    """Check that tmux is available and return its path."""
    tmux_path = shutil.which("tmux")
    if not tmux_path:
        print("Error: tmux is not installed.", file=sys.stderr)
        print("Install it with: sudo apt install tmux", file=sys.stderr)
        sys.exit(1)
    return tmux_path


def tmux_session_exists() -> bool:
    """Check if the agents tmux session already exists."""
    result = subprocess.run(
        ["tmux", "has-session", "-t", TMUX_SESSION],
        capture_output=True,
    )
    return result.returncode == 0


def kill_tmux_session() -> bool:
    """Kill the agents tmux session if it exists."""
    if not tmux_session_exists():
        print(f"No tmux session '{TMUX_SESSION}' found.")
        return False
    subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION], check=True)
    print(f"Killed tmux session '{TMUX_SESSION}'.")
    return True


def launch_tmux(agents: list[AgentDef], directive: str | None = None,
                layout: str = "tiled") -> None:
    """Launch agents in a tmux session with one window per agent.

    Each agent gets a full-screen tmux window (tab) so you can monitor them
    individually. Switch between agents with Ctrl+B + n/p or Ctrl+B + <number>.

    Args:
        agents: List of agent definitions to launch.
        directive: Optional high-level directive to steer agents' autonomous work.
        layout: Unused, kept for CLI compat.
    """
    check_tmux()

    if tmux_session_exists():
        print(f"tmux session '{TMUX_SESSION}' already exists.", file=sys.stderr)
        print("Use 'stop' to kill it first, or 'status' to inspect it.", file=sys.stderr)
        sys.exit(1)

    if not agents:
        print("No agents to launch.", file=sys.stderr)
        sys.exit(1)

    print(f"Launching {len(agents)} agent(s) in tmux session '{TMUX_SESSION}'...")
    print()

    # Create session with first agent as window 0
    first = agents[0]
    first_cmd = _build_tmux_agent_cmd(first, directive)
    print(f"  [0]  {first.human_name:<12} {first.file_stem} ({first.model})")

    subprocess.run([
        "tmux", "new-session",
        "-d", "-s", TMUX_SESSION,
        "-n", first.human_name,
        "-x", "220", "-y", "50",
    ], check=True)

    subprocess.run([
        "tmux", "send-keys", "-t", f"{TMUX_SESSION}:0",
        f"cd {PROJECT_DIR} && {first_cmd}", "Enter",
    ], check=True)

    # Create a new window for each remaining agent
    for i, agent in enumerate(agents[1:], 1):
        cmd = _build_tmux_agent_cmd(agent, directive)
        print(f"  [{i}]  {agent.human_name:<12} {agent.file_stem} ({agent.model})")

        subprocess.run([
            "tmux", "new-window", "-t", TMUX_SESSION,
            "-n", agent.human_name,
        ], check=True)

        subprocess.run([
            "tmux", "send-keys", "-t", f"{TMUX_SESSION}:{i}",
            f"cd {PROJECT_DIR} && {cmd}", "Enter",
        ], check=True)

    # Select window 0 (Jason or first agent)
    subprocess.run([
        "tmux", "select-window", "-t", f"{TMUX_SESSION}:0",
    ], check=True)

    # Show window names in status bar
    subprocess.run([
        "tmux", "set-option", "-t", TMUX_SESSION,
        "status-left-length", "30",
    ], check=True)

    print()
    print(f"All {len(agents)} agents launched in tmux session '{TMUX_SESSION}'.")
    print()
    print("  tmux attach -t agents")
    print()
    print("Navigation:")
    print("  Ctrl+B n/p    -- next/prev agent")
    print("  Ctrl+B <num>  -- jump to agent by window number")
    print("  Ctrl+B w      -- list all agent windows")
    print("  Ctrl+B d      -- detach (agents keep running)")


def _build_tmux_agent_cmd(agent: AgentDef, directive: str | None = None) -> str:
    """Build the claude CLI command string for a tmux window.

    The bootstrap prompt is passed as the initial user message (positional arg)
    so the agent starts working immediately without waiting for input.
    """
    prompt = build_bootstrap_prompt(agent, directive)
    escaped = prompt.replace("'", "'\\''")
    cmd_parts = [
        "claude",
        f"--agent {agent.name}",
        f"--name '{agent.human_name}'",
        "--dangerously-skip-permissions",
        f"'{escaped}'",
    ]
    return " ".join(cmd_parts)


# ── Status / logs / send (shared between modes) ─────────────────────────────

def show_status(mode: str) -> None:
    """Show the status of running agents."""
    # Always show agent tracker status
    _show_tracker_status()

    # Mode-specific status
    if mode == "tmux" or (mode == "auto" and shutil.which("tmux")):
        _show_tmux_status()
    elif mode == "vscode" or mode == "auto":
        _show_vscode_status()


def _show_tracker_status() -> None:
    """Show agent tracker dashboard."""
    try:
        result = subprocess.run(
            [str(PROJECT_DIR / ".venv" / "bin" / "python"), "-c",
             "from db import get_all_agents; import json; print(json.dumps(get_all_agents(), indent=2))"],
            capture_output=True, text=True,
            cwd=str(PROJECT_DIR / ".claude" / "mcp" / "agent_tracker"),
        )
        if result.returncode == 0 and result.stdout.strip():
            agents_data = json.loads(result.stdout)
            if agents_data:
                print("Agent Tracker Status:")
                print()
                print(f"  {'Name':<14} {'Status':<12} {'Turns':<8} {'Current Task'}")
                print(f"  {'----':<14} {'------':<12} {'-----':<8} {'------------'}")
                for a in agents_data:
                    turns = f"{a['turns_used']}/{a['max_turns']}"
                    task_str = a.get("current_task") or "-"
                    if len(task_str) > 50:
                        task_str = task_str[:47] + "..."
                    print(f"  {a['name']:<14} {a['status']:<12} {turns:<8} {task_str}")
                print()
            else:
                print("No agents registered in tracker yet.")
                print()
    except Exception:
        print("(Agent tracker not accessible)")
        print()


def _show_tmux_status() -> None:
    """Show tmux pane status."""
    if not shutil.which("tmux"):
        return

    result = subprocess.run(
        ["tmux", "has-session", "-t", TMUX_SESSION],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"No tmux session '{TMUX_SESSION}' found.")
        return

    result = subprocess.run(
        ["tmux", "list-panes", "-t", f"{TMUX_SESSION}:0",
         "-F", "#{pane_index}\t#{pane_title}\t#{pane_width}x#{pane_height}\t#{pane_pid}\t#{pane_current_command}"],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        return

    print(f"tmux session: {TMUX_SESSION}")
    print()
    print(f"  {'Pane':<6} {'Agent':<14} {'Size':<14} {'PID':<8} {'Command'}")
    print(f"  {'----':<6} {'-----':<14} {'----':<14} {'---':<8} {'-------'}")

    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 5:
            idx, title, size, pid, cmd = parts[0], parts[1], parts[2], parts[3], parts[4]
            print(f"  {idx:<6} {title:<14} {size:<14} {pid:<8} {cmd}")
    print()


def _show_vscode_status() -> None:
    """Show VS Code tasks.json agent configuration."""
    if not TASKS_JSON.exists():
        print("No .vscode/tasks.json found. Run 'launch' to generate it.")
        return

    try:
        data = json.loads(TASKS_JSON.read_text())
        tasks = data.get("tasks", [])
        agent_tasks = [t for t in tasks if t.get("label", "").startswith("Agent: ")]
        compound_tasks = [t for t in tasks if t.get("label", "").startswith("Launch ")]

        if agent_tasks:
            print("VS Code agent tasks configured:")
            print()
            for t in agent_tasks:
                label = t["label"]
                args = t.get("args", [])
                agent_name = args[1] if len(args) > 1 else "?"
                print(f"  {label:<24} (--agent {agent_name})")
            print()

        if compound_tasks:
            print("Compound tasks (launch groups):")
            for t in compound_tasks:
                deps = t.get("dependsOn", [])
                print(f"  {t['label']:<30} ({len(deps)} agents)")
            print()

        print("To run: Ctrl+Shift+P -> 'Tasks: Run Task' -> select a task")
        print()
    except (json.JSONDecodeError, KeyError):
        print("Error reading .vscode/tasks.json", file=sys.stderr)


def send_message(agent_name: str, message: str) -> None:
    """Send a message/task to a specific agent via the agent tracker."""
    all_agents = load_all_agents()
    resolved = resolve_agent_names([agent_name], all_agents)
    if not resolved:
        return

    agent = resolved[0]

    try:
        result = subprocess.run(
            [str(PROJECT_DIR / ".venv" / "bin" / "python"), "-c",
             f"""
from db import init_db, log_message, create_task
init_db()
log_message("launcher", "{agent.human_name}", {json.dumps(message)})
task = create_task({json.dumps(message)}, "{agent.human_name}")
print(f"Message sent to {agent.human_name}, task #{{task['id']}} created.")
"""],
            capture_output=True, text=True,
            cwd=str(PROJECT_DIR / ".claude" / "mcp" / "agent_tracker"),
        )
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"Error: {result.stderr.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"Error sending message: {e}", file=sys.stderr)


def show_logs() -> None:
    """Show recent agent tracker messages and tasks."""
    try:
        result = subprocess.run(
            [str(PROJECT_DIR / ".venv" / "bin" / "python"), "-c",
             """
import json
from db import get_all_agents, get_tasks, get_messages

print("=== Agents ===")
agents = get_all_agents()
if agents:
    for a in agents:
        turns = f"{a['turns_used']}/{a['max_turns']}"
        task = a.get('current_task') or '-'
        print(f"  {a['name']:<14} {a['status']:<12} {turns:<8} {task}")
else:
    print("  (none registered)")

print()
print("=== Recent Messages ===")
msgs = get_messages(limit=20)
if msgs:
    for m in msgs:
        ts = m['timestamp'][:19]
        print(f"  [{ts}] {m['from_agent']} -> {m['to_agent']}: {m['summary'][:80]}")
else:
    print("  (no messages)")

print()
print("=== Tasks ===")
tasks = get_tasks()
if tasks:
    for t in tasks:
        assignee = t.get('assigned_to') or 'unassigned'
        print(f"  #{t['id']} [{t['status']:<12}] ({assignee:<14}) {t['description'][:60]}")
else:
    print("  (no tasks)")
"""],
            capture_output=True, text=True,
            cwd=str(PROJECT_DIR / ".claude" / "mcp" / "agent_tracker"),
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error: {result.stderr.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"Error reading logs: {e}", file=sys.stderr)


def list_available_agents() -> None:
    """Print all available agents and groups."""
    all_agents = load_all_agents()
    print(f"Available agents ({len(all_agents)}):")
    print()
    print(f"  {'Name':<14} {'Agent ID':<24} {'Model':<8} {'Description'}")
    print(f"  {'----':<14} {'--------':<24} {'-----':<8} {'-----------'}")
    for agent in all_agents.values():
        desc = agent.description[:55]
        if len(agent.description) > 55:
            desc = desc[:52] + "..."
        print(f"  {agent.human_name:<14} {agent.file_stem:<24} {agent.model:<8} {desc}")

    print()
    print("Agent groups (use as shorthand in 'launch' command):")
    print()
    for group_name, group_stems in AGENT_GROUPS.items():
        agents_in_group = []
        for stem in group_stems:
            if stem in all_agents:
                agents_in_group.append(all_agents[stem].human_name)
        if agents_in_group:
            print(f"  {group_name:<16} {', '.join(agents_in_group)}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Claude Code agent sessions (VS Code terminals or tmux panes).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s                              launch all agents (auto-detect mode)
              %(prog)s launch adam kyle jack         launch specific agents
              %(prog)s launch ai-team               launch an agent group
              %(prog)s launch --mode tmux            force tmux mode
              %(prog)s launch --mode vscode          force VS Code mode
              %(prog)s launch --directive "focus on Phase 1"  steer all agents
              %(prog)s status                       show agent status
              %(prog)s stop                         kill tmux session
              %(prog)s send adam "check the losses"  send message to an agent
              %(prog)s logs                         show tracker dashboard
              %(prog)s list                         list agents and groups
        """),
    )

    parser.add_argument(
        "--mode", "-m", choices=["vscode", "tmux", "auto"], default="auto",
        help="Launch mode: vscode (VS Code terminals), tmux (terminal panes), "
             "auto (detect environment). Default: auto.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # launch
    launch_parser = subparsers.add_parser(
        "launch", help="Launch agent sessions",
    )
    launch_parser.add_argument(
        "agents", nargs="*",
        help="Agent names, human names, or group names to launch. Omit for all.",
    )
    launch_parser.add_argument(
        "--mode", "-m", choices=["vscode", "tmux", "auto"], default=None,
        help="Override launch mode (overrides top-level --mode).",
    )
    launch_parser.add_argument(
        "--directive", "-d", type=str, default=None,
        help="Optional high-level directive from the founder to steer all agents' "
             "self-directed work (e.g. 'focus on Phase 1 degradation pipeline').",
    )
    launch_parser.add_argument(
        "--layout", "-l", type=str, default="tiled",
        choices=["tiled", "even-horizontal", "even-vertical",
                 "main-horizontal", "main-vertical"],
        help="tmux pane layout (tmux mode only, default: tiled).",
    )

    # status
    subparsers.add_parser("status", help="Show agent status")

    # stop
    subparsers.add_parser("stop", help="Kill the agents tmux session")

    # send
    send_parser = subparsers.add_parser(
        "send", help="Send a message to a running agent via the tracker",
    )
    send_parser.add_argument("agent", help="Agent name (human name or slug)")
    send_parser.add_argument("message", help="Message to send (agent will pick it up via MCP)")

    # logs
    subparsers.add_parser("logs", help="Show agent tracker messages and tasks")

    # list
    subparsers.add_parser("list", help="List available agents and groups")

    args = parser.parse_args()

    # Resolve mode: subcommand --mode overrides top-level --mode
    mode = getattr(args, "mode", None) or "auto"
    if mode == "auto":
        mode = detect_mode()

    # Default to 'launch' if no subcommand given
    if args.command is None:
        args.command = "launch"
        args.agents = []
        args.directive = None
        args.layout = "tiled"

    if args.command == "launch":
        all_agents = load_all_agents()

        if args.agents:
            agents_to_launch = resolve_agent_names(args.agents, all_agents)
        else:
            agents_to_launch = list(all_agents.values())

        print(f"Mode: {mode}")
        print()

        if mode == "vscode":
            launch_vscode(agents_to_launch, directive=args.directive)
        elif mode == "tmux":
            launch_tmux(agents_to_launch, directive=args.directive, layout=args.layout)
        else:
            # Should not happen after auto-detection
            print(f"Unknown mode: {mode}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "status":
        show_status(mode)

    elif args.command == "stop":
        if mode == "tmux":
            check_tmux()
            kill_tmux_session()
        else:
            # In VS Code mode, there is no tmux session to stop, but we can
            # try to stop tmux if it was started separately
            if shutil.which("tmux"):
                result = subprocess.run(
                    ["tmux", "has-session", "-t", TMUX_SESSION],
                    capture_output=True,
                )
                if result.returncode == 0:
                    kill_tmux_session()
                else:
                    print("No tmux agent session to stop.")
                    print("To close VS Code terminal agents, right-click each")
                    print("terminal tab and select 'Kill Terminal'.")
            else:
                print("To close VS Code terminal agents, right-click each")
                print("terminal tab and select 'Kill Terminal'.")

    elif args.command == "send":
        send_message(args.agent, args.message)

    elif args.command == "logs":
        show_logs()

    elif args.command == "list":
        list_available_agents()


if __name__ == "__main__":
    main()
