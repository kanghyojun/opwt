#!/usr/bin/env python3
from __future__ import annotations

import argparse
import curses
import json
import os
import re
import sqlite3
import subprocess
import textwrap
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REFRESH_INTERVAL_SECONDS = 3.0
GIT_TIMEOUT_SECONDS = 2.0
SIDEBAR_SESSION_PREVIEW = 4
DETAIL_SPLIT_MIN_WIDTH = 96
RUNNING_ANIMATION_FRAMES = (
    "⠋",
    "⠙",
    "⠹",
    "⠸",
    "⠼",
    "⠴",
    "⠦",
    "⠧",
    "⠇",
    "⠏",
)
RUNNING_ANIMATION_FRAME_SECONDS = 0.09
PERMISSION_ASK_ANIMATION_FRAMES = ("?", "!")
PERMISSION_ASK_ANIMATION_FRAME_SECONDS = 0.45

SHELL_NAMES = {
    "bash",
    "zsh",
    "fish",
    "sh",
    "dash",
    "ksh",
    "tcsh",
    "csh",
    "pwsh",
    "nu",
}

AGENT_HINTS = ("opencode", "claude", "codex", "aider")
PERMISSION_HINTS = ("permission", "approve", "approval", "allow")
OPENCODE_SESSION_ARG_RE = re.compile(
    r"(?:^|\s)(?:-s|--session)\s+(ses_[A-Za-z0-9]+)(?:\s|$)"
)
OPENCODE_NONINTERACTIVE_SUBCOMMANDS = {
    "run",
    "--version",
    "version",
    "--help",
    "help",
    "completion",
}


@dataclass
class GitStatus:
    branch: str = ""
    upstream: str = ""
    ahead: int = 0
    behind: int = 0
    staged: int = 0
    unstaged: int = 0
    dirty: bool = False
    changed: int = 0
    untracked: int = 0


@dataclass
class ProcessInfo:
    pid: int
    ppid: int
    pgrp: int
    sid: int
    tty_nr: int
    tpgid: int
    state: str
    comm: str
    cmdline: str
    cwd: str
    fd0: str


@dataclass
class Session:
    tty: str
    shell_pid: int
    foreground: ProcessInfo
    state: str  # idle | running | permission-ask
    session_id: str = ""
    session_title: str = ""
    session_directory: str = ""


@dataclass
class Worktree:
    path: str
    alias: str = ""
    is_current: bool = False
    head: str = ""
    branch_ref: str = ""
    branch: str = ""
    detached: bool = False
    locked: bool = False
    prunable: bool = False
    status: GitStatus = field(default_factory=GitStatus)
    git_error: str = ""
    sessions: List[Session] = field(default_factory=list)


class AliasStore:
    def __init__(self) -> None:
        config_root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        self.file_path = config_root / "opwt" / "aliases.json"
        self.aliases: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = self.file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.aliases = {}
            return

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.aliases = {}
            return

        aliases = data.get("aliases", {})
        if isinstance(aliases, dict):
            self.aliases = {
                normalize_path(str(path)): str(name)
                for path, name in aliases.items()
                if str(name).strip()
            }
        else:
            self.aliases = {}

    def name_for(self, worktree_path: str) -> str:
        key = normalize_path(worktree_path)
        default_name = os.path.basename(key)
        value = self.aliases.get(key, "").strip()
        return value if value else default_name

    def set(self, worktree_path: str, alias: str) -> None:
        key = normalize_path(worktree_path)
        alias = alias.strip()
        default_name = os.path.basename(key)

        if not alias or alias == default_name:
            self.aliases.pop(key, None)
        else:
            self.aliases[key] = alias

        payload = {"aliases": self.aliases}
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.file_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(self.file_path)


@dataclass
class AppState:
    base_path: str
    alias_store: AliasStore
    worktrees: List[Worktree] = field(default_factory=list)
    selected: int = 0
    table_offset: int = 0
    error: str = ""
    refreshing: bool = False
    last_refresh_at: float = 0.0
    rename_mode: bool = False
    rename_target_path: str = ""
    rename_input: str = ""
    detail_mode: bool = False


def normalize_path(path: str) -> str:
    try:
        return os.path.realpath(os.path.abspath(path))
    except OSError:
        return os.path.abspath(path)


def resolve_base_path(raw_path: Optional[str]) -> str:
    path = raw_path or "."
    resolved = normalize_path(path)
    if os.path.isdir(resolved):
        return resolved
    return normalize_path(os.path.dirname(resolved))


def run_git(path: str, args: Iterable[str]) -> str:
    command = ["git", "-C", path, *list(args)]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("git command timed out") from exc

    if completed.returncode != 0:
        err = (
            completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        )
        raise RuntimeError(err)

    return completed.stdout


def discover_worktrees(base_path: str) -> List[Worktree]:
    output = run_git(base_path, ("worktree", "list", "--porcelain"))
    lines = output.replace("\r\n", "\n").split("\n")

    worktrees: List[Worktree] = []
    current: Optional[Worktree] = None

    def flush() -> None:
        nonlocal current
        if current and current.path:
            current.path = normalize_path(current.path)
            worktrees.append(current)
        current = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush()
            continue

        if line.startswith("worktree "):
            flush()
            current = Worktree(path=line.removeprefix("worktree ").strip())
            continue

        if current is None:
            continue

        if line.startswith("HEAD "):
            current.head = line.removeprefix("HEAD ").strip()
        elif line.startswith("branch "):
            current.branch_ref = line.removeprefix("branch ").strip()
            current.branch = current.branch_ref.removeprefix("refs/heads/")
        elif line.startswith("detached"):
            current.detached = True
        elif line.startswith("locked"):
            current.locked = True
        elif line.startswith("prunable"):
            current.prunable = True

    flush()

    if not worktrees:
        raise RuntimeError("no worktrees discovered")

    return worktrees


def git_status_for_worktree(worktree_path: str) -> GitStatus:
    output = run_git(worktree_path, ("status", "--porcelain=v2", "--branch"))
    status = GitStatus()

    for line in output.replace("\r\n", "\n").split("\n"):
        if not line:
            continue

        if line.startswith("# "):
            if line.startswith("# branch.head "):
                status.branch = line.removeprefix("# branch.head ").strip()
            elif line.startswith("# branch.upstream "):
                status.upstream = line.removeprefix("# branch.upstream ").strip()
            elif line.startswith("# branch.ab "):
                for token in line.removeprefix("# branch.ab ").split():
                    if token.startswith("+"):
                        try:
                            status.ahead = int(token[1:])
                        except ValueError:
                            status.ahead = 0
                    elif token.startswith("-"):
                        try:
                            status.behind = int(token[1:])
                        except ValueError:
                            status.behind = 0
            continue

        status.dirty = True
        if line.startswith("? ") or line.startswith("??"):
            status.untracked += 1
            status.unstaged += 1
        else:
            if line.startswith(("1 ", "2 ", "u ")):
                fields = line.split()
                xy = fields[1] if len(fields) > 1 else ""
                if len(xy) >= 1 and xy[0] != ".":
                    status.staged += 1
                if len(xy) >= 2 and xy[1] != ".":
                    status.unstaged += 1
            status.changed += 1

    return status


def opencode_data_root() -> Path:
    xdg_data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return xdg_data_home / "opencode"


def opencode_db_path() -> Path:
    return opencode_data_root() / "opencode.db"


def extract_opencode_session_id(cmdline: str) -> str:
    if not cmdline:
        return ""
    matched = OPENCODE_SESSION_ARG_RE.search(f" {cmdline} ")
    return matched.group(1) if matched else ""


def cmdline_args(cmdline: str) -> List[str]:
    if not cmdline:
        return []
    return cmdline.split()


def is_interactive_opencode_process(process: ProcessInfo) -> bool:
    if process.comm.strip().lower() != "opencode":
        return False

    if process.tty_nr == 0 and not process.fd0.startswith("/dev/pts/"):
        return False

    args = cmdline_args(process.cmdline)
    if len(args) >= 2 and args[1].lower() in OPENCODE_NONINTERACTIVE_SUBCOMMANDS:
        return False

    return True


def process_priority(process: ProcessInfo) -> Tuple[int, int, int]:
    return (
        1 if process.tty_nr != 0 else 0,
        1 if process.state != "Z" else 0,
        process.pid,
    )


def load_opencode_session_metadata(
    session_ids: List[str],
) -> Dict[str, Dict[str, str]]:
    if not session_ids:
        return {}

    db_path = opencode_db_path()
    if not db_path.exists():
        return {}

    metadata: Dict[str, Dict[str, str]] = {}
    placeholders = ",".join("?" for _ in session_ids)
    query = f"SELECT id, directory, title FROM session WHERE id IN ({placeholders})"

    try:
        connection = sqlite3.connect(str(db_path))
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(query, session_ids).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return {}

    for row in rows:
        session_id = str(row["id"])
        directory = normalize_path(str(row["directory"])) if row["directory"] else ""
        title = str(row["title"] or "")
        metadata[session_id] = {
            "directory": directory,
            "title": title,
        }

    return metadata


def load_recent_session_candidates(limit: int = 600) -> List[Dict[str, str]]:
    db_path = opencode_db_path()
    if not db_path.exists():
        return []

    query = (
        "SELECT id, directory, title, time_updated "
        "FROM session "
        "WHERE time_archived IS NULL "
        "ORDER BY time_updated DESC "
        "LIMIT ?"
    )

    try:
        connection = sqlite3.connect(str(db_path))
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(query, (limit,)).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return []

    candidates: List[Dict[str, str]] = []
    for row in rows:
        directory = normalize_path(str(row["directory"])) if row["directory"] else ""
        if not directory:
            continue
        candidates.append(
            {
                "id": str(row["id"]),
                "directory": directory,
                "title": str(row["title"] or ""),
                "time_updated": str(row["time_updated"] or "0"),
            }
        )

    return candidates


def infer_session_metadata_from_process(
    process: ProcessInfo,
    worktree_paths: List[str],
    candidates: List[Dict[str, str]],
) -> Dict[str, str]:
    if not candidates:
        return {}

    process_cwd = normalize_path(process.cwd) if process.cwd else ""
    process_worktree = (
        match_worktree_path(process_cwd, worktree_paths) if process_cwd else ""
    )

    best_candidate: Dict[str, str] = {}
    best_score: Tuple[int, int] = (-1, -1)

    for candidate in candidates:
        directory = candidate.get("directory", "")
        if not directory:
            continue

        candidate_worktree = match_worktree_path(directory, worktree_paths)
        if process_worktree and candidate_worktree != process_worktree:
            continue
        if not process_worktree and not candidate_worktree:
            continue

        score = 0
        if process_cwd and directory == process_cwd:
            score = 4
        elif process_cwd and is_path_within(process_cwd, directory):
            score = 3
        elif process_cwd and is_path_within(directory, process_cwd):
            score = 2
        elif candidate_worktree:
            score = 1

        try:
            updated = int(candidate.get("time_updated", "0"))
        except ValueError:
            updated = 0
        current_score = (score, updated)

        if current_score > best_score:
            best_score = current_score
            best_candidate = candidate

    return best_candidate


def load_opencode_running_tools(session_ids: List[str]) -> Dict[str, Tuple[str, str]]:
    if not session_ids:
        return {}

    db_path = opencode_db_path()
    if not db_path.exists():
        return {}

    placeholders = ",".join("?" for _ in session_ids)
    query = (
        "SELECT session_id, "
        "COALESCE(json_extract(data, '$.tool'), ''), "
        "COALESCE(json_extract(data, '$.state.status'), ''), "
        "time_updated "
        "FROM part "
        "WHERE json_extract(data, '$.type') = 'tool' "
        f"AND session_id IN ({placeholders}) "
        "AND json_extract(data, '$.state.status') IN ('running', 'pending') "
        "ORDER BY time_updated DESC"
    )

    try:
        connection = sqlite3.connect(str(db_path))
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(query, session_ids).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return {}

    result: Dict[str, Tuple[str, str]] = {}
    for row in rows:
        session_id = str(row["session_id"])
        if session_id in result:
            continue
        status = str(row[2] or "")
        tool = str(row[1] or "")
        result[session_id] = (status, tool)

    return result


def load_pending_permission_sessions(session_ids: List[str]) -> Dict[str, bool]:
    if not session_ids:
        return {}

    db_path = opencode_db_path()
    if not db_path.exists():
        return {}

    placeholders = ",".join("?" for _ in session_ids)
    query = (
        "SELECT "
        "COALESCE(json_extract(data, '$.sessionID'), json_extract(data, '$.sessionId'), '') "
        "FROM permission "
        f"WHERE COALESCE(json_extract(data, '$.sessionID'), json_extract(data, '$.sessionId'), '') IN ({placeholders})"
    )

    try:
        connection = sqlite3.connect(str(db_path))
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(query, session_ids).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return {}

    pending: Dict[str, bool] = {}
    for row in rows:
        session_id = str(row[0] or "")
        if session_id:
            pending[session_id] = True

    return pending


def load_opencode_step_activity(session_ids: List[str]) -> Dict[str, bool]:
    if not session_ids:
        return {}

    db_path = opencode_db_path()
    if not db_path.exists():
        return {}

    placeholders = ",".join("?" for _ in session_ids)
    query = (
        "SELECT session_id, "
        "MAX(CASE "
        "WHEN json_extract(data, '$.type') = 'step-start' THEN time_updated "
        "ELSE 0 END), "
        "MAX(CASE "
        "WHEN json_extract(data, '$.type') = 'step-finish' THEN time_updated "
        "ELSE 0 END) "
        "FROM part "
        f"WHERE session_id IN ({placeholders}) "
        "AND json_extract(data, '$.type') IN ('step-start', 'step-finish') "
        "GROUP BY session_id"
    )

    try:
        connection = sqlite3.connect(str(db_path))
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(query, session_ids).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return {}

    activity: Dict[str, bool] = {}
    for row in rows:
        session_id = str(row["session_id"])
        try:
            last_step_start = int(row[1] or 0)
        except (TypeError, ValueError):
            last_step_start = 0
        try:
            last_step_finish = int(row[2] or 0)
        except (TypeError, ValueError):
            last_step_finish = 0
        activity[session_id] = last_step_start > last_step_finish

    return activity


def opencode_session_state(
    activity: Optional[Tuple[str, str]],
    step_active: bool = False,
    permission_pending: bool = False,
) -> str:
    if permission_pending:
        return "permission-ask"

    if not activity:
        if step_active:
            return "running"
        return "idle"

    status, tool = activity
    if status not in {"running", "pending"}:
        if step_active:
            return "running"
        return "idle"

    tool_lower = tool.lower().strip()
    if tool_lower == "question" or "permission" in tool_lower:
        return "permission-ask"

    return "running"


def read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def read_cmdline(path: str) -> str:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return ""

    data = data.rstrip(b"\x00")
    if not data:
        return ""

    parts = [p.decode("utf-8", errors="replace") for p in data.split(b"\x00") if p]
    return " ".join(parts).strip()


def parse_proc_stat(
    raw: str,
) -> Optional[Tuple[int, str, str, int, int, int, int, int]]:
    raw = raw.strip()
    open_idx = raw.find("(")
    close_idx = raw.rfind(")")
    if open_idx < 0 or close_idx < 0 or close_idx <= open_idx:
        return None

    try:
        pid = int(raw[:open_idx].strip())
    except ValueError:
        return None

    comm = raw[open_idx + 1 : close_idx]
    fields = raw[close_idx + 1 :].strip().split()
    if len(fields) < 6:
        return None

    state = fields[0]
    try:
        ppid = int(fields[1])
        pgrp = int(fields[2])
        sid = int(fields[3])
        tty_nr = int(fields[4])
        tpgid = int(fields[5])
    except ValueError:
        return None

    return pid, comm, state, ppid, pgrp, sid, tty_nr, tpgid


def scan_user_processes() -> List[ProcessInfo]:
    uid = os.getuid()
    processes: List[ProcessInfo] = []

    try:
        entries = list(os.scandir("/proc"))
    except OSError as exc:
        raise RuntimeError(f"failed to scan /proc: {exc}") from exc

    for entry in entries:
        if not entry.is_dir(follow_symlinks=False) or not entry.name.isdigit():
            continue

        pid = int(entry.name)

        try:
            stat_result = entry.stat(follow_symlinks=False)
        except OSError:
            continue

        if stat_result.st_uid != uid:
            continue

        proc_root = f"/proc/{pid}"
        stat_raw = read_text(f"{proc_root}/stat")
        if not stat_raw:
            continue

        parsed = parse_proc_stat(stat_raw)
        if not parsed:
            continue

        parsed_pid, comm, state, ppid, pgrp, sid, tty_nr, tpgid = parsed
        if parsed_pid != pid:
            continue

        cmdline = read_cmdline(f"{proc_root}/cmdline")

        try:
            cwd = os.readlink(f"{proc_root}/cwd")
        except OSError:
            cwd = ""

        try:
            fd0 = os.readlink(f"{proc_root}/fd/0")
        except OSError:
            fd0 = ""

        processes.append(
            ProcessInfo(
                pid=pid,
                ppid=ppid,
                pgrp=pgrp,
                sid=sid,
                tty_nr=tty_nr,
                tpgid=tpgid,
                state=state,
                comm=comm,
                cmdline=cmdline,
                cwd=cwd,
                fd0=fd0,
            )
        )

    return processes


def is_shell_process(comm: str) -> bool:
    return os.path.basename(comm.strip().lower()) in SHELL_NAMES


def contains_any(haystack: str, needles: Iterable[str]) -> bool:
    return any(needle in haystack for needle in needles)


def looks_like_permission_ask(process: ProcessInfo) -> bool:
    lower = f"{process.comm} {process.cmdline}".lower()
    if not contains_any(lower, AGENT_HINTS):
        return False
    if contains_any(lower, PERMISSION_HINTS):
        return True
    return process.state in {"S", "I"}


def pick_foreground_process(
    group: List[ProcessInfo], fg_pgid: int
) -> Optional[ProcessInfo]:
    if fg_pgid <= 0:
        return None

    foreground = [p for p in group if p.pgrp == fg_pgid and p.state != "Z"]
    if not foreground:
        return None

    leader = next((p for p in foreground if p.pid == fg_pgid), None)
    if leader is not None:
        return leader

    return min(foreground, key=lambda p: p.pid)


def pick_shell_process(group: List[ProcessInfo]) -> Optional[ProcessInfo]:
    shells = [p for p in group if is_shell_process(p.comm)]
    if not shells:
        return None

    shells.sort(key=lambda p: (0 if p.pid == p.sid else 1, p.pid))
    return shells[0]


def is_path_within(path: str, root: str) -> bool:
    try:
        path_real = normalize_path(path)
        root_real = normalize_path(root)
        common = os.path.commonpath([path_real, root_real])
    except (OSError, ValueError):
        return False
    return common == root_real


def match_worktree_path(cwd: str, worktree_paths: List[str]) -> str:
    if not cwd:
        return ""

    best = ""
    for root in worktree_paths:
        if is_path_within(cwd, root) and len(root) > len(best):
            best = root
    return best


def tty_label(
    shell: Optional[ProcessInfo], foreground: Optional[ProcessInfo], tty_nr: int
) -> str:
    for process in (shell, foreground):
        if process and process.fd0:
            if process.fd0.startswith("/dev/"):
                return process.fd0.removeprefix("/dev/")
            return process.fd0
    return f"tty:{tty_nr}"


def foreground_command_summary(process: ProcessInfo) -> str:
    if process.cmdline:
        parts = process.cmdline.split()
        if parts:
            head = os.path.basename(parts[0])
            if len(parts) == 1:
                return head
            return f"{head} {' '.join(parts[1:])}"
    return process.comm or "-"


def build_session(
    group: List[ProcessInfo], worktree_paths: List[str]
) -> Tuple[Optional[Session], str]:
    if not group:
        return None, ""

    fg_pgid = next((p.tpgid for p in group if p.tpgid > 0), 0)
    foreground = pick_foreground_process(group, fg_pgid)
    shell = pick_shell_process(group)

    if foreground is None and shell is None:
        return None, ""

    best_path = ""
    candidates: List[str] = []
    if shell and shell.cwd:
        candidates.append(shell.cwd)
    if foreground and foreground.cwd:
        candidates.append(foreground.cwd)
    candidates.extend(p.cwd for p in group if p.cwd)

    for cwd in candidates:
        matched = match_worktree_path(cwd, worktree_paths)
        if matched and len(matched) > len(best_path):
            best_path = matched

    if not best_path:
        return None, ""

    if foreground is None:
        if shell is None:
            return None, ""
        foreground = shell
        session_state = "idle"
    elif is_shell_process(foreground.comm):
        session_state = "idle"
    elif looks_like_permission_ask(foreground):
        session_state = "permission-ask"
    else:
        session_state = "running"

    session = Session(
        tty=tty_label(shell, foreground, group[0].tty_nr),
        shell_pid=shell.pid if shell else 0,
        foreground=foreground,
        state=session_state,
    )
    return session, best_path


def sessions_by_worktree(worktree_paths: List[str]) -> Dict[str, List[Session]]:
    result: Dict[str, List[Session]] = {path: [] for path in worktree_paths}
    if not worktree_paths:
        return result

    processes = scan_user_processes()

    interactive_processes = [p for p in processes if is_interactive_opencode_process(p)]
    if not interactive_processes:
        return result

    sessions_by_id: Dict[str, ProcessInfo] = {}
    unknown_processes: List[ProcessInfo] = []

    for process in interactive_processes:
        session_id = extract_opencode_session_id(process.cmdline)
        if not session_id:
            unknown_processes.append(process)
            continue

        existing = sessions_by_id.get(session_id)
        if existing is None or process_priority(process) > process_priority(existing):
            sessions_by_id[session_id] = process

    inferred_by_id: Dict[str, Tuple[ProcessInfo, Dict[str, str]]] = {}
    unresolved_processes: List[ProcessInfo] = []
    if unknown_processes:
        candidates = load_recent_session_candidates()
        for process in unknown_processes:
            inferred = infer_session_metadata_from_process(
                process, worktree_paths, candidates
            )
            session_id = inferred.get("id", "")
            if not session_id:
                unresolved_processes.append(process)
                continue

            current = inferred_by_id.get(session_id)
            if current is None or process_priority(process) > process_priority(
                current[0]
            ):
                inferred_by_id[session_id] = (process, inferred)

    for session_id, (process, _) in inferred_by_id.items():
        existing = sessions_by_id.get(session_id)
        if existing is None or process_priority(process) > process_priority(existing):
            sessions_by_id[session_id] = process

    session_ids = list(sessions_by_id.keys())
    metadata = load_opencode_session_metadata(session_ids)
    for session_id, (_, inferred) in inferred_by_id.items():
        if session_id in metadata:
            continue
        metadata[session_id] = {
            "directory": inferred.get("directory", ""),
            "title": inferred.get("title", ""),
        }

    activity_map = load_opencode_running_tools(session_ids)
    permission_map = load_pending_permission_sessions(session_ids)
    step_activity_map = load_opencode_step_activity(session_ids)

    for session_id, process in sessions_by_id.items():
        session_meta = metadata.get(session_id, {})
        session_directory = session_meta.get("directory", "")
        session_title = session_meta.get("title", "")

        match_source = session_directory or process.cwd
        worktree_path = match_worktree_path(match_source, worktree_paths)
        if not worktree_path and process.cwd:
            worktree_path = match_worktree_path(process.cwd, worktree_paths)
        if not worktree_path:
            continue

        state = opencode_session_state(
            activity_map.get(session_id),
            step_active=step_activity_map.get(session_id, False),
            permission_pending=permission_map.get(session_id, False),
        )
        tty = tty_label(None, process, process.tty_nr)
        session = Session(
            tty=tty,
            shell_pid=process.ppid,
            foreground=process,
            state=state,
            session_id=session_id,
            session_title=session_title,
            session_directory=session_directory,
        )
        result.setdefault(worktree_path, []).append(session)

    for process in unresolved_processes:
        worktree_path = match_worktree_path(process.cwd, worktree_paths)
        if not worktree_path:
            continue

        state = "running" if process.state in {"R", "D"} else "idle"
        session = Session(
            tty=tty_label(None, process, process.tty_nr),
            shell_pid=process.ppid,
            foreground=process,
            state=state,
            session_id="",
            session_title="",
            session_directory=normalize_path(process.cwd) if process.cwd else "",
        )
        result.setdefault(worktree_path, []).append(session)

    for sessions in result.values():
        sessions.sort(key=lambda s: (s.tty, s.session_id))

    return result


def display_branch(worktree: Worktree) -> str:
    branch = worktree.status.branch.strip()
    if branch:
        if branch in {"(detached)", "detached"}:
            return "detached"
        return branch
    if worktree.branch:
        return worktree.branch
    if worktree.detached:
        return "detached"
    return "-"


def format_git_column(worktree: Worktree) -> str:
    if worktree.git_error:
        return "status:error"
    state = "dirty" if worktree.status.dirty else "clean"
    return f"{state} +{worktree.status.staged}/-{worktree.status.unstaged}"


def session_summary(sessions: List[Session]) -> str:
    if not sessions:
        return "none"
    idle = sum(1 for s in sessions if s.state == "idle")
    ask = sum(1 for s in sessions if s.state == "permission-ask")
    running = sum(1 for s in sessions if s.state == "running")
    return f"i:{idle} r:{running} p:{ask}"


def compact_session_title(title: str) -> str:
    if not title:
        return ""

    normalized = " ".join(title.split())
    normalized = re.sub(r"\s*\(@[^)]*\)\s*$", "", normalized)
    return normalized.strip()


def session_display_name(session: Session) -> str:
    title = compact_session_title(session.session_title)
    if title:
        return title
    if session.session_id.startswith("ses_"):
        return f"sid:{session.session_id[4:12]}"
    if session.session_id:
        return f"sid:{session.session_id[:8]}"
    return "opencode"


def session_state_symbol(
    state: str,
    running_frame: int = 0,
    permission_ask_frame: int = 0,
) -> str:
    if state == "idle":
        return "·"
    if state == "running":
        if RUNNING_ANIMATION_FRAMES:
            return RUNNING_ANIMATION_FRAMES[
                running_frame % len(RUNNING_ANIMATION_FRAMES)
            ]
        return "⚙"
    if state == "permission-ask":
        if PERMISSION_ASK_ANIMATION_FRAMES:
            return PERMISSION_ASK_ANIMATION_FRAMES[
                permission_ask_frame % len(PERMISSION_ASK_ANIMATION_FRAMES)
            ]
        return "?"
    return "*"


def session_state_display(
    state: str,
    running_frame: int = 0,
    permission_ask_frame: int = 0,
) -> str:
    return session_state_symbol(state, running_frame, permission_ask_frame)


def build_snapshot(base_path: str, alias_store: AliasStore) -> List[Worktree]:
    worktrees = discover_worktrees(base_path)
    current_worktree_path = match_worktree_path(base_path, [w.path for w in worktrees])

    for worktree in worktrees:
        try:
            worktree.status = git_status_for_worktree(worktree.path)
        except Exception as exc:  # pylint: disable=broad-except
            worktree.git_error = str(exc)

    sessions_map = sessions_by_worktree([w.path for w in worktrees])

    for worktree in worktrees:
        worktree.alias = alias_store.name_for(worktree.path)
        worktree.is_current = worktree.path == current_worktree_path
        worktree.sessions = sessions_map.get(worktree.path, [])

    return worktrees


def char_display_width(char: str) -> int:
    if not char:
        return 0
    if unicodedata.combining(char):
        return 0
    if unicodedata.category(char).startswith("C"):
        return 0
    if unicodedata.east_asian_width(char) in {"W", "F"}:
        return 2
    return 1


def display_width(text: str) -> int:
    return sum(char_display_width(char) for char in text)


def clip_to_display_width(text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""
    consumed = 0
    chars: List[str] = []
    for char in text:
        width = char_display_width(char)
        if consumed + width > max_width:
            break
        chars.append(char)
        consumed += width
    return "".join(chars)


def trim_left_display_width(text: str, trim_width: int) -> str:
    if trim_width <= 0:
        return text

    consumed = 0
    index = 0
    while index < len(text) and consumed < trim_width:
        width = char_display_width(text[index])
        consumed += width if width > 0 else 1
        index += 1
    return text[index:]


def fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if display_width(text) <= width:
        return text
    if width <= 3:
        return clip_to_display_width(text, width)
    return clip_to_display_width(text, width - 3) + "..."


def safe_addstr(
    stdscr: curses.window, y: int, x: int, text: str, attr: int = 0
) -> None:
    height, width = stdscr.getmaxyx()
    if y < 0 or y >= height or x >= width:
        return
    if x < 0:
        text = trim_left_display_width(text, -x)
        x = 0
    clipped = fit(text, width - x - 1)
    if not clipped:
        return
    try:
        stdscr.addstr(y, x, clipped, attr)
    except curses.error:
        return


def wrapped_lines(text: str, width: int) -> List[str]:
    if width <= 1:
        return [text]
    if text == "":
        return [""]
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )


def draw_wrapped_key_value(
    stdscr: curses.window,
    y: int,
    max_y: int,
    start_x: int,
    panel_width: int,
    key: str,
    value: str,
    attr: int = 0,
) -> int:
    if y >= max_y:
        return y

    prefix = f"{key}: "
    content_width = max(1, panel_width - len(prefix))
    lines = wrapped_lines(value, content_width)

    for index, content in enumerate(lines):
        if y >= max_y:
            break
        left = prefix if index == 0 else " " * len(prefix)
        safe_addstr(stdscr, y, start_x, left + content, attr)
        y += 1

    return y


def sidebar_preview_sessions(sessions: List[Session]) -> Tuple[List[Session], int]:
    if len(sessions) <= SIDEBAR_SESSION_PREVIEW:
        return sessions, 0
    return sessions[:SIDEBAR_SESSION_PREVIEW], len(sessions) - SIDEBAR_SESSION_PREVIEW


def sidebar_block_height(worktree: Worktree) -> int:
    preview, hidden = sidebar_preview_sessions(worktree.sessions)
    session_lines = len(preview) if preview else 1
    if hidden > 0:
        session_lines += 1
    return 1 + session_lines + 1


def sidebar_visible_indices(
    worktrees: List[Worktree], start_idx: int, max_lines: int
) -> List[int]:
    if not worktrees or max_lines <= 0:
        return []

    start = max(0, min(start_idx, len(worktrees) - 1))
    visible: List[int] = []
    used = 0
    idx = start

    while idx < len(worktrees):
        block_height = sidebar_block_height(worktrees[idx])
        if visible and used + block_height > max_lines:
            break
        if not visible and block_height > max_lines:
            visible.append(idx)
            break

        visible.append(idx)
        used += block_height
        idx += 1

    return visible


def ensure_sidebar_selection_visible(state: AppState, max_lines: int) -> None:
    if not state.worktrees:
        state.table_offset = 0
        return

    state.selected = max(-1, min(state.selected, len(state.worktrees) - 1))
    state.table_offset = max(0, min(state.table_offset, len(state.worktrees) - 1))

    if state.selected == -1:
        return

    if state.selected < state.table_offset:
        state.table_offset = state.selected

    while True:
        visible = sidebar_visible_indices(
            state.worktrees, state.table_offset, max_lines
        )
        if state.selected in visible:
            return
        if state.selected < state.table_offset:
            state.table_offset = state.selected
            continue
        state.table_offset += 1
        if state.table_offset >= len(state.worktrees):
            state.table_offset = len(state.worktrees) - 1
            return


def draw_worktree_header_line(
    stdscr: curses.window,
    y: int,
    x: int,
    width: int,
    worktree: Worktree,
    selected: bool,
    colors: Dict[str, int],
) -> None:
    marker = ">" if selected else " "
    current_marker = "v" if worktree.is_current else " "
    name = fit(worktree.alias, max(8, width // 3))
    branch = fit(display_branch(worktree), max(6, width // 4))
    select_attr = curses.A_REVERSE if selected else 0
    cursor = x

    def put(segment: str, attr: int = 0) -> None:
        nonlocal cursor
        safe_addstr(stdscr, y, cursor, segment, select_attr | attr)
        cursor += display_width(segment)

    put(f"{marker}{current_marker} {name} ", curses.A_BOLD)
    put(branch, colors.get("branch", curses.A_DIM))

    if worktree.git_error:
        put(" (")
        put("status:error", colors.get("error", 0))
        put(")")
        return

    git_state = "dirty" if worktree.status.dirty else "clean"
    put(" (")
    put(git_state)
    put(" ")
    put(f"+{worktree.status.staged}", colors.get("plus", 0))
    put("/")
    put(f"-{worktree.status.unstaged}", colors.get("minus", 0))
    put(")")


def draw_sidebar(
    stdscr: curses.window,
    state: AppState,
    colors: Dict[str, int],
    running_frame: int,
    permission_ask_frame: int,
    start_y: int,
    start_x: int,
    width: int,
    max_lines: int,
) -> None:
    if width <= 6 or max_lines <= 0:
        return

    if not state.worktrees:
        safe_addstr(stdscr, start_y, start_x, "(no worktrees found)")
        return

    ensure_sidebar_selection_visible(state, max_lines)
    visible = sidebar_visible_indices(state.worktrees, state.table_offset, max_lines)

    y = start_y
    max_y = start_y + max_lines
    for idx in visible:
        if y >= max_y:
            break

        worktree = state.worktrees[idx]
        selected = idx == state.selected
        select_attr = curses.A_REVERSE if selected else 0

        draw_worktree_header_line(stdscr, y, start_x, width, worktree, selected, colors)
        y += 1
        if y >= max_y:
            break

        preview, hidden = sidebar_preview_sessions(worktree.sessions)
        if not preview:
            safe_addstr(
                stdscr, y, start_x, "   (no sessions)", select_attr | curses.A_DIM
            )
            y += 1
        else:
            line_width = max(8, width - 4)
            for session in preview:
                if y >= max_y:
                    break
                state_label = fit(
                    session_state_display(
                        session.state,
                        running_frame,
                        permission_ask_frame,
                    ),
                    max(6, line_width // 3),
                )
                name_label = session_display_name(session)
                prefix = f"   {state_label} "
                prefix_width = display_width(prefix)
                safe_addstr(
                    stdscr,
                    y,
                    start_x,
                    prefix,
                    select_attr | colors.get(session.state, 0),
                )
                safe_addstr(
                    stdscr,
                    y,
                    start_x + prefix_width,
                    fit(name_label, max(6, line_width - prefix_width + 3)),
                    select_attr,
                )
                y += 1

        if hidden > 0 and y < max_y:
            safe_addstr(
                stdscr,
                y,
                start_x,
                f"   ... +{hidden} more",
                select_attr | colors.get("branch", curses.A_DIM),
            )
            y += 1

        if y < max_y:
            y += 1


def draw_detail_panel(
    stdscr: curses.window,
    worktree: Worktree,
    colors: Dict[str, int],
    running_frame: int,
    permission_ask_frame: int,
    start_y: int,
    start_x: int,
    width: int,
    max_lines: int,
) -> None:
    if width <= 8 or max_lines <= 0:
        return

    y = start_y
    max_y = start_y + max_lines

    safe_addstr(stdscr, y, start_x, "details", colors.get("header", curses.A_BOLD))
    y += 1
    if y >= max_y:
        return

    safe_addstr(stdscr, y, start_x, f"name: {worktree.alias}")
    y += 1
    if y >= max_y:
        return

    y = draw_wrapped_key_value(
        stdscr,
        y,
        max_y,
        start_x,
        width,
        "path",
        worktree.path,
    )
    if y >= max_y:
        return

    safe_addstr(stdscr, y, start_x, f"branch: {display_branch(worktree)}")
    y += 1
    if y >= max_y:
        return

    if worktree.git_error:
        y = draw_wrapped_key_value(
            stdscr,
            y,
            max_y,
            start_x,
            width,
            "git",
            worktree.git_error,
            colors.get("error", 0),
        )
    else:
        git_state = "dirty" if worktree.status.dirty else "clean"
        safe_addstr(stdscr, y, start_x, f"git: {git_state} ")
        safe_addstr(
            stdscr,
            y,
            start_x + len(f"git: {git_state} "),
            f"+{worktree.status.staged}",
            colors.get("plus", 0),
        )
        safe_addstr(
            stdscr,
            y,
            start_x + len(f"git: {git_state} +{worktree.status.staged}"),
            "/",
        )
        safe_addstr(
            stdscr,
            y,
            start_x + len(f"git: {git_state} +{worktree.status.staged}/"),
            f"-{worktree.status.unstaged}",
            colors.get("minus", 0),
        )
        y += 1
        if y < max_y:
            safe_addstr(
                stdscr,
                y,
                start_x,
                f"changes: {worktree.status.changed}, untracked: {worktree.status.untracked}",
            )
            y += 1

    if y >= max_y:
        return

    flags = []
    if worktree.detached:
        flags.append("detached")
    if worktree.locked:
        flags.append("locked")
    if worktree.prunable:
        flags.append("prunable")
    if flags:
        safe_addstr(stdscr, y, start_x, f"flags: {', '.join(flags)}")
        y += 1
    if y >= max_y:
        return

    y += 1
    if y >= max_y:
        return

    safe_addstr(stdscr, y, start_x, "sessions", colors.get("header", curses.A_BOLD))
    y += 1
    if y >= max_y:
        return

    if not worktree.sessions:
        safe_addstr(stdscr, y, start_x, "(no active sessions)")
        return

    for idx, session in enumerate(worktree.sessions, start=1):
        if y >= max_y:
            break

        prefix = (
            f"[{idx}] "
            f"{session_state_display(session.state, running_frame, permission_ask_frame)} "
        )
        prefix_width = display_width(prefix)
        safe_addstr(stdscr, y, start_x, prefix, colors.get(session.state, 0))
        safe_addstr(
            stdscr,
            y,
            start_x + prefix_width,
            fit(session_display_name(session), max(8, width - prefix_width - 1)),
        )
        y += 1

        if y < max_y:
            safe_addstr(
                stdscr,
                y,
                start_x,
                f"    tty: {session.tty}  pid: {session.foreground.pid}  shell_pid: {session.shell_pid}",
            )
            y += 1

        if y < max_y:
            cwd = session.foreground.cwd or "-"
            y = draw_wrapped_key_value(
                stdscr,
                y,
                max_y,
                start_x + 4,
                max(8, width - 4),
                "cwd",
                cwd,
            )

        if y < max_y:
            command = session.foreground.cmdline or session.foreground.comm or "-"
            safe_addstr(
                stdscr,
                y,
                start_x,
                f"    cmd: {fit(command, max(10, width - 10))}",
            )
            y += 1

        if y < max_y:
            y += 1


def selected_worktree(state: AppState) -> Optional[Worktree]:
    if not state.worktrees:
        return None
    if state.selected < 0 or state.selected >= len(state.worktrees):
        return None
    return state.worktrees[state.selected]


def refresh_state(state: AppState) -> None:
    previous_selected = state.selected
    current = selected_worktree(state)
    previous_path = current.path if current else ""

    state.refreshing = True
    try:
        worktrees = build_snapshot(state.base_path, state.alias_store)
        state.worktrees = worktrees
        state.error = ""

        if not worktrees:
            state.selected = -1
        elif previous_selected == -1:
            state.selected = -1
        elif previous_path:
            for idx, wt in enumerate(worktrees):
                if wt.path == previous_path:
                    state.selected = idx
                    break
            else:
                state.selected = min(max(0, state.selected), len(worktrees) - 1)
        else:
            state.selected = min(max(0, state.selected), len(worktrees) - 1)

        if worktrees and state.selected < -1:
            state.selected = 0
        if not worktrees:
            state.selected = -1
        if state.selected >= len(worktrees):
            state.selected = len(worktrees) - 1
        if state.selected < -1:
            state.selected = 0
    except Exception as exc:  # pylint: disable=broad-except
        state.error = str(exc)
        state.worktrees = []
        state.selected = -1
        state.table_offset = 0
    finally:
        state.refreshing = False
        state.last_refresh_at = time.monotonic()


def init_colors() -> Dict[str, int]:
    if not curses.has_colors():
        return {"branch": curses.A_DIM}

    curses.start_color()
    curses.use_default_colors()

    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_CYAN, -1)
    curses.init_pair(4, curses.COLOR_RED, -1)
    curses.init_pair(5, curses.COLOR_WHITE, -1)

    return {
        "idle": curses.color_pair(1),
        "running": curses.color_pair(2),
        "permission-ask": curses.color_pair(3),
        "error": curses.color_pair(4),
        "header": curses.color_pair(5) | curses.A_BOLD,
        "plus": curses.color_pair(1) | curses.A_BOLD,
        "minus": curses.color_pair(4) | curses.A_BOLD,
        "branch": curses.A_DIM,
    }


def draw(stdscr: curses.window, state: AppState, colors: Dict[str, int]) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    now = time.monotonic()
    running_frame = int(now / RUNNING_ANIMATION_FRAME_SECONDS)
    permission_ask_frame = int(now / PERMISSION_ASK_ANIMATION_FRAME_SECONDS)

    line = 0
    safe_addstr(stdscr, line, 0, "opwt", curses.A_BOLD)
    line += 1

    status_line = f"base: {state.base_path}"
    if state.refreshing:
        status_line += " | refreshing"
    if state.error:
        status_line += f" | error: {state.error}"
    status_attr = colors.get("error", 0) if state.error else 0
    safe_addstr(stdscr, line, 0, status_line, status_attr)
    line += 2

    reserved_bottom = 5 if state.rename_mode else 3
    content_top = line
    content_bottom = max(content_top + 3, height - reserved_bottom)
    content_height = max(3, content_bottom - content_top)

    split_detail = state.detail_mode and width >= DETAIL_SPLIT_MIN_WIDTH

    if split_detail:
        sidebar_width = max(36, min(68, width // 2))
        sidebar_width = min(sidebar_width, width - 26)
        draw_sidebar(
            stdscr,
            state,
            colors,
            running_frame,
            permission_ask_frame,
            content_top,
            0,
            sidebar_width,
            content_height,
        )

        divider_x = sidebar_width + 1
        for y in range(content_top, content_bottom):
            safe_addstr(stdscr, y, divider_x, "|", colors.get("branch", curses.A_DIM))

        wt = selected_worktree(state)
        if wt is None:
            safe_addstr(
                stdscr,
                content_top,
                divider_x + 2,
                "details",
                colors.get("header", curses.A_BOLD),
            )
            safe_addstr(
                stdscr, content_top + 1, divider_x + 2, "(no worktree selected)"
            )
        else:
            draw_detail_panel(
                stdscr,
                wt,
                colors,
                running_frame,
                permission_ask_frame,
                content_top,
                divider_x + 2,
                max(8, width - (divider_x + 3)),
                content_height,
            )
    elif state.detail_mode:
        wt = selected_worktree(state)
        if wt is None:
            safe_addstr(
                stdscr,
                content_top,
                0,
                "details",
                colors.get("header", curses.A_BOLD),
            )
            safe_addstr(stdscr, content_top + 1, 0, "(no worktree selected)")
        else:
            draw_detail_panel(
                stdscr,
                wt,
                colors,
                running_frame,
                permission_ask_frame,
                content_top,
                0,
                max(8, width - 1),
                content_height,
            )
    else:
        draw_sidebar(
            stdscr,
            state,
            colors,
            running_frame,
            permission_ask_frame,
            content_top,
            0,
            max(8, width - 1),
            content_height,
        )

    if state.rename_mode:
        prompt_y = height - 4
        safe_addstr(
            stdscr,
            prompt_y,
            0,
            "rename alias (Enter save / Esc cancel):",
            colors.get("header", curses.A_BOLD),
        )
        safe_addstr(stdscr, prompt_y + 1, 0, "> " + state.rename_input)
        safe_addstr(stdscr, prompt_y + 2, 0, "default: directory name")
        cursor_x = min(width - 2, 2 + display_width(state.rename_input))
        try:
            curses.curs_set(1)
            stdscr.move(prompt_y + 1, cursor_x)
        except curses.error:
            pass
    else:
        try:
            curses.curs_set(0)
        except curses.error:
            pass

    if state.detail_mode:
        keys_hint = "keys: up/down or j/k, Enter close details, r rename alias, g refresh, q quit"
    else:
        keys_hint = (
            "keys: up/down or j/k, Enter details, r rename alias, g refresh, q quit"
        )
    safe_addstr(stdscr, height - 2, 0, keys_hint)
    safe_addstr(stdscr, height - 1, 0, "note: permission-ask detection is heuristic")

    stdscr.refresh()


def handle_rename_key(state: AppState, key: int) -> bool:
    if key in (3,):
        return True

    if key == 27:  # ESC
        state.rename_mode = False
        state.rename_target_path = ""
        state.rename_input = ""
        return False

    if key in (10, 13):
        try:
            state.alias_store.set(state.rename_target_path, state.rename_input)
            state.error = ""
        except Exception as exc:  # pylint: disable=broad-except
            state.error = f"failed to save alias: {exc}"
        state.rename_mode = False
        state.rename_target_path = ""
        state.rename_input = ""
        refresh_state(state)
        return False

    if key in (curses.KEY_BACKSPACE, 127, 8):
        state.rename_input = state.rename_input[:-1]
        return False

    if 32 <= key <= 126:
        state.rename_input += chr(key)

    return False


def handle_key(state: AppState, key: int) -> bool:
    if state.rename_mode:
        return handle_rename_key(state, key)

    if key in (ord("q"), 3):
        return True

    if key in (curses.KEY_UP, ord("k")):
        if not state.worktrees:
            return False
        if state.selected == -1:
            state.selected = len(state.worktrees) - 1
        elif state.selected == 0:
            state.selected = -1
        else:
            state.selected -= 1
        return False

    if key in (curses.KEY_DOWN, ord("j")):
        if not state.worktrees:
            return False
        if state.selected == -1:
            state.selected = 0
        elif state.selected == len(state.worktrees) - 1:
            state.selected = -1
        else:
            state.selected += 1
        return False

    if key in (10, 13, curses.KEY_ENTER):
        if selected_worktree(state) is not None:
            state.detail_mode = not state.detail_mode
        return False

    if key in (ord("g"), ord("R")):
        refresh_state(state)
        return False

    if key == ord("r"):
        wt = selected_worktree(state)
        if wt is not None:
            state.rename_mode = True
            state.rename_target_path = wt.path
            state.rename_input = wt.alias
        return False

    return False


def run_tui(stdscr: curses.window, state: AppState) -> None:
    stdscr.nodelay(True)
    stdscr.keypad(True)
    colors = init_colors()

    refresh_state(state)

    while True:
        now = time.monotonic()
        if (
            not state.rename_mode
            and now - state.last_refresh_at >= REFRESH_INTERVAL_SECONDS
        ):
            refresh_state(state)

        draw(stdscr, state, colors)
        key = stdscr.getch()
        if key == -1:
            time.sleep(0.05)
            continue

        should_quit = handle_key(state, key)
        if should_quit:
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="opwt: git worktree + OpenCode session TUI"
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path inside target git repository or worktree",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        base_path = resolve_base_path(args.path)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"failed to resolve path: {exc}")
        return 1

    try:
        alias_store = AliasStore()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"failed to initialize alias store: {exc}")
        return 1

    state = AppState(base_path=base_path, alias_store=alias_store)

    try:
        curses.wrapper(run_tui, state)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"tui failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
