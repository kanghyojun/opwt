# opwt

Terminal UI that shows:

- Git worktrees from a target path
- Git status per worktree (`clean/dirty`, `ahead/behind`)
- Active OpenCode sessions mapped to each worktree
- Session state (`idle`, `running`, `permission-ask`)

## Run

```bash
python3 opwt.py [path]
```

- If `path` is omitted, current directory is used.
- `path` should be inside a Git repository or one of its worktrees.

## Keys

- `up/down` or `j/k`: move selection
- selection wraps with a `none-selected` state at boundaries
- `Enter`: toggle detail panel for selected worktree
- `r`: rename selected worktree alias (defaults to directory name)
- `g`: manual refresh
- `q`: quit

Default layout is sidebar-friendly:

- Worktree header: `name branch (git-state +ahead/-behind)`
- Session preview rows under each worktree: `state-symbol session-title`

State symbols:

- `·`: idle
- `⚙`: running
- `?`: permission-ask or question tool waiting

## Build Binary

```bash
./build.sh
```

- Output: `dist/opwt`

Aliases are saved to:

```text
~/.config/opwt/aliases.json
```

## Notes

- Session detection is Linux `/proc` + OpenCode SQLite metadata based and tracks only current-user processes.
- Even without `opencode -s <session_id>`, the app infers sessions by matching process `cwd` with recent OpenCode session metadata.
- `permission-ask` is heuristic (keyword + process state), so treat it as a hint.
