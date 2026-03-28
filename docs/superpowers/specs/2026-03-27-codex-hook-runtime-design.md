# Codex Hook Runtime Design

## Goal

Add a `claude-mem`-style hook runtime for `codex-memory` so Codex users can invoke
`SessionStart`, `UserPromptSubmit`, `PostToolUse`, `Stop`, and `SessionEnd`
against the local memory sidecar, even though Codex does not expose official
hook APIs.

## Constraints

- Codex currently exposes session transcripts and thread metadata on disk, not a
  public lifecycle hook API.
- The implementation must stay local-first and use the existing SQLite + Qdrant
  storage.
- It must not require remote embeddings or a network dependency.
- The CLI should expose the same lifecycle event names as Claude Code so a
  future wrapper or launcher can call them directly.

## Recommended Approach

Implement a local hook shim with three layers:

1. `hook runtime` functions that accept one lifecycle event at a time.
2. `hook` CLI entrypoint that exposes the event names directly.
3. Hook-aware renderers that return either an injection block or an ingestion
   status payload depending on the event type.

`SessionStart` should return a context block suitable for injection into a new
session. `UserPromptSubmit`, `PostToolUse`, `Stop`, and `SessionEnd` should run
incremental ingestion against the latest changed threads and report what was
synced.

## Event Mapping

- `SessionStart`
  - Run incremental sync for recent threads in the target workspace.
  - Return the freshest `context` block so a caller can inject it into the next
    prompt.
- `UserPromptSubmit`
  - Sync latest changed thread(s).
  - Return ingestion status only.
- `PostToolUse`
  - Sync latest changed thread(s).
  - Return ingestion status only.
- `Stop`
  - Sync latest changed thread(s).
  - Return ingestion status only.
- `SessionEnd`
  - Sync latest changed thread(s).
  - Return ingestion status and the latest thread timeline for debugging.

## Files

- `codex_memory/hook_runtime.py`
  - Hook event validation, incremental sync orchestration, hook payload
    rendering.
- `codex_memory/cli.py`
  - New `hook` command and arguments.
- `codex_memory/indexer.py`
  - Reuse `sync_latest_threads`, `recent_threads`, `resolve_thread`,
    `render_context`, and `render_timeline`.
- `tests/test_memory_quality.py`
  - Hook CLI parsing and hook behavior regression coverage.
- `README.md`
  - Document the hook runtime commands and intended usage.

## Non-Goals

- No daemon or background service in this step.
- No OS-specific auto-start integration in this step.
- No claim of using official Codex hooks.

