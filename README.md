# codex-memory

`codex-memory` is a local memory sidecar for Codex Desktop. It indexes Codex
thread metadata from `~/.codex/state_*.sqlite` and session transcripts from
`~/.codex/sessions/**/*.jsonl`, then stores a hybrid recall index in the local
project data directory at `.data/codex-memory.sqlite` by default.

It is designed to be:

- local-first: all indexed data stays on disk
- hybrid: SQLite metadata + FTS5 + OpenAI-compatible embeddings + local Qdrant
- project-aware: filter results by `cwd`
- sandbox-friendly: default writes stay inside this workspace
- lifecycle-aware: extract task/result/decision/observation style memory, not just raw chat

## What It Stores

For each thread it extracts:

- title and workspace
- user requests
- assistant final answer
- compact result and decision signals
- tool observations from relevant command outputs
- tool usage and shell commands
- file paths mentioned or edited
- a compact thread summary

Those are written into a searchable memory index and grouped back into thread
results at query time.

## Setup

The wrapper prefers the local virtualenv automatically when
[`/.venv`](/Users/alex/Desktop/dev/codex-memory/.venv) exists.

To install the vector backend dependency:

```bash
./.venv/bin/pip install qdrant-client
```

Default embedding provider:

```bash
fastembed
```

Default local model:

```bash
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

This gives you real local embeddings plus local Qdrant without depending on
your API proxy.

If you explicitly want to use an OpenAI-compatible `/embeddings` endpoint
instead, `codex-memory` can read:

1. `OPENAI_API_KEY` and `OPENAI_BASE_URL`
2. `~/.codex/auth.json` and `~/.codex/config.toml`
3. fallback base URL: `https://api.openai.com/v1` only when explicit fallback is enabled

Default remote embedding model:

```bash
text-embedding-3-small
```

You can override it with:

```bash
export CODEX_MEMORY_EMBED_MODEL=text-embedding-3-large
```

To force remote embeddings:

```bash
export CODEX_MEMORY_EMBED_PROVIDER=openai-compatible
```

To force local embeddings:

```bash
export CODEX_MEMORY_EMBED_PROVIDER=fastembed
```

By default, `codex-memory` does not silently fall back from your configured
proxy/base URL to the official OpenAI endpoint. If you explicitly want that
behavior, enable it with:

```bash
export CODEX_MEMORY_EMBED_ALLOW_OPENAI_FALLBACK=1
```

## Commands

Run from this project root:

```bash
./codex-memory index
./codex-memory sync --limit 3
./codex-memory status
./codex-memory search "claude-mem" --cwd /Users/alex/Desktop/dev
./codex-memory context "skills install" --cwd /Users/alex/Desktop/dev
./codex-memory brief latest
./codex-memory timeline latest
./codex-memory hook SessionStart --cwd /Users/alex/Desktop/dev
./codex-memory hook PostToolUse --cwd /Users/alex/Desktop/dev
./codex-memory hook SessionEnd --cwd /Users/alex/Desktop/dev
./codex-memory watch --cwd /Users/alex/Desktop/dev --max-loops 1
./codex-memory autostart status --cwd /Users/alex/Desktop/dev
```

`sync` incrementally ingests only the most recent changed threads, which makes
it a good fit for hook-like or near-realtime refresh loops. `search` returns the most relevant past threads. `context` renders a compact
block that can be pasted into a fresh Codex conversation and now includes
decision/observation signals when available. `brief` shows a single thread
summary with commands, files, and the top extracted signals. `timeline`
replays the structured memory flow for one thread.

## Hook Shim

`hook` exposes Claude Code style lifecycle event names even though Codex does
not currently expose official hook APIs. This gives you a compatible runtime
surface that wrappers or launch scripts can call:

```bash
./codex-memory hook SessionStart --cwd /Users/alex/Desktop/dev
./codex-memory hook UserPromptSubmit --cwd /Users/alex/Desktop/dev
./codex-memory hook PostToolUse --cwd /Users/alex/Desktop/dev
./codex-memory hook Stop --cwd /Users/alex/Desktop/dev
./codex-memory hook SessionEnd --cwd /Users/alex/Desktop/dev
```

- `SessionStart` returns an injection-ready context block.
- `UserPromptSubmit`, `PostToolUse`, and `Stop` run incremental sync.
- `SessionEnd` runs incremental sync and returns the latest structured timeline.

## Watch Mode

`watch` turns the hook shim into an automatic local runtime by polling recent
threads and new rollout events:

```bash
./codex-memory watch --cwd /Users/alex/Desktop/dev
./codex-memory watch --cwd /Users/alex/Desktop/dev --poll-interval 1.0
./codex-memory watch --cwd /Users/alex/Desktop/dev --max-loops 1
./codex-memory watch --cwd /Users/alex/Desktop/dev --emit-dir /tmp/codex-memory-runtime
./codex-memory watch --cwd /Users/alex/Desktop/dev --emit-session-end-on-exit
```

It emits:

- `SessionStart` when a new active thread is detected
- `UserPromptSubmit` when a new `user_message` arrives
- `PostToolUse` when a new `function_call_output` arrives
- `Stop` when a `task_complete` event appears
- `SessionEnd` when the active thread switches

`watch` now streams events as they happen instead of buffering until exit. When
`--emit-dir` is set it also writes:

- `events.jsonl` as an append-only event log
- `latest/<Event>.json` and `latest/<Event>.txt` snapshots per hook event
- `latest.json` and `latest.txt` for the most recent emitted payload

This makes the runtime usable both interactively and as a background watcher.

## Native Autostart

On macOS you can install a `launchd` agent so the watcher starts automatically
and keeps writing hook artifacts in the background:

```bash
./codex-memory autostart install --cwd /Users/alex/Desktop/dev --emit-dir /Users/alex/.codex/memory/hook-runtime/dev
./codex-memory autostart status --cwd /Users/alex/Desktop/dev
./codex-memory autostart remove --cwd /Users/alex/Desktop/dev
```

Add `--load` to `install` if you want the agent loaded immediately with
`launchctl`.

`autostart install` now also stages a self-contained runtime under the chosen
emit directory:

- `bundle/` contains a runnable copy of `codex_memory`
- `vendor/` contains the Python dependencies needed by the background runtime
- `logs/` is reset on reinstall so stale permission errors do not linger

The installed watcher also runs with `--emit-session-end-on-exit` so graceful
shutdowns write a final `SessionEnd` payload.

## Notes

- Qdrant stores the real embedding vectors under
  [`.data/qdrant`](/Users/alex/Desktop/dev/codex-memory/.data/qdrant).
- Local model weights are cached under
  [`.data/fastembed-cache`](/Users/alex/Desktop/dev/codex-memory/.data/fastembed-cache).
- A hashed fallback vector is still stored in SQLite so search can degrade
  gracefully if embeddings are temporarily unavailable.
- FTS5 is used when available; if the local SQLite build lacks FTS5, search
  still works with vector and token overlap scoring.
- Re-running `index` is incremental and only refreshes changed threads unless
  `--force` is supplied.
- If you want the database somewhere else, pass `--db /path/to/db.sqlite`.
- With the default `fastembed` provider, `index`, `search`, and `context`
  download the model once and then run locally.
