---
name: codex-memory
description: Search local Codex conversation history and bring relevant past work into the current task. Use when the user refers to previous work, asks "we did this before", wants project continuity, or needs a quick recap of recent sessions.
metadata:
  argument-hint: "[query or latest]"
---

# Codex Memory

Use this skill to recall prior Codex work from the local memory index built from
`~/.codex/sessions` and `state_*.sqlite`.

This project now uses local `fastembed` embeddings plus a local Qdrant vector
store by default. The first run downloads the model once; after that, semantic
search is local.

## When To Use It

- The user refers to a previous conversation or asks what we did before.
- You want context from recent work in the current project.
- You need to find commands, file paths, or outcomes from an earlier session.

## Workflow

1. Ensure the index exists:

```bash
/Users/alex/Desktop/dev/codex-memory/codex-memory status
```

If it is empty, build it:

```bash
/Users/alex/Desktop/dev/codex-memory/codex-memory index
```

2. Search by topic:

```bash
/Users/alex/Desktop/dev/codex-memory/codex-memory search "<query>" --cwd "$PWD"
```

3. Get a compact project handoff block for a fresh thread:

```bash
/Users/alex/Desktop/dev/codex-memory/codex-memory context "<query>" --cwd "$PWD"
```

If no query is needed, use recent project context:

```bash
/Users/alex/Desktop/dev/codex-memory/codex-memory context --cwd "$PWD"
```

4. Inspect a specific thread:

```bash
/Users/alex/Desktop/dev/codex-memory/codex-memory brief latest
```

## Response Guidance

- Summarize the most relevant thread or threads in plain language.
- Call out commands, files, and outcomes that matter to the current task.
- If nothing solid matches, say that clearly rather than over-claiming.
