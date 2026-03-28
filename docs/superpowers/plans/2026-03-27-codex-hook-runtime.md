# Codex Hook Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `claude-mem`-style hook shim to `codex-memory` with event names aligned to Claude Code lifecycle hooks.

**Architecture:** Add a dedicated hook runtime module that maps lifecycle events to incremental sync plus hook-specific rendering. Wire the runtime into the CLI through a `hook` command and verify behavior with focused unit tests.

**Tech Stack:** Python 3, argparse, SQLite, existing `codex-memory` index/render pipeline.

---

### Task 1: Add Hook Runtime Tests

**Files:**
- Modify: `/Users/alex/Desktop/dev/codex-memory/tests/test_memory_quality.py`
- Test: `/Users/alex/Desktop/dev/codex-memory/tests/test_memory_quality.py`

- [ ] **Step 1: Write failing tests for hook parsing and behavior**
- [ ] **Step 2: Run `./.venv/bin/python -m unittest discover -s tests -v` and verify failure**
- [ ] **Step 3: Implement minimal runtime and CLI wiring**
- [ ] **Step 4: Run `./.venv/bin/python -m unittest discover -s tests -v` and verify pass**

### Task 2: Implement Hook Runtime

**Files:**
- Create: `/Users/alex/Desktop/dev/codex-memory/codex_memory/hook_runtime.py`
- Modify: `/Users/alex/Desktop/dev/codex-memory/codex_memory/cli.py`
- Modify: `/Users/alex/Desktop/dev/codex-memory/codex_memory/indexer.py`

- [ ] **Step 1: Add hook event constants and dispatcher**
- [ ] **Step 2: Implement `SessionStart` context injection behavior**
- [ ] **Step 3: Implement ingestion behavior for the remaining hook events**
- [ ] **Step 4: Keep output concise in both text and JSON modes**

### Task 3: Document and Verify

**Files:**
- Modify: `/Users/alex/Desktop/dev/codex-memory/README.md`

- [ ] **Step 1: Add hook command examples**
- [ ] **Step 2: Run `./codex-memory hook SessionStart --cwd /Users/alex/Desktop/dev`**
- [ ] **Step 3: Run `./codex-memory hook PostToolUse --cwd /Users/alex/Desktop/dev`**
- [ ] **Step 4: Summarize remaining gap vs official hooks**
