import json
import os
import sqlite3
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from codex_memory.codex_data import (
    ThreadRecord,
    choose_latest_focus,
    choose_outcome,
    clean_message_text,
    extract_thread,
    summarize_outcome_text,
)
from codex_memory.indexer import (
    open_db,
    recent_threads,
    render_context,
    render_timeline,
    search_threads,
    status,
    sync_latest_threads,
)
from codex_memory.cli import build_parser, main
from codex_memory.autostart import autostart_status, install_autostart, launchd_label_for_cwd
from codex_memory.hook_sink import emit_hook_payload
from codex_memory.hook_watch import HookWatchState, run_watch_iteration, watch_hooks
from codex_memory.hook_runtime import run_hook_event
from codex_memory.vectorizer import encode_vector, text_to_vector


class MemoryQualityTests(unittest.TestCase):
    def test_clean_message_text_strips_file_wrapper_and_heading_markup(self):
        raw = """
# Files mentioned by the user:

## example.jpg: /tmp/example.jpg

## My request for Codex:
## 把这写skills都装上
"""

        self.assertEqual(clean_message_text(raw), "把这写skills都装上")

    def test_choose_latest_focus_skips_meta_evaluation_follow_up(self):
        messages = [
            "把这写skills都装上",
            "claude-mem 你搜搜这个",
            "它能不能改造成一个适合 Codex 的本地 memory 方案",
            "升级成真 embedding + 真向量库版本。",
            "好的，可以 然后你再对比你一下Claude哪个版本 我们做得如何 还需要补充什么",
        ]

        self.assertEqual(choose_latest_focus(messages), "升级成真 embedding + 真向量库版本。")

    def test_choose_outcome_prefers_task_result_over_meta_eval(self):
        final_messages = [
            "已经切到本地 fastembed + Qdrant，而且默认走本地 embedding。",
            "现在这套已经好用了，而且比重启前明显更像真 memory。后面还可以继续对标 claude-mem。",
        ]

        self.assertEqual(
            choose_outcome(final_messages, []),
            "已经切到本地 fastembed + Qdrant，而且默认走本地 embedding。",
        )

    def test_extract_thread_uses_final_answer_phase_for_outcome(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_path = os.path.join(temp_dir, "rollout.jsonl")
            events = [
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "user_message",
                        "message": """
# Files mentioned by the user:

## image.jpg: /tmp/image.jpg

## My request for Codex:
## 把这写skills都装上
""",
                        "images": [],
                        "local_images": [],
                        "text_elements": [],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": [{"type": "output_text", "text": "我先安装并检查结构。"}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "已经升成真 embedding + 真向量库了。",
                            }
                        ],
                    },
                },
            ]
            with open(rollout_path, "w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")

            thread = ThreadRecord(
                id="thread-1",
                rollout_path=rollout_path,
                created_at=int(time.time()),
                updated_at=int(time.time()),
                source="test",
                cwd="/Users/alex/Desktop/dev",
                title="把这写skills都装上",
                first_user_message="把这写skills都装上",
                model="gpt-test",
            )

            extracted = extract_thread(thread)

            self.assertEqual(extracted.final_answer, "已经升成真 embedding + 真向量库了。")
            self.assertIn("User asked: 把这写skills都装上.", extracted.summary)
            self.assertIn("Outcome: 已经升成真 embedding + 真向量库了。", extracted.summary)
            self.assertIn("answer", [item.item_type for item in extracted.memory_items])

    def test_extract_thread_captures_decision_and_tool_observation_items(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_path = os.path.join(temp_dir, "rollout.jsonl")
            events = [
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "user_message",
                        "message": "升级成真 embedding + 真向量库版本。",
                        "images": [],
                        "local_images": [],
                        "text_elements": [],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": [{"type": "output_text", "text": "我把默认 provider 改成 fastembed。"}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "call-1",
                        "output": (
                            "Command: /bin/zsh -lc \"rg -n 'chromadb' README.md\"\n"
                            "Process exited with code 0\n"
                            "Output:\n"
                            "chromadb: no\n"
                        ),
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "call-2",
                        "output": (
                            "Command: /bin/zsh -lc './codex-memory status'\n"
                            "Process exited with code 0\n"
                            "Output:\n"
                            "vector backend: qdrant\n"
                            "embedding dimensions: 384\n"
                            "vector points: 22\n"
                        ),
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": "已经切到本地 fastembed + Qdrant。"}],
                    },
                },
            ]
            with open(rollout_path, "w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")

            thread = ThreadRecord(
                id="thread-obs",
                rollout_path=rollout_path,
                created_at=int(time.time()),
                updated_at=int(time.time()),
                source="test",
                cwd="/Users/alex/Desktop/dev",
                title="升级 memory",
                first_user_message="升级成真 embedding + 真向量库版本。",
                model="gpt-test",
            )

            extracted = extract_thread(thread)
            items_by_type = {}
            for item in extracted.memory_items:
                items_by_type.setdefault(item.item_type, []).append(item.text)

            self.assertIn("decision", items_by_type)
            self.assertIn("observation", items_by_type)
            self.assertTrue(any("fastembed" in text for text in items_by_type["decision"]))
            self.assertTrue(any("vector backend: qdrant" in text for text in items_by_type["observation"]))
            self.assertFalse(any("chromadb: no" in text for text in items_by_type["observation"]))

    def test_summarize_outcome_text_prefers_first_sentence(self):
        text = (
            "已经升成真 embedding + 真向量库了，而且现在默认走本地，不再依赖你的中转。 "
            "**结果** - 向量后端：Qdrant - 向量维度：384"
        )

        self.assertEqual(
            summarize_outcome_text(text),
            "已经升成真 embedding + 真向量库了，而且现在默认走本地，不再依赖你的中转。",
        )

    def test_search_prefers_top_scoring_matches_over_first_items(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "memory.sqlite")
            connection = open_db(db_path)
            try:
                now = int(time.time())
                connection.execute(
                    """
                    INSERT INTO threads (
                        id, title, cwd, source, model, rollout_path, created_at, updated_at,
                        indexed_at, first_user_message, summary, final_answer,
                        commands_json, files_json, tool_names_json, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "thread-1",
                        "把这写skills都装上",
                        "/Users/alex/Desktop/dev",
                        "test",
                        "gpt-test",
                        "/tmp/rollout.jsonl",
                        now,
                        now,
                        now,
                        "把这写skills都装上",
                        "Thread about local memory improvements.",
                        "已经升级完成。",
                        "[]",
                        "[]",
                        "[]",
                        "hash-1",
                    ),
                )

                items = [
                    ("summary", "Thread about local memory improvements."),
                    ("title", "Codex memory"),
                    ("user", "claude-mem 能不能改造成一个适合 Codex 的本地 memory 方案"),
                    ("assistant", "我们已经把 Codex memory 升级成真 embedding + 真向量库，并参考 claude-mem 的做法。"),
                ]
                for ordinal, (item_type, text) in enumerate(items):
                    cursor = connection.execute(
                        """
                        INSERT INTO memory_items (
                            thread_id, ordinal, item_type, text, preview, vector_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "thread-1",
                            ordinal,
                            item_type,
                            text,
                            text,
                            encode_vector(text_to_vector(text)),
                            now,
                        ),
                    )
                    row_id = int(cursor.lastrowid)
                    connection.execute(
                        "INSERT INTO memory_fts(rowid, text, thread_id, item_type) VALUES (?, ?, ?, ?)",
                        (row_id, text, "thread-1", item_type),
                    )
                connection.commit()
            finally:
                connection.close()

            results = search_threads(
                db_path=db_path,
                query="claude-mem Codex memory",
                cwd="/Users/alex/Desktop/dev",
                limit=1,
            )

            self.assertEqual(len(results), 1)
            match_types = [match["item_type"] for match in results[0]["matches"]]
            self.assertIn("assistant", match_types)
            self.assertIn("user", match_types)
            self.assertNotEqual(match_types[:2], ["summary", "title"])

    def test_recent_context_renders_decisions_and_observations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "memory.sqlite")
            connection = open_db(db_path)
            try:
                now = int(time.time())
                connection.execute(
                    """
                    INSERT INTO threads (
                        id, title, cwd, source, model, rollout_path, created_at, updated_at,
                        indexed_at, first_user_message, summary, final_answer,
                        commands_json, files_json, tool_names_json, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "thread-ctx",
                        "升级 memory",
                        "/Users/alex/Desktop/dev",
                        "test",
                        "gpt-test",
                        "/tmp/rollout.jsonl",
                        now,
                        now,
                        now,
                        "升级成真 embedding + 真向量库版本。",
                        "Thread: 升级 memory. Outcome: 已经切到本地 fastembed + Qdrant。",
                        "已经切到本地 fastembed + Qdrant。",
                        json.dumps(["./codex-memory status"]),
                        json.dumps(["/Users/alex/Desktop/dev/codex-memory/codex_memory/indexer.py"]),
                        "[]",
                        "hash-ctx",
                    ),
                )
                items = [
                    ("decision", "默认 provider 改成 fastembed。"),
                    ("observation", "vector backend: qdrant; vector points: 22"),
                ]
                for ordinal, (item_type, text) in enumerate(items):
                    connection.execute(
                        """
                        INSERT INTO memory_items (
                            thread_id, ordinal, item_type, text, preview, vector_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "thread-ctx",
                            ordinal,
                            item_type,
                            text,
                            text,
                            encode_vector(text_to_vector(text)),
                            now,
                        ),
                    )
                connection.commit()
            finally:
                connection.close()

            results = recent_threads(db_path=db_path, cwd="/Users/alex/Desktop/dev", limit=1)
            rendered = render_context(results)

            self.assertIn("decision: 默认 provider 改成 fastembed。", rendered)
            self.assertIn("observation: vector backend: qdrant; vector points: 22", rendered)

    def test_render_timeline_shows_structured_memory_flow(self):
        brief = {
            "title": "升级 memory",
            "thread_id": "thread-timeline",
            "updated_at": int(time.time()),
            "items": [
                {"item_type": "user", "text": "升级成真 embedding + 真向量库版本。", "preview": "升级成真 embedding + 真向量库版本。"},
                {"item_type": "decision", "text": "默认 provider 改成 fastembed。", "preview": "默认 provider 改成 fastembed。"},
                {"item_type": "observation", "text": "vector backend: qdrant", "preview": "vector backend: qdrant"},
                {"item_type": "result", "text": "已经切到本地 fastembed + Qdrant。", "preview": "已经切到本地 fastembed + Qdrant。"},
            ],
        }

        rendered = render_timeline(brief)

        self.assertIn("timeline:", rendered)
        self.assertIn("1. user: 升级成真 embedding + 真向量库版本。", rendered)
        self.assertIn("2. decision: 默认 provider 改成 fastembed。", rendered)
        self.assertIn("3. observation: vector backend: qdrant", rendered)
        self.assertIn("4. result: 已经切到本地 fastembed + Qdrant。", rendered)

    def test_cli_includes_sync_command(self):
        parser = build_parser()
        args = parser.parse_args(["sync"])
        self.assertEqual(args.command, "sync")

    def test_cli_includes_watch_command(self):
        parser = build_parser()
        args = parser.parse_args(["watch", "--max-loops", "1"])
        self.assertEqual(args.command, "watch")
        self.assertEqual(args.max_loops, 1)

    def test_cli_includes_autostart_install_command(self):
        parser = build_parser()
        args = parser.parse_args(["autostart", "install", "--cwd", "/Users/alex/Desktop/dev"])
        self.assertEqual(args.command, "autostart")
        self.assertEqual(args.autostart_command, "install")
        self.assertEqual(args.cwd, "/Users/alex/Desktop/dev")

    def test_cli_autostart_install_uses_autostart_db_default(self):
        with mock.patch("codex_memory.cli.install_autostart") as install_mock:
            install_mock.return_value = {
                "label": "test",
                "cwd": "/Users/alex/Desktop/dev",
                "plist_path": "/tmp/test.plist",
                "emit_dir": "/tmp/runtime",
                "installed": True,
                "loaded": False,
            }

            exit_code = main(["autostart", "install", "--cwd", "/Users/alex/Desktop/dev"])

            self.assertEqual(exit_code, 0)
            self.assertIsNone(install_mock.call_args.kwargs["db_path"])

    def test_sync_latest_threads_only_indexes_changed_threads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            codex_home = os.path.join(temp_dir, ".codex")
            os.makedirs(codex_home, exist_ok=True)
            state_db_path = os.path.join(codex_home, "state_test.sqlite")
            connection = sqlite3.connect(state_db_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE threads (
                        id TEXT PRIMARY KEY,
                        rollout_path TEXT NOT NULL,
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        source TEXT NOT NULL,
                        cwd TEXT NOT NULL,
                        title TEXT NOT NULL,
                        first_user_message TEXT NOT NULL,
                        model TEXT
                    )
                    """
                )
                rollout_old = os.path.join(temp_dir, "old.jsonl")
                rollout_new = os.path.join(temp_dir, "new.jsonl")
                for rollout_path, user_text, final_text in (
                    (rollout_old, "旧任务", "旧任务完成。"),
                    (rollout_new, "新任务", "新任务完成。"),
                ):
                    with open(rollout_path, "w", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "type": "event_msg",
                                    "payload": {
                                        "type": "user_message",
                                        "message": user_text,
                                        "images": [],
                                        "local_images": [],
                                        "text_elements": [],
                                    },
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        handle.write(
                            json.dumps(
                                {
                                    "type": "response_item",
                                    "payload": {
                                        "type": "message",
                                        "role": "assistant",
                                        "phase": "final_answer",
                                        "content": [{"type": "output_text", "text": final_text}],
                                    },
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                now = int(time.time())
                connection.executemany(
                    """
                    INSERT INTO threads (
                        id, rollout_path, created_at, updated_at, source, cwd, title, first_user_message, model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "thread-old",
                            rollout_old,
                            now - 60,
                            now - 60,
                            "test",
                            "/Users/alex/Desktop/dev",
                            "旧任务",
                            "旧任务",
                            "gpt-test",
                        ),
                        (
                            "thread-new",
                            rollout_new,
                            now,
                            now,
                            "test",
                            "/Users/alex/Desktop/dev",
                            "新任务",
                            "新任务",
                            "gpt-test",
                        ),
                    ],
                )
                connection.commit()
            finally:
                connection.close()

            db_path = os.path.join(temp_dir, "memory.sqlite")
            first = sync_latest_threads(codex_home=codex_home, db_path=db_path, limit=1)
            second = sync_latest_threads(codex_home=codex_home, db_path=db_path, limit=1)

            self.assertEqual(first["indexed"], 1)
            self.assertEqual(first["thread_ids"], ["thread-new"])
            self.assertEqual(second["indexed"], 0)
            self.assertEqual(second["stale_thread_ids"], [])

    def test_hook_session_start_returns_context_block(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            codex_home = os.path.join(temp_dir, ".codex")
            os.makedirs(codex_home, exist_ok=True)
            state_db_path = os.path.join(codex_home, "state_test.sqlite")
            rollout_path = os.path.join(temp_dir, "thread.jsonl")
            with open(rollout_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "type": "event_msg",
                            "payload": {
                                "type": "user_message",
                                "message": "升级成真 embedding + 真向量库版本。",
                                "images": [],
                                "local_images": [],
                                "text_elements": [],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.write(
                    json.dumps(
                        {
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "assistant",
                                "phase": "final_answer",
                                "content": [{"type": "output_text", "text": "已经切到本地 fastembed + Qdrant。"}],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            connection = sqlite3.connect(state_db_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE threads (
                        id TEXT PRIMARY KEY,
                        rollout_path TEXT NOT NULL,
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        source TEXT NOT NULL,
                        cwd TEXT NOT NULL,
                        title TEXT NOT NULL,
                        first_user_message TEXT NOT NULL,
                        model TEXT
                    )
                    """
                )
                now = int(time.time())
                connection.execute(
                    """
                    INSERT INTO threads (
                        id, rollout_path, created_at, updated_at, source, cwd, title, first_user_message, model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "thread-hook",
                        rollout_path,
                        now,
                        now,
                        "test",
                        "/Users/alex/Desktop/dev",
                        "升级 memory",
                        "升级成真 embedding + 真向量库版本。",
                        "gpt-test",
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            payload = run_hook_event(
                "SessionStart",
                codex_home=codex_home,
                db_path=os.path.join(temp_dir, "memory.sqlite"),
                cwd="/Users/alex/Desktop/dev",
            )

            self.assertEqual(payload["event"], "SessionStart")
            self.assertEqual(payload["mode"], "inject")
            self.assertIn("Relevant Codex memory:", payload["text"])
            self.assertIn("state:", payload["text"])

    def test_watch_iteration_emits_session_start_for_new_thread(self):
        calls = []

        def hook_runner(event, **kwargs):
            calls.append((event, kwargs.get("thread_ref")))
            return {"event": event}

        state = HookWatchState()
        threads = [
            ThreadRecord(
                id="thread-1",
                rollout_path="/tmp/thread-1.jsonl",
                created_at=1,
                updated_at=2,
                source="test",
                cwd="/Users/alex/Desktop/dev",
                title="t1",
                first_user_message="hello",
                model="gpt-test",
            )
        ]

        next_state, payloads = run_watch_iteration(
            state=state,
            threads=threads,
            hook_runner=hook_runner,
            event_reader=lambda path, offset: (offset, []),
            cwd="/Users/alex/Desktop/dev",
        )

        self.assertEqual(calls, [("SessionStart", "thread-1")])
        self.assertEqual(next_state.active_thread_id, "thread-1")
        self.assertEqual(len(payloads), 1)

    def test_watch_iteration_maps_rollout_changes_to_hook_events(self):
        calls = []

        def hook_runner(event, **kwargs):
            calls.append((event, kwargs.get("thread_ref")))
            return {"event": event}

        thread = ThreadRecord(
            id="thread-1",
            rollout_path="/tmp/thread-1.jsonl",
            created_at=1,
            updated_at=3,
            source="test",
            cwd="/Users/alex/Desktop/dev",
            title="t1",
            first_user_message="hello",
            model="gpt-test",
        )
        state = HookWatchState(active_thread_id="thread-1", rollout_offsets={"thread-1": 10})
        new_events = [
            {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
            {"type": "response_item", "payload": {"type": "function_call_output", "output": "ok"}},
            {"type": "event_msg", "payload": {"type": "task_complete"}},
        ]

        next_state, payloads = run_watch_iteration(
            state=state,
            threads=[thread],
            hook_runner=hook_runner,
            event_reader=lambda path, offset: (99, new_events),
            cwd="/Users/alex/Desktop/dev",
        )

        self.assertEqual(
            calls,
            [("UserPromptSubmit", "thread-1"), ("PostToolUse", "thread-1"), ("Stop", "thread-1")],
        )
        self.assertEqual(next_state.rollout_offsets["thread-1"], 99)
        self.assertEqual(len(payloads), 3)

    def test_watch_iteration_emits_session_end_then_start_on_thread_switch(self):
        calls = []

        def hook_runner(event, **kwargs):
            calls.append((event, kwargs.get("thread_ref")))
            return {"event": event}

        old_thread = ThreadRecord(
            id="thread-old",
            rollout_path="/tmp/old.jsonl",
            created_at=1,
            updated_at=2,
            source="test",
            cwd="/Users/alex/Desktop/dev",
            title="old",
            first_user_message="old",
            model="gpt-test",
        )
        new_thread = ThreadRecord(
            id="thread-new",
            rollout_path="/tmp/new.jsonl",
            created_at=3,
            updated_at=4,
            source="test",
            cwd="/Users/alex/Desktop/dev",
            title="new",
            first_user_message="new",
            model="gpt-test",
        )
        state = HookWatchState(active_thread_id="thread-old", rollout_offsets={"thread-old": 10})

        next_state, payloads = run_watch_iteration(
            state=state,
            threads=[new_thread, old_thread],
            hook_runner=hook_runner,
            event_reader=lambda path, offset: (offset, []),
            cwd="/Users/alex/Desktop/dev",
        )

        self.assertEqual(calls, [("SessionEnd", "thread-old"), ("SessionStart", "thread-new")])
        self.assertEqual(next_state.active_thread_id, "thread-new")
        self.assertEqual(len(payloads), 2)

    def test_emit_hook_payload_writes_latest_files_and_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = {
                "event": "SessionStart",
                "mode": "inject",
                "text": "Relevant Codex memory:\n- hello",
                "sync": {"indexed": 1},
            }

            paths = emit_hook_payload(payload, temp_dir)

            self.assertTrue(os.path.exists(paths["events_log"]))
            self.assertTrue(os.path.exists(paths["latest_json"]))
            self.assertTrue(os.path.exists(paths["latest_text"]))
            with open(paths["latest_text"], "r", encoding="utf-8") as handle:
                self.assertIn("Relevant Codex memory:", handle.read())
            with open(paths["events_log"], "r", encoding="utf-8") as handle:
                self.assertIn('"event": "SessionStart"', handle.read())

    def test_watch_hooks_can_emit_payload_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            thread = ThreadRecord(
                id="thread-1",
                rollout_path="/tmp/thread-1.jsonl",
                created_at=1,
                updated_at=2,
                source="test",
                cwd="/Users/alex/Desktop/dev",
                title="t1",
                first_user_message="hello",
                model="gpt-test",
            )

            payloads = []

            def hook_runner(event, **kwargs):
                payload = {"event": event, "mode": "inject", "text": "hello", "sync": {"indexed": 0}}
                payloads.append(payload)
                return payload

            collected = watch_hooks(
                cwd="/Users/alex/Desktop/dev",
                max_loops=1,
                emit_dir=temp_dir,
                hook_runner=hook_runner,
                thread_fetcher=lambda **kwargs: [thread],
                event_reader=lambda path, offset: (offset, []),
            )

            self.assertEqual(len(collected), 1)
            self.assertEqual(collected[0]["event"], "SessionStart")
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "latest", "SessionStart.txt")))

    def test_watch_hooks_can_emit_session_end_on_exit(self):
        thread = ThreadRecord(
            id="thread-1",
            rollout_path="/tmp/thread-1.jsonl",
            created_at=1,
            updated_at=2,
            source="test",
            cwd="/Users/alex/Desktop/dev",
            title="t1",
            first_user_message="hello",
            model="gpt-test",
        )
        calls = []

        def hook_runner(event, **kwargs):
            calls.append((event, kwargs.get("thread_ref")))
            return {"event": event, "mode": "sync", "text": event, "sync": {"indexed": 0}}

        collected = watch_hooks(
            cwd="/Users/alex/Desktop/dev",
            max_loops=1,
            emit_shutdown_event=True,
            hook_runner=hook_runner,
            thread_fetcher=lambda **kwargs: [thread],
            event_reader=lambda path, offset: (offset, []),
        )

        self.assertEqual([payload["event"] for payload in collected], ["SessionStart", "SessionEnd"])
        self.assertEqual(calls, [("SessionStart", "thread-1"), ("SessionEnd", "thread-1")])

    def test_autostart_install_writes_launchd_plist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = "/Users/alex/Desktop/dev/codex-memory"
            cwd = "/Users/alex/Desktop/dev"
            emit_dir = os.path.join(temp_dir, "runtime")
            vendor_dir = os.path.join(emit_dir, "vendor")

            payload = install_autostart(
                cwd=cwd,
                repo_root=repo_root,
                emit_dir=emit_dir,
                launch_agents_dir=temp_dir,
                load=False,
                bootstrap_runtime=False,
            )

            self.assertEqual(payload["cwd"], cwd)
            self.assertTrue(os.path.exists(payload["plist_path"]))
            self.assertTrue(os.path.exists(os.path.join(payload["bundle_root"], "codex_memory", "__main__.py")))
            with open(payload["plist_path"], "r", encoding="utf-8") as handle:
                plist_text = handle.read()
            self.assertIn("python3", plist_text)
            self.assertIn("PYTHONPATH", plist_text)
            self.assertNotIn(vendor_dir, plist_text)
            self.assertIn("watch", plist_text)
            self.assertIn("RunAtLoad", plist_text)

    def test_autostart_install_includes_vendor_path_when_bootstrapped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = "/Users/alex/Desktop/dev/codex-memory"
            cwd = "/Users/alex/Desktop/dev"
            emit_dir = os.path.join(temp_dir, "runtime")
            vendor_dir = os.path.join(emit_dir, "vendor")

            payload = install_autostart(
                cwd=cwd,
                repo_root=repo_root,
                emit_dir=emit_dir,
                launch_agents_dir=temp_dir,
                load=False,
                bootstrapper=lambda emit_dir, source_root=None: vendor_dir,
            )

            self.assertEqual(payload["vendor_root"], vendor_dir)
            with open(payload["plist_path"], "r", encoding="utf-8") as handle:
                plist_text = handle.read()
            self.assertIn(vendor_dir, plist_text)

    def test_autostart_status_reports_installed_plist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cwd = "/Users/alex/Desktop/dev"
            label = launchd_label_for_cwd(cwd)
            plist_path = os.path.join(temp_dir, "{}.plist".format(label))
            with open(plist_path, "w", encoding="utf-8") as handle:
                handle.write("plist")

            payload = autostart_status(
                cwd=cwd,
                launch_agents_dir=temp_dir,
                emit_dir=os.path.join(temp_dir, "runtime"),
            )

            self.assertTrue(payload["installed"])
            self.assertEqual(payload["label"], label)
            self.assertEqual(payload["plist_path"], plist_path)

    def test_staged_autostart_bundle_runs_with_system_python(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            codex_home = os.path.join(temp_dir, ".codex")
            os.makedirs(codex_home, exist_ok=True)
            state_db_path = os.path.join(codex_home, "state_test.sqlite")
            rollout_path = os.path.join(temp_dir, "thread.jsonl")
            with open(rollout_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "type": "event_msg",
                            "payload": {
                                "type": "user_message",
                                "message": "native hook bundle test",
                                "images": [],
                                "local_images": [],
                                "text_elements": [],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            connection = sqlite3.connect(state_db_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE threads (
                        id TEXT PRIMARY KEY,
                        rollout_path TEXT NOT NULL,
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        source TEXT NOT NULL,
                        cwd TEXT NOT NULL,
                        title TEXT NOT NULL,
                        first_user_message TEXT NOT NULL,
                        model TEXT
                    )
                    """
                )
                now = int(time.time())
                connection.execute(
                    """
                    INSERT INTO threads (
                        id, rollout_path, created_at, updated_at, source, cwd, title, first_user_message, model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "thread-bundle",
                        rollout_path,
                        now,
                        now,
                        "test",
                        "/Users/alex/Desktop/dev",
                        "bundle test",
                        "native hook bundle test",
                        "gpt-test",
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            launch_agents_dir = os.path.join(temp_dir, "agents")
            emit_dir = os.path.join(temp_dir, "runtime")
            install_payload = install_autostart(
                cwd="/Users/alex/Desktop/dev",
                repo_root="/Users/alex/Desktop/dev/codex-memory",
                emit_dir=emit_dir,
                launch_agents_dir=launch_agents_dir,
                codex_home=codex_home,
                load=False,
                bootstrap_runtime=False,
            )

            bundle_root = os.path.join(install_payload["emit_dir"], "bundle")
            run = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-m",
                    "codex_memory",
                    "watch",
                    "--codex-home",
                    codex_home,
                    "--db",
                    os.path.join(temp_dir, "memory.sqlite"),
                    "--cwd",
                    "/Users/alex/Desktop/dev",
                    "--emit-dir",
                    os.path.join(temp_dir, "emitted"),
                    "--max-loops",
                    "1",
                ],
                env={"PYTHONPATH": bundle_root},
                capture_output=True,
                text=True,
                cwd=temp_dir,
            )

            self.assertEqual(run.returncode, 0, msg=run.stderr)
            self.assertIn("SessionStart", run.stdout)

    def test_hook_post_tool_use_runs_incremental_sync(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            codex_home = os.path.join(temp_dir, ".codex")
            os.makedirs(codex_home, exist_ok=True)
            state_db_path = os.path.join(codex_home, "state_test.sqlite")
            rollout_path = os.path.join(temp_dir, "thread.jsonl")
            with open(rollout_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "type": "event_msg",
                            "payload": {
                                "type": "user_message",
                                "message": "新任务",
                                "images": [],
                                "local_images": [],
                                "text_elements": [],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            connection = sqlite3.connect(state_db_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE threads (
                        id TEXT PRIMARY KEY,
                        rollout_path TEXT NOT NULL,
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        source TEXT NOT NULL,
                        cwd TEXT NOT NULL,
                        title TEXT NOT NULL,
                        first_user_message TEXT NOT NULL,
                        model TEXT
                    )
                    """
                )
                now = int(time.time())
                connection.execute(
                    """
                    INSERT INTO threads (
                        id, rollout_path, created_at, updated_at, source, cwd, title, first_user_message, model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "thread-post",
                        rollout_path,
                        now,
                        now,
                        "test",
                        "/Users/alex/Desktop/dev",
                        "新任务",
                        "新任务",
                        "gpt-test",
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            payload = run_hook_event(
                "PostToolUse",
                codex_home=codex_home,
                db_path=os.path.join(temp_dir, "memory.sqlite"),
                cwd="/Users/alex/Desktop/dev",
            )

            self.assertEqual(payload["event"], "PostToolUse")
            self.assertEqual(payload["mode"], "sync")
            self.assertEqual(payload["sync"]["indexed"], 1)
            self.assertEqual(payload["sync"]["thread_ids"], ["thread-post"])

    def test_sync_latest_threads_falls_back_when_qdrant_is_locked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            codex_home = os.path.join(temp_dir, ".codex")
            os.makedirs(codex_home, exist_ok=True)
            state_db_path = os.path.join(codex_home, "state_test.sqlite")
            rollout_path = os.path.join(temp_dir, "thread.jsonl")
            with open(rollout_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "type": "event_msg",
                            "payload": {
                                "type": "user_message",
                                "message": "锁冲突回退测试",
                                "images": [],
                                "local_images": [],
                                "text_elements": [],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            connection = sqlite3.connect(state_db_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE threads (
                        id TEXT PRIMARY KEY,
                        rollout_path TEXT NOT NULL,
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        source TEXT NOT NULL,
                        cwd TEXT NOT NULL,
                        title TEXT NOT NULL,
                        first_user_message TEXT NOT NULL,
                        model TEXT
                    )
                    """
                )
                now = int(time.time())
                connection.execute(
                    """
                    INSERT INTO threads (
                        id, rollout_path, created_at, updated_at, source, cwd, title, first_user_message, model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "thread-lock",
                        rollout_path,
                        now,
                        now,
                        "test",
                        "/Users/alex/Desktop/dev",
                        "锁冲突测试",
                        "锁冲突回退测试",
                        "gpt-test",
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            db_path = os.path.join(temp_dir, "memory.sqlite")
            with mock.patch("codex_memory.indexer.open_qdrant", side_effect=RuntimeError("locked")):
                payload = sync_latest_threads(
                    codex_home=codex_home,
                    db_path=db_path,
                    limit=1,
                    cwd="/Users/alex/Desktop/dev",
                )

            self.assertEqual(payload["indexed"], 1)
            db = open_db(db_path)
            try:
                count = db.execute("SELECT COUNT(*) AS count FROM memory_items").fetchone()["count"]
            finally:
                db.close()
            self.assertGreater(count, 0)

    def test_status_can_read_while_writer_holds_immediate_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "memory.sqlite")
            db = open_db(db_path)
            try:
                db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('last_indexed_at', '1')")
                db.commit()
            finally:
                db.close()

            writer = sqlite3.connect(db_path, timeout=1)
            try:
                writer.execute("PRAGMA journal_mode=WAL")
                writer.execute("BEGIN IMMEDIATE")
                writer.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('last_indexed_at', '2')")

                payload = status(db_path)
            finally:
                writer.rollback()
                writer.close()

            self.assertEqual(payload["db_path"], db_path)
            self.assertEqual(payload["thread_count"], 0)

    def test_status_uses_cached_vector_point_count_when_qdrant_is_locked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "memory.sqlite")
            db = open_db(db_path)
            try:
                db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('vector_backend', 'qdrant')")
                db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('vector_qdrant_path', ?)", (os.path.join(temp_dir, "qdrant"),))
                db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('vector_point_count', '42')")
                db.commit()
            finally:
                db.close()

            with mock.patch("codex_memory.indexer.open_qdrant", side_effect=RuntimeError("locked")):
                payload = status(db_path)

            self.assertEqual(payload["vector_points"], 42)
            self.assertEqual(payload["vector_lock_error"], "locked")


if __name__ == "__main__":
    unittest.main()
