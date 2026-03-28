import argparse
import json
import os
import sys

from codex_memory.autostart import autostart_status, install_autostart, remove_autostart
from codex_memory.codex_data import default_codex_home
from codex_memory.hook_watch import iter_watch_hooks, watch_hooks
from codex_memory.hook_runtime import HOOK_EVENTS, run_hook_event
from codex_memory.indexer import (
    default_db_path,
    ensure_populated,
    format_timestamp,
    index_threads,
    recent_threads,
    render_brief,
    render_context,
    render_search_results,
    render_sync,
    render_timeline,
    resolve_thread,
    search_threads,
    status,
    sync_latest_threads,
)


def _common_storage_args(parser):
    parser.add_argument(
        "--codex-home",
        default=default_codex_home(),
        help="Path to the Codex home directory (default: ~/.codex)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to the memory SQLite database (default: project-local .data/codex-memory.sqlite)",
    )


def _json_or_text(payload, as_json, renderer=None):
    if as_json:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    if renderer:
        return renderer(payload)
    return str(payload)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="codex-memory",
        description="Local hybrid memory index for Codex Desktop sessions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index Codex thread history into the local memory DB")
    _common_storage_args(index_parser)
    index_parser.add_argument("--limit", type=int, default=None, help="Only index the N most recent threads")
    index_parser.add_argument("--force", action="store_true", help="Rebuild indexed threads even if unchanged")
    index_parser.add_argument("--thread-id", action="append", dest="thread_ids", default=None, help="Index a specific thread id")
    index_parser.add_argument("--json", action="store_true", help="Emit JSON")

    status_parser = subparsers.add_parser("status", help="Show memory index status")
    _common_storage_args(status_parser)
    status_parser.add_argument("--json", action="store_true", help="Emit JSON")

    sync_parser = subparsers.add_parser("sync", help="Incrementally ingest only the latest changed threads")
    _common_storage_args(sync_parser)
    sync_parser.add_argument("--limit", type=int, default=3, help="How many recent threads to check")
    sync_parser.add_argument("--cwd", default=None, help="Filter to a workspace path")
    sync_parser.add_argument("--json", action="store_true", help="Emit JSON")

    hook_parser = subparsers.add_parser("hook", help="Run a claude-mem style hook event against codex-memory")
    _common_storage_args(hook_parser)
    hook_parser.add_argument("event", choices=HOOK_EVENTS, help="Hook event name")
    hook_parser.add_argument("--cwd", default=None, help="Filter to a workspace path")
    hook_parser.add_argument("--limit", type=int, default=3, help="How many recent threads to inspect")
    hook_parser.add_argument("--thread", default="latest", help="Thread id or 'latest' for SessionEnd")
    hook_parser.add_argument("--json", action="store_true", help="Emit JSON")

    watch_parser = subparsers.add_parser("watch", help="Continuously infer and run hook events from Codex transcripts")
    _common_storage_args(watch_parser)
    watch_parser.add_argument("--cwd", default=None, help="Filter to a workspace path")
    watch_parser.add_argument("--limit", type=int, default=3, help="How many recent threads to inspect")
    watch_parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in seconds")
    watch_parser.add_argument("--max-loops", type=int, default=None, help="Stop after N polling loops")
    watch_parser.add_argument("--emit-dir", default=None, help="Persist emitted hook payloads under this directory")
    watch_parser.add_argument("--emit-session-end-on-exit", action="store_true", help="Emit SessionEnd when the watcher exits")
    watch_parser.add_argument("--json", action="store_true", help="Emit JSON lines")

    autostart_parser = subparsers.add_parser("autostart", help="Manage macOS launchd auto-start for watch mode")
    _common_storage_args(autostart_parser)
    autostart_subparsers = autostart_parser.add_subparsers(dest="autostart_command", required=True)

    autostart_install_parser = autostart_subparsers.add_parser("install", help="Install a launchd agent for watch mode")
    autostart_install_parser.add_argument("--cwd", default=None, help="Workspace path to watch")
    autostart_install_parser.add_argument("--limit", type=int, default=3, help="How many recent threads to inspect")
    autostart_install_parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in seconds")
    autostart_install_parser.add_argument("--emit-dir", default=None, help="Directory for emitted hook artifacts")
    autostart_install_parser.add_argument("--launch-agents-dir", default=None, help="Override launch agents directory")
    autostart_install_parser.add_argument("--load", action="store_true", help="Load the agent immediately with launchctl")
    autostart_install_parser.add_argument("--json", action="store_true", help="Emit JSON")

    autostart_status_parser = autostart_subparsers.add_parser("status", help="Show launchd auto-start status")
    autostart_status_parser.add_argument("--cwd", default=None, help="Workspace path to inspect")
    autostart_status_parser.add_argument("--emit-dir", default=None, help="Override emitted hook artifacts directory")
    autostart_status_parser.add_argument("--launch-agents-dir", default=None, help="Override launch agents directory")
    autostart_status_parser.add_argument("--json", action="store_true", help="Emit JSON")

    autostart_remove_parser = autostart_subparsers.add_parser("remove", help="Remove the launchd agent")
    autostart_remove_parser.add_argument("--cwd", default=None, help="Workspace path to remove")
    autostart_remove_parser.add_argument("--launch-agents-dir", default=None, help="Override launch agents directory")
    autostart_remove_parser.add_argument("--unload", action="store_true", help="Unload the agent first with launchctl")
    autostart_remove_parser.add_argument("--json", action="store_true", help="Emit JSON")

    search_parser = subparsers.add_parser("search", help="Search local memory by query")
    _common_storage_args(search_parser)
    search_parser.add_argument("query", nargs="+", help="Search query")
    search_parser.add_argument("--cwd", default=None, help="Filter to a workspace path")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of threads to return")
    search_parser.add_argument("--json", action="store_true", help="Emit JSON")

    context_parser = subparsers.add_parser("context", help="Render a compact memory context block")
    _common_storage_args(context_parser)
    context_parser.add_argument("query", nargs="*", help="Optional topical query; omit to show recent context")
    context_parser.add_argument("--cwd", default=None, help="Filter to a workspace path")
    context_parser.add_argument("--limit", type=int, default=4, help="Number of threads to include")
    context_parser.add_argument("--json", action="store_true", help="Emit JSON")

    brief_parser = subparsers.add_parser("brief", help="Show a detailed brief for one thread")
    _common_storage_args(brief_parser)
    brief_parser.add_argument("thread", nargs="?", default="latest", help="Thread id, id prefix, or 'latest'")
    brief_parser.add_argument("--cwd", default=None, help="Workspace filter when using 'latest'")
    brief_parser.add_argument("--json", action="store_true", help="Emit JSON")

    timeline_parser = subparsers.add_parser("timeline", help="Show structured timeline items for one thread")
    _common_storage_args(timeline_parser)
    timeline_parser.add_argument("thread", nargs="?", default="latest", help="Thread id, id prefix, or 'latest'")
    timeline_parser.add_argument("--cwd", default=None, help="Workspace filter when using 'latest'")
    timeline_parser.add_argument("--json", action="store_true", help="Emit JSON")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    codex_home = default_codex_home(args.codex_home)
    db_path = args.db or default_db_path(codex_home)

    if args.command == "index":
        payload = index_threads(
            codex_home=codex_home,
            db_path=db_path,
            limit=args.limit,
            force=args.force,
            thread_ids=args.thread_ids,
        )
        print(_json_or_text(payload, args.json))
        return 0

    if args.command == "status":
        payload = status(db_path)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("db: {}".format(payload["db_path"]))
            print("threads: {}".format(payload["thread_count"]))
            print("items: {}".format(payload["item_count"]))
            print("fts5: {}".format("enabled" if payload["fts_enabled"] else "disabled"))
            print("vector backend: {}".format(payload["vector_backend"]))
            if payload["embedding_model"]:
                print("embedding model: {}".format(payload["embedding_model"]))
            if payload["embedding_dimensions"]:
                print("embedding dimensions: {}".format(payload["embedding_dimensions"]))
            if payload["vector_backend"] == "qdrant":
                print("vector points: {}".format(payload["vector_points"]))
                print("qdrant path: {}".format(payload["qdrant_path"]))
                if payload.get("vector_lock_error"):
                    print("vector status: locked by another local process")
            print(
                "last indexed: {}".format(
                    format_timestamp(payload["last_indexed_at"]) if payload["last_indexed_at"] else "never"
                )
            )
        return 0

    if args.command == "sync":
        payload = sync_latest_threads(
            codex_home=codex_home,
            db_path=db_path,
            limit=args.limit,
            cwd=args.cwd,
        )
        print(_json_or_text(payload, args.json, render_sync))
        return 0

    if args.command == "hook":
        payload = run_hook_event(
            args.event,
            codex_home=codex_home,
            db_path=db_path,
            cwd=args.cwd,
            limit=args.limit,
            thread_ref=args.thread,
        )
        print(_json_or_text(payload, args.json, lambda item: item["text"]))
        return 0

    if args.command == "watch":
        first = True
        for payload in iter_watch_hooks(
            codex_home=codex_home,
            db_path=db_path,
            cwd=args.cwd,
            limit=args.limit,
                poll_interval=args.poll_interval,
                max_loops=args.max_loops,
                emit_dir=args.emit_dir,
                emit_shutdown_event=args.emit_session_end_on_exit,
            ):
            if args.json:
                print(json.dumps(payload, ensure_ascii=False), flush=True)
            else:
                if not first:
                    print()
                print("== {} ==".format(payload["event"]))
                print(payload["text"])
                sys.stdout.flush()
                first = False
        return 0

    if args.command == "autostart":
        cwd = os.path.abspath(args.cwd or os.getcwd())
        if args.autostart_command == "install":
            payload = install_autostart(
                cwd=cwd,
                repo_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                emit_dir=args.emit_dir,
                launch_agents_dir=args.launch_agents_dir,
                poll_interval=args.poll_interval,
                limit=args.limit,
                codex_home=codex_home,
                db_path=args.db,
                load=args.load,
            )
        elif args.autostart_command == "status":
            payload = autostart_status(
                cwd=cwd,
                launch_agents_dir=args.launch_agents_dir,
                codex_home=codex_home,
                emit_dir=args.emit_dir,
            )
        elif args.autostart_command == "remove":
            payload = remove_autostart(
                cwd=cwd,
                launch_agents_dir=args.launch_agents_dir,
                unload=args.unload,
            )
        else:
            parser.error("Unknown autostart command")
            return 2
        if getattr(args, "json", False):
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("label: {}".format(payload["label"]))
            print("cwd: {}".format(payload["cwd"]))
            print("plist: {}".format(payload["plist_path"]))
            if "emit_dir" in payload:
                print("emit dir: {}".format(payload["emit_dir"]))
            if "installed" in payload:
                print("installed: {}".format("yes" if payload["installed"] else "no"))
            if "loaded" in payload:
                print("loaded: {}".format("yes" if payload["loaded"] else "no"))
            if "removed" in payload:
                print("removed: {}".format("yes" if payload["removed"] else "no"))
            if "unloaded" in payload:
                print("unloaded: {}".format("yes" if payload["unloaded"] else "no"))
        return 0

    ensure_populated(codex_home=codex_home, db_path=db_path)

    if args.command == "search":
        cwd = args.cwd
        query = " ".join(args.query)
        results = search_threads(db_path=db_path, query=query, cwd=cwd, limit=args.limit, codex_home=codex_home)
        print(_json_or_text(results, args.json, render_search_results))
        return 0

    if args.command == "context":
        cwd = args.cwd or os.getcwd()
        query = " ".join(args.query).strip()
        if query:
            results = search_threads(
                db_path=db_path,
                query=query,
                cwd=cwd,
                limit=args.limit,
                codex_home=codex_home,
            )
        else:
            results = recent_threads(db_path=db_path, cwd=cwd, limit=args.limit)
        print(_json_or_text(results, args.json, render_context))
        return 0

    if args.command == "brief":
        brief = resolve_thread(db_path=db_path, thread_ref=args.thread, cwd=args.cwd)
        print(_json_or_text(brief, args.json, render_brief))
        return 0

    if args.command == "timeline":
        brief = resolve_thread(db_path=db_path, thread_ref=args.thread, cwd=args.cwd)
        print(_json_or_text(brief, args.json, render_timeline))
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
