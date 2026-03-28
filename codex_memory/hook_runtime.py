from codex_memory.indexer import (
    recent_threads,
    render_context,
    render_sync,
    render_timeline,
    resolve_thread,
    sync_latest_threads,
)


HOOK_EVENTS = (
    "SessionStart",
    "UserPromptSubmit",
    "PostToolUse",
    "Stop",
    "SessionEnd",
)


def _validate_event(event):
    if event not in HOOK_EVENTS:
        raise ValueError("Unknown hook event: {}".format(event))


def run_hook_event(event, codex_home=None, db_path=None, cwd=None, limit=3, thread_ref="latest"):
    _validate_event(event)
    sync_payload = sync_latest_threads(
        codex_home=codex_home,
        db_path=db_path,
        limit=limit,
        cwd=cwd,
    )

    if event == "SessionStart":
        results = recent_threads(db_path=db_path, cwd=cwd, limit=limit)
        text = render_context(results)
        return {
            "event": event,
            "mode": "inject",
            "sync": sync_payload,
            "text": text,
        }

    if event == "SessionEnd":
        brief = resolve_thread(db_path=db_path, thread_ref=thread_ref, cwd=cwd)
        text = render_timeline(brief)
        return {
            "event": event,
            "mode": "sync+timeline",
            "sync": sync_payload,
            "text": text,
        }

    return {
        "event": event,
        "mode": "sync",
        "sync": sync_payload,
        "text": render_sync(sync_payload),
    }
