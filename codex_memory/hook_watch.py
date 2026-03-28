import copy
import json
import time
from dataclasses import dataclass, field

from codex_memory.codex_data import discover_threads
from codex_memory.hook_sink import emit_hook_payload
from codex_memory.hook_runtime import run_hook_event


@dataclass
class HookWatchState:
    active_thread_id: str = ""
    rollout_offsets: dict = field(default_factory=dict)


def _cwd_matches(thread_cwd, cwd):
    if not cwd:
        return True
    thread_cwd = thread_cwd or ""
    return thread_cwd == cwd or thread_cwd.startswith(cwd.rstrip("/") + "/")


def _read_jsonl_events(path, offset):
    events = []
    with open(path, "rb") as handle:
        handle.seek(offset)
        for raw_line in handle:
            try:
                events.append(json.loads(raw_line.decode("utf-8")))
            except Exception:
                continue
        new_offset = handle.tell()
    return new_offset, events


def _map_event_to_hook_names(event):
    event_type = event.get("type")
    payload = event.get("payload", {})
    if event_type == "event_msg" and payload.get("type") == "user_message":
        return ["UserPromptSubmit"]
    if event_type == "response_item" and payload.get("type") == "function_call_output":
        return ["PostToolUse"]
    if event_type == "event_msg" and payload.get("type") == "task_complete":
        return ["Stop"]
    return []


def run_watch_iteration(
    state,
    threads,
    hook_runner,
    event_reader,
    cwd=None,
    codex_home=None,
    db_path=None,
    limit=3,
):
    next_state = HookWatchState(
        active_thread_id=state.active_thread_id,
        rollout_offsets=copy.deepcopy(state.rollout_offsets),
    )
    filtered = [thread for thread in threads if _cwd_matches(thread.cwd, cwd)]
    if not filtered:
        return next_state, []
    latest = filtered[0]
    payloads = []

    if not next_state.active_thread_id:
        end_offset, _ = event_reader(latest.rollout_path, 0)
        next_state.active_thread_id = latest.id
        next_state.rollout_offsets[latest.id] = end_offset
        payloads.append(
            hook_runner(
                "SessionStart",
                codex_home=codex_home,
                db_path=db_path,
                cwd=cwd,
                limit=limit,
                thread_ref=latest.id,
            )
        )
        return next_state, payloads

    if next_state.active_thread_id != latest.id:
        payloads.append(
            hook_runner(
                "SessionEnd",
                codex_home=codex_home,
                db_path=db_path,
                cwd=cwd,
                limit=limit,
                thread_ref=next_state.active_thread_id,
            )
        )
        end_offset, _ = event_reader(latest.rollout_path, 0)
        next_state.active_thread_id = latest.id
        next_state.rollout_offsets[latest.id] = end_offset
        payloads.append(
            hook_runner(
                "SessionStart",
                codex_home=codex_home,
                db_path=db_path,
                cwd=cwd,
                limit=limit,
                thread_ref=latest.id,
            )
        )
        return next_state, payloads

    previous_offset = next_state.rollout_offsets.get(latest.id, 0)
    new_offset, events = event_reader(latest.rollout_path, previous_offset)
    next_state.rollout_offsets[latest.id] = new_offset
    for event in events:
        for hook_name in _map_event_to_hook_names(event):
            payloads.append(
                hook_runner(
                    hook_name,
                    codex_home=codex_home,
                    db_path=db_path,
                    cwd=cwd,
                    limit=limit,
                    thread_ref=latest.id,
                )
            )
    return next_state, payloads


def watch_hooks(
    codex_home=None,
    db_path=None,
    cwd=None,
    limit=3,
    poll_interval=2.0,
    max_loops=None,
    emit_dir=None,
    emitter=None,
    emit_shutdown_event=False,
    hook_runner=run_hook_event,
    thread_fetcher=discover_threads,
    event_reader=_read_jsonl_events,
):
    return list(
        iter_watch_hooks(
            codex_home=codex_home,
            db_path=db_path,
            cwd=cwd,
            limit=limit,
            poll_interval=poll_interval,
            max_loops=max_loops,
            emit_dir=emit_dir,
            emitter=emitter,
            emit_shutdown_event=emit_shutdown_event,
            hook_runner=hook_runner,
            thread_fetcher=thread_fetcher,
            event_reader=event_reader,
        )
    )


def iter_watch_hooks(
    codex_home=None,
    db_path=None,
    cwd=None,
    limit=3,
    poll_interval=2.0,
    max_loops=None,
    emit_dir=None,
    emitter=None,
    emit_shutdown_event=False,
    hook_runner=run_hook_event,
    thread_fetcher=discover_threads,
    event_reader=_read_jsonl_events,
):
    state = HookWatchState()
    loops = 0
    try:
        while True:
            threads = thread_fetcher(codex_home=codex_home, limit=max(1, limit))
            state, payloads = run_watch_iteration(
                state=state,
                threads=threads,
                hook_runner=hook_runner,
                event_reader=event_reader,
                cwd=cwd,
                codex_home=codex_home,
                db_path=db_path,
                limit=limit,
            )
            for payload in payloads:
                if emit_dir:
                    emit_hook_payload(payload, emit_dir)
                if emitter:
                    emitter(payload)
                yield payload
            loops += 1
            if max_loops is not None and loops >= max_loops:
                return
            time.sleep(max(0.1, float(poll_interval)))
    finally:
        if emit_shutdown_event and state.active_thread_id:
            payload = hook_runner(
                "SessionEnd",
                codex_home=codex_home,
                db_path=db_path,
                cwd=cwd,
                limit=limit,
                thread_ref=state.active_thread_id,
            )
            if emit_dir:
                emit_hook_payload(payload, emit_dir)
            if emitter:
                emitter(payload)
            yield payload
