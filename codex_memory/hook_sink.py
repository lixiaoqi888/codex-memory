import json
import os
import time


def emit_hook_payload(payload, emit_dir):
    os.makedirs(emit_dir, exist_ok=True)
    latest_dir = os.path.join(emit_dir, "latest")
    os.makedirs(latest_dir, exist_ok=True)

    event_name = payload["event"]
    record = dict(payload)
    record["emitted_at"] = int(time.time())

    events_log = os.path.join(emit_dir, "events.jsonl")
    latest_json = os.path.join(latest_dir, "{}.json".format(event_name))
    latest_text = os.path.join(latest_dir, "{}.txt".format(event_name))
    latest_event_json = os.path.join(emit_dir, "latest.json")
    latest_event_text = os.path.join(emit_dir, "latest.txt")

    with open(events_log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(latest_json, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)

    with open(latest_text, "w", encoding="utf-8") as handle:
        handle.write(payload.get("text", ""))
        handle.write("\n")

    with open(latest_event_json, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)

    with open(latest_event_text, "w", encoding="utf-8") as handle:
        handle.write(payload.get("text", ""))
        handle.write("\n")

    return {
        "events_log": events_log,
        "latest_json": latest_json,
        "latest_text": latest_text,
        "latest_event_json": latest_event_json,
        "latest_event_text": latest_event_text,
    }
