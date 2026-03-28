import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from glob import glob
from typing import Dict, Iterable, List, Optional

from codex_memory.vectorizer import top_keywords


ABS_PATH_RE = re.compile(r"(/(?:Users|tmp|var|private|Volumes)[^\s\]\)\}\"']+)")
PATCH_FILE_RE = re.compile(r"\*\*\* (?:Add|Update|Delete) File: (.+)")
IMAGE_TAG_RE = re.compile(r"<image\b.*?</image>", re.IGNORECASE | re.DOTALL)
FILES_SECTION_RE = re.compile(
    r"# Files mentioned by the user:\s*(?:## .+?: .+\n?)+",
    re.IGNORECASE,
)
FILE_MENTION_LINE_RE = re.compile(r"^## .+?: .+$", re.MULTILINE)
MARKDOWN_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
LOW_SIGNAL_MESSAGES = {
    "继续",
    "好",
    "好的",
    "可以",
    "ok",
    "okay",
}
META_FOLLOWUP_HINTS = (
    "对比",
    "做得如何",
    "还需要补充",
    "告诉我结果",
    "评估",
    "最后一个是啥",
    "干啥用的",
    "你自己会用吗",
)
TASK_FOCUS_HINTS = (
    "装",
    "安装",
    "skills",
    "skill",
    "claude-mem",
    "codex",
    "memory",
    "改造",
    "升级",
    "embedding",
    "向量",
    "qdrant",
    "搜索",
    "索引",
)
DECISION_HINTS = (
    "默认",
    "改成",
    "切到",
    "不再",
    "使用",
    "改为",
    "切成",
    "优先",
    "fallback",
)
OUTCOME_HINTS = (
    "已经",
    "完成",
    "装好了",
    "升级",
    "切到",
    "支持",
    "重建",
    "验收",
    "结果",
    "显示",
    "通过",
)
OBSERVATION_HINTS = (
    "vector backend:",
    "embedding model:",
    "embedding dimensions:",
    "vector points:",
    "threads:",
    "items:",
    "fts5:",
    "qdrant path:",
    "last indexed:",
    "indexed':",
    "indexed\":",
    "skipped':",
    "skipped\":",
    "db:",
)
NOISY_OUTPUT_PREFIXES = (
    "Command:",
    "Chunk ID:",
    "Wall time:",
    "Process exited",
    "Process running",
    "Original token count:",
    "Output:",
    "Cloning into ",
    "Plan updated",
    "Approved command prefix saved",
)
META_OUTCOME_HINTS = (
    "对标",
    "还可以继续",
    "后面还可以",
    "比重启前",
    "评分",
    "/10",
)
PLANNING_PREFIXES = (
    "我先",
    "我会",
    "接下来",
    "下一步",
    "准备",
    "现在开始",
    "现在我先",
    "我准备",
    "我去",
)
STRONG_DECISION_HINTS = (
    "provider",
    "embedding",
    "向量",
    "索引",
    "search",
    "context",
)


@dataclass
class ThreadRecord:
    id: str
    rollout_path: str
    created_at: int
    updated_at: int
    source: str
    cwd: str
    title: str
    first_user_message: str
    model: str


@dataclass
class MemoryItem:
    item_type: str
    text: str


@dataclass
class ExtractedThread:
    thread: ThreadRecord
    summary: str
    final_answer: str
    commands: List[str]
    files: List[str]
    tool_names: List[str]
    memory_items: List[MemoryItem]
    content_hash: str


def default_codex_home(explicit=None):
    if explicit:
        return os.path.expanduser(explicit)
    return os.environ.get("CODEX_HOME", os.path.expanduser("~/.codex"))


def find_state_db(codex_home):
    candidates = sorted(glob(os.path.join(codex_home, "state_*.sqlite")))
    if not candidates:
        raise FileNotFoundError("No state_*.sqlite database found under {}".format(codex_home))
    return max(candidates, key=os.path.getmtime)


def find_paths(text):
    seen = set()
    paths = []
    for match in ABS_PATH_RE.findall(text or ""):
        cleaned = match.rstrip(".,:;`")
        if cleaned not in seen:
            seen.add(cleaned)
            paths.append(cleaned)
    return paths


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def clean_message_text(text):
    text = text or ""
    text = IMAGE_TAG_RE.sub(" ", text)
    text = FILES_SECTION_RE.sub(" ", text)
    text = FILE_MENTION_LINE_RE.sub(" ", text)
    text = re.sub(
        r"(?im)^\s{0,3}#{0,6}\s*My request for Codex:\s*",
        " ",
        text,
    )
    text = MARKDOWN_HEADING_RE.sub("", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return normalize_whitespace(text)


def shorten(text, limit=220):
    text = normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def summarize_outcome_text(text, limit=180):
    text = normalize_whitespace(text)
    if not text:
        return ""
    sentence_match = re.match(r"(.{1,%d}?[。！？!?\.])(?:\s|$)" % limit, text)
    if sentence_match:
        return sentence_match.group(1).strip()
    trimmed = re.split(r"\s(?:\*\*|[-*]\s|\d+\.\s)", text, maxsplit=1)[0]
    return shorten(trimmed, limit=limit)


def split_sentences(text):
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    pieces = re.split(r"(?<=[。！？!?\.])\s+|\s*[•·]\s+|\s+-\s+", normalized)
    return [normalize_whitespace(piece) for piece in pieces if normalize_whitespace(piece)]


def _score_outcome_message(message):
    lowered = (message or "").lower()
    score = 0
    if any(hint in message for hint in OUTCOME_HINTS):
        score += 2
    if any(hint in lowered for hint in TASK_FOCUS_HINTS):
        score += 3
    if any(hint in message for hint in META_OUTCOME_HINTS):
        score -= 4
    return score


def _extract_command_line(output):
    for raw_line in (output or "").splitlines():
        line = raw_line.strip()
        if line.startswith("Command:"):
            return normalize_whitespace(line[len("Command:") :])
    return ""


def _command_is_relevant_for_observation(command):
    lowered = (command or "").lower()
    return any(
        token in lowered
        for token in (
            "./codex-memory",
            "codex-memory ",
            "sqlite3 ",
            "qdrant",
            "state_",
        )
    )


def compact_path(path):
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path


def is_noisy_path(path):
    compact = compact_path(path)
    lowered = compact.lower()
    if not compact or compact in ("/tmp", "~/.codex", "~/.codex/skills", "~/.codex/sessions"):
        return True
    if "/rwtemp/" in lowered or "xwechat_files" in lowered:
        return True
    if lowered.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")) and ("/temp/" in lowered or "/tmp/" in lowered):
        return True
    if lowered.startswith("/tmp/") or compact.startswith("~/Library/Containers/com.tencent.xinWeChat/"):
        return True
    if "/.codex/sessions/" in lowered:
        return True
    return False


def prioritize_files(paths, limit=8):
    scored = []
    seen = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        score = 0
        lowered = path.lower()
        if is_noisy_path(path):
            score -= 10
        if "/desktop/dev/" in lowered:
            score += 8
        if "/.codex/skills/" in lowered:
            score += 5
        if lowered.endswith(("skill.md", ".py", ".md", ".toml", ".jsonl", ".sqlite")):
            score += 3
        if "/tmp/" in lowered:
            score -= 4
        scored.append((score, path))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [path for score, path in scored if score > -5][:limit]


def prioritize_commands(commands, limit=5):
    scored = []
    seen = set()
    for command in commands:
        if not command or command in seen:
            continue
        seen.add(command)
        score = 0
        lowered = command.lower()
        if lowered.startswith(("ls ", "sed ", "find ", "cat ", "pwd", "which ", "sqlite3 ")):
            score -= 3
        if any(keyword in lowered for keyword in ("pip install", "git clone", "./codex-memory", "python3 -m venv", "cp -r", "install-skill")):
            score += 5
        if "index --force" in lowered or "search " in lowered or "context " in lowered:
            score += 4
        if "status" in lowered:
            score += 1
        scored.append((score, command))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [command for score, command in scored if score >= -1][:limit]


def choose_primary_message(messages):
    cleaned = []
    for message in messages:
        normalized = clean_message_text(message)
        if normalized:
            cleaned.append(normalized)
    if not cleaned:
        return ""
    for message in cleaned:
        if message.lower() not in LOW_SIGNAL_MESSAGES and len(message) >= 6:
            return message
    return cleaned[0]


def choose_latest_focus(messages):
    cleaned = []
    for message in messages:
        normalized = clean_message_text(message)
        if normalized:
            cleaned.append(normalized)
    if not cleaned:
        return ""
    fallback = cleaned[-1]
    for message in reversed(cleaned):
        lowered = message.lower()
        if lowered in LOW_SIGNAL_MESSAGES or len(message) < 4:
            continue
        score = 1 if len(message) >= 8 else 0
        if any(hint in lowered for hint in TASK_FOCUS_HINTS):
            score += 3
        if any(hint in message for hint in META_FOLLOWUP_HINTS):
            score -= 4
        if score >= 2:
            return message
        if fallback == cleaned[-1]:
            fallback = message
    return fallback


def choose_outcome(final_messages, commentary_messages):
    candidates = []
    for index, message in enumerate(reversed(final_messages or [])):
        candidates.append((20 - index, message))
    for index, message in enumerate(reversed(commentary_messages or [])):
        candidates.append((10 - index, message))
    if not candidates:
        return ""
    best_score = None
    best_message = ""
    for recency_bonus, message in candidates:
        score = _score_outcome_message(message) + recency_bonus
        if best_score is None or score > best_score:
            best_score = score
            best_message = message
    return summarize_outcome_text(best_message)


def extract_decisions(messages, limit=4):
    decisions = []
    for message in reversed(messages):
        for sentence in reversed(split_sentences(message)):
            lowered = sentence.lower()
            if any(hint in lowered for hint in DECISION_HINTS):
                if any(sentence.startswith(prefix) for prefix in PLANNING_PREFIXES) and not any(
                    hint in lowered for hint in STRONG_DECISION_HINTS
                ):
                    continue
                candidate = shorten(sentence, limit=220)
                if candidate not in decisions:
                    decisions.append(candidate)
            if len(decisions) >= limit:
                return decisions
    return decisions


def extract_observations(outputs, limit=4):
    observations = []
    for payload in outputs:
        command = _extract_command_line(payload)
        if command and not _command_is_relevant_for_observation(command):
            continue
        lines = []
        for raw_line in (payload or "").splitlines():
            line = normalize_whitespace(raw_line)
            if not line:
                continue
            if any(line.startswith(prefix) for prefix in NOISY_OUTPUT_PREFIXES):
                continue
            lowered = line.lower()
            if any(hint in lowered for hint in OBSERVATION_HINTS):
                lines.append(line)
        if lines:
            candidate = shorten("; ".join(lines[:4]), limit=240)
            if candidate not in observations:
                observations.append(candidate)
        if len(observations) >= limit:
            break
    return observations


def _extract_output_text(content):
    parts = []
    for item in content or []:
        if item.get("type") == "output_text" and item.get("text"):
            parts.append(item["text"])
    return "\n".join(parts).strip()


def _parse_json_object(payload):
    try:
        return json.loads(payload)
    except Exception:
        return {}


def _dedupe_keep_order(values):
    seen = set()
    ordered = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def discover_threads(codex_home, limit=None, thread_ids=None):
    state_db = find_state_db(codex_home)
    connection = sqlite3.connect(state_db)
    connection.row_factory = sqlite3.Row
    try:
        sql = [
            "SELECT id, rollout_path, created_at, updated_at, source, cwd, title,",
            "       first_user_message, COALESCE(model, '') AS model",
            "FROM threads",
        ]
        params = []
        if thread_ids:
            placeholders = ",".join("?" for _ in thread_ids)
            sql.append("WHERE id IN ({})".format(placeholders))
            params.extend(thread_ids)
        sql.append("ORDER BY updated_at DESC")
        if limit:
            sql.append("LIMIT ?")
            params.append(limit)
        rows = connection.execute("\n".join(sql), params).fetchall()
    finally:
        connection.close()
    return [
        ThreadRecord(
            id=row["id"],
            rollout_path=row["rollout_path"],
            created_at=int(row["created_at"]),
            updated_at=int(row["updated_at"]),
            source=row["source"],
            cwd=row["cwd"],
            title=row["title"],
            first_user_message=row["first_user_message"] or "",
            model=row["model"] or "",
        )
        for row in rows
    ]


def _iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                yield json.loads(raw_line)
            except json.JSONDecodeError:
                continue


def _build_summary(thread, user_messages, final_messages, commentary_messages, commands, files, tool_names):
    parts = []
    title = normalize_whitespace(thread.title) or shorten(thread.first_user_message, 120) or thread.id
    parts.append("Thread: {}.".format(title))
    if thread.cwd:
        parts.append("Workspace: {}.".format(compact_path(thread.cwd)))
    first_user = choose_primary_message(user_messages) if user_messages else clean_message_text(thread.first_user_message)
    latest_focus = choose_latest_focus(user_messages)
    if first_user:
        parts.append("User asked: {}.".format(shorten(first_user, 220)))
    if latest_focus and latest_focus != first_user:
        parts.append("Later focus: {}.".format(shorten(latest_focus, 180)))
    outcome = choose_outcome(final_messages, commentary_messages)
    if outcome:
        parts.append("Outcome: {}.".format(shorten(outcome, 260)))
    useful_commands = prioritize_commands(commands, limit=4)
    if useful_commands:
        pretty_commands = [shorten(command, 84) for command in useful_commands]
        parts.append("Commands: {}.".format("; ".join(pretty_commands)))
    useful_files = prioritize_files(files, limit=6)
    if useful_files:
        pretty_files = [compact_path(path) for path in useful_files]
        parts.append("Files: {}.".format(", ".join(pretty_files)))
    if tool_names:
        parts.append("Tools: {}.".format(", ".join(tool_names[:8])))
    return " ".join(parts)


def _chunk_text(text, limit=900):
    normalized = (text or "").strip()
    if not normalized:
        return []
    if len(normalized) <= limit:
        return [normalized]
    chunks = []
    cursor = 0
    while cursor < len(normalized):
        end = min(len(normalized), cursor + limit)
        if end < len(normalized):
            split = normalized.rfind("\n", cursor, end)
            if split <= cursor:
                split = normalized.rfind(" ", cursor, end)
            if split > cursor:
                end = split
        chunks.append(normalized[cursor:end].strip())
        cursor = end
    return [chunk for chunk in chunks if chunk]


def extract_thread(thread):
    user_messages = []
    commentary_messages = []
    final_messages = []
    function_outputs = []
    commands = []
    tool_names = []
    files = list(find_paths(thread.first_user_message))

    for event in _iter_jsonl(thread.rollout_path):
        event_type = event.get("type")
        payload = event.get("payload", {})
        if event_type == "event_msg" and payload.get("type") == "user_message":
            message = payload.get("message", "")
            normalized = clean_message_text(message)
            if normalized:
                user_messages.append(normalized)
            files.extend(find_paths(message))
            for image_path in payload.get("local_images", []) or []:
                files.append(image_path)
        elif event_type == "response_item":
            item_type = payload.get("type")
            if item_type == "message" and payload.get("role") == "assistant":
                text = _extract_output_text(payload.get("content"))
                normalized = normalize_whitespace(text)
                if not normalized:
                    continue
                if payload.get("phase") in ("final", "final_answer"):
                    final_messages.append(normalized)
                else:
                    commentary_messages.append(normalized)
                files.extend(find_paths(text))
            elif item_type == "function_call_output":
                output = payload.get("output", "")
                normalized = normalize_whitespace(output)
                if normalized:
                    function_outputs.append(output)
                    files.extend(find_paths(output))
            elif item_type == "function_call":
                tool_name = payload.get("name", "")
                if tool_name:
                    tool_names.append(tool_name)
                arguments = _parse_json_object(payload.get("arguments", "{}"))
                if tool_name == "exec_command":
                    command = normalize_whitespace(arguments.get("cmd", ""))
                    if command:
                        commands.append(command)
                        files.extend(find_paths(command))
                elif tool_name == "apply_patch":
                    patch_text = payload.get("arguments", "")
                    for match in PATCH_FILE_RE.findall(patch_text):
                        files.append(match)
                else:
                    files.extend(find_paths(payload.get("arguments", "")))

    user_messages = _dedupe_keep_order(user_messages)
    commentary_messages = _dedupe_keep_order(commentary_messages)
    final_messages = _dedupe_keep_order(final_messages)
    commands = _dedupe_keep_order(commands)
    files = _dedupe_keep_order(files)
    tool_names = _dedupe_keep_order(tool_names)
    files = prioritize_files(files, limit=24)

    summary = _build_summary(
        thread=thread,
        user_messages=user_messages,
        final_messages=final_messages,
        commentary_messages=commentary_messages,
        commands=commands,
        files=files,
        tool_names=tool_names,
    )
    final_answer = final_messages[-1] if final_messages else ""
    result_text = choose_outcome(final_messages, commentary_messages)
    decisions = extract_decisions(commentary_messages + ([final_answer] if final_answer else []))
    observations = extract_observations(function_outputs)

    memory_items = [MemoryItem(item_type="summary", text=summary)]
    if thread.title:
        memory_items.append(MemoryItem(item_type="title", text=normalize_whitespace(thread.title)))
    if user_messages:
        selected_users = []
        for message in user_messages[:2]:
            selected_users.append(message)
        for message in user_messages[-6:]:
            if message not in selected_users:
                selected_users.append(message)
        for message in selected_users:
            for chunk in _chunk_text(message, limit=700):
                memory_items.append(MemoryItem(item_type="user", text=chunk))
    elif thread.first_user_message:
        cleaned_first = clean_message_text(thread.first_user_message)
        if cleaned_first:
            memory_items.append(MemoryItem(item_type="user", text=cleaned_first))

    assistant_sources = []
    for message in commentary_messages[-6:]:
        if message not in assistant_sources:
            assistant_sources.append(message)
    for source in assistant_sources:
        for chunk in _chunk_text(source, limit=900):
            memory_items.append(MemoryItem(item_type="assistant", text=chunk))
    if final_answer:
        for chunk in _chunk_text(final_answer, limit=900):
            memory_items.append(MemoryItem(item_type="answer", text=chunk))
    if result_text:
        memory_items.append(MemoryItem(item_type="result", text=result_text))
    for text in decisions:
        memory_items.append(MemoryItem(item_type="decision", text=text))
    for text in observations:
        memory_items.append(MemoryItem(item_type="observation", text=text))

    useful_commands = prioritize_commands(commands, limit=10)
    if useful_commands:
        command_block = "\n".join("$ " + command for command in useful_commands)
        memory_items.append(MemoryItem(item_type="commands", text=command_block))
    if files:
        memory_items.append(
            MemoryItem(
                item_type="files",
                text="\n".join(compact_path(path) for path in files[:24]),
            )
        )
    if tool_names:
        memory_items.append(MemoryItem(item_type="tools", text=", ".join(tool_names)))
    if summary:
        memory_items.append(
            MemoryItem(
                item_type="keywords",
                text=", ".join(top_keywords(summary + " " + " ".join(user_messages), limit=18)),
            )
        )

    canonical = {
        "summary": summary,
        "final_answer": final_answer,
        "commands": commands,
        "files": files,
        "tool_names": tool_names,
        "items": [(item.item_type, item.text) for item in memory_items],
    }
    content_hash = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return ExtractedThread(
        thread=thread,
        summary=summary,
        final_answer=final_answer,
        commands=commands,
        files=files,
        tool_names=tool_names,
        memory_items=memory_items,
        content_hash=content_hash,
    )
