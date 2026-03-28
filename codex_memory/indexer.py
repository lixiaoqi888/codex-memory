import json
import os
import sqlite3
import time
from datetime import datetime

from codex_memory.codex_data import (
    clean_message_text,
    compact_path,
    default_codex_home,
    discover_threads,
    extract_thread,
    prioritize_commands,
    prioritize_files,
    shorten,
    summarize_outcome_text,
)
from codex_memory.embedding_provider import EmbeddingError, embed_texts, resolve_embedding_settings
from codex_memory.vector_store import (
    COLLECTION_NAME,
    collection_exists,
    collection_point_count,
    default_qdrant_path,
    delete_points,
    open_qdrant,
    query_vectors,
    recreate_collection,
    upsert_points,
)
from codex_memory.vectorizer import cosine_similarity, decode_vector, encode_vector, text_to_vector, token_overlap


ITEM_TYPE_SCORE_MULTIPLIER = {
    "result": 1.18,
    "decision": 1.12,
    "answer": 1.14,
    "assistant": 1.08,
    "user": 1.04,
    "observation": 1.02,
    "commands": 0.68,
    "files": 0.94,
    "keywords": 0.88,
    "summary": 0.78,
    "title": 0.66,
    "tools": 0.5,
}
ITEM_TYPE_MATCH_PRIORITY = {
    "result": 9,
    "decision": 8,
    "answer": 8,
    "assistant": 7,
    "user": 6,
    "observation": 6,
    "commands": 5,
    "files": 4,
    "keywords": 3,
    "summary": 2,
    "title": 1,
    "tools": 0,
}
SIGNAL_ITEM_TYPES = ("result", "decision", "observation")


def default_db_path(codex_home=None):
    override = os.environ.get("CODEX_MEMORY_DB")
    if override:
        return os.path.expanduser(override)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, ".data", "codex-memory.sqlite")


def open_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    ensure_schema(connection)
    return connection


def open_db_readonly(db_path):
    if not os.path.exists(db_path):
        return open_db(db_path)
    connection = sqlite3.connect(
        "file:{}?mode=ro".format(os.path.abspath(db_path)),
        uri=True,
        timeout=30,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def ensure_schema(connection):
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            cwd TEXT NOT NULL,
            source TEXT NOT NULL,
            model TEXT NOT NULL,
            rollout_path TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            indexed_at INTEGER NOT NULL,
            first_user_message TEXT NOT NULL,
            summary TEXT NOT NULL,
            final_answer TEXT NOT NULL,
            commands_json TEXT NOT NULL,
            files_json TEXT NOT NULL,
            tool_names_json TEXT NOT NULL,
            content_hash TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_threads_updated_at ON threads(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_threads_cwd ON threads(cwd);

        CREATE TABLE IF NOT EXISTS memory_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            item_type TEXT NOT NULL,
            text TEXT NOT NULL,
            preview TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_memory_items_thread_id ON memory_items(thread_id);
        CREATE INDEX IF NOT EXISTS idx_memory_items_item_type ON memory_items(item_type);
        """
    )
    try:
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
            USING fts5(text, thread_id UNINDEXED, item_type UNINDEXED, tokenize='unicode61')
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('fts_enabled', '1')"
        )
    except sqlite3.OperationalError:
        connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('fts_enabled', '0')"
        )
    connection.commit()


def _meta_get(connection, key, default=None):
    row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if not row:
        return default
    return row["value"]


def _meta_set(connection, key, value):
    connection.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
        (key, "" if value is None else str(value)),
    )


def _fts_enabled(connection):
    return _meta_get(connection, "fts_enabled", "0") == "1"


def _vector_settings_signature(settings):
    return json.dumps(
        {
            "provider": settings.provider_name,
            "base_url": settings.base_url,
            "model": settings.model,
            "dimensions": settings.dimensions,
            "local_model": settings.local_model,
        },
        sort_keys=True,
    )


def _prepare_vector_backend(connection, db_path, codex_home, force):
    settings = resolve_embedding_settings(codex_home=codex_home)
    qdrant_path = default_qdrant_path(db_path)
    signature = _vector_settings_signature(settings)
    stored_signature = _meta_get(connection, "embedding_signature")
    try:
        client = open_qdrant(qdrant_path)
    except RuntimeError as exc:
        return {
            "client": None,
            "settings": settings,
            "qdrant_path": qdrant_path,
            "signature": signature,
            "collection_ready": False,
            "rebuild": False,
            "active": False,
            "error": str(exc),
        }
    needs_rebuild = force or stored_signature != signature or not collection_exists(client, COLLECTION_NAME)
    if needs_rebuild and collection_exists(client, COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    return {
        "client": client,
        "settings": settings,
        "qdrant_path": qdrant_path,
        "signature": signature,
        "collection_ready": False,
        "rebuild": needs_rebuild,
        "active": True,
        "error": None,
    }


def _store_backend_meta(connection, backend, embedding_dimensions):
    settings = backend["settings"]
    _meta_set(connection, "vector_backend", "qdrant")
    _meta_set(connection, "vector_collection", COLLECTION_NAME)
    _meta_set(connection, "vector_qdrant_path", backend["qdrant_path"])
    _meta_set(connection, "embedding_provider", settings.effective_provider or settings.provider_name)
    _meta_set(connection, "embedding_model", settings.effective_model or settings.model)
    _meta_set(connection, "embedding_dimensions", embedding_dimensions)
    _meta_set(connection, "embedding_signature", backend["signature"])
    _meta_set(connection, "embedding_base_url", settings.effective_base_url or settings.base_url)


def _update_vector_point_count_meta(connection, backend):
    if not backend or not backend.get("active") or not backend.get("client"):
        return
    try:
        point_count = collection_point_count(backend["client"], COLLECTION_NAME)
    except Exception:
        return
    _meta_set(connection, "vector_point_count", int(point_count))


def _clear_thread_items(connection, backend, thread_id):
    item_rows = connection.execute(
        "SELECT id FROM memory_items WHERE thread_id = ?", (thread_id,)
    ).fetchall()
    point_ids = [row["id"] for row in item_rows]
    if backend and backend.get("active"):
        delete_points(backend["client"], point_ids, collection_name=COLLECTION_NAME)
    if _fts_enabled(connection):
        for point_id in point_ids:
            connection.execute("DELETE FROM memory_fts WHERE rowid = ?", (point_id,))
    connection.execute("DELETE FROM memory_items WHERE thread_id = ?", (thread_id,))


def _insert_thread(connection, backend, extracted):
    now_ts = int(time.time())
    connection.execute(
        """
        INSERT OR REPLACE INTO threads (
            id, title, cwd, source, model, rollout_path, created_at, updated_at,
            indexed_at, first_user_message, summary, final_answer,
            commands_json, files_json, tool_names_json, content_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            extracted.thread.id,
            extracted.thread.title or "",
            extracted.thread.cwd or "",
            extracted.thread.source or "",
            extracted.thread.model or "",
            extracted.thread.rollout_path,
            int(extracted.thread.created_at),
            int(extracted.thread.updated_at),
            now_ts,
            extracted.thread.first_user_message or "",
            extracted.summary or "",
            extracted.final_answer or "",
            json.dumps(extracted.commands, ensure_ascii=False),
            json.dumps(extracted.files, ensure_ascii=False),
            json.dumps(extracted.tool_names, ensure_ascii=False),
            extracted.content_hash,
        ),
    )
    _clear_thread_items(connection, backend, extracted.thread.id)

    inserted_ids = []
    inserted_texts = []
    inserted_payloads = []
    for ordinal, item in enumerate(extracted.memory_items):
        preview = shorten(item.text, limit=180)
        vector_payload = encode_vector(text_to_vector(item.text))
        cursor = connection.execute(
            """
            INSERT INTO memory_items (
                thread_id, ordinal, item_type, text, preview, vector_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                extracted.thread.id,
                ordinal,
                item.item_type,
                item.text,
                preview,
                vector_payload,
                int(extracted.thread.updated_at),
            ),
        )
        row_id = int(cursor.lastrowid)
        inserted_ids.append(row_id)
        inserted_texts.append(item.text)
        inserted_payloads.append(
            {
                "thread_id": extracted.thread.id,
                "item_type": item.item_type,
                "cwd": extracted.thread.cwd or "",
                "preview": preview,
                "title": extracted.thread.title or "",
            }
        )
        if _fts_enabled(connection):
            connection.execute(
                "INSERT INTO memory_fts(rowid, text, thread_id, item_type) VALUES (?, ?, ?, ?)",
                (row_id, item.text, extracted.thread.id, item.item_type),
            )
    if backend and backend.get("active") and inserted_texts:
        try:
            vectors = embed_texts(inserted_texts, backend["settings"])
        except EmbeddingError as exc:
            backend["active"] = False
            backend["error"] = str(exc)
            return
        if not backend["collection_ready"]:
            recreate_collection(
                backend["client"],
                vector_size=len(vectors[0]),
                collection_name=COLLECTION_NAME,
            )
            _store_backend_meta(connection, backend, embedding_dimensions=len(vectors[0]))
            backend["collection_ready"] = True
        upsert_points(
            backend["client"],
            ids=inserted_ids,
            vectors=vectors,
            payloads=inserted_payloads,
            collection_name=COLLECTION_NAME,
        )


def index_threads(codex_home=None, db_path=None, limit=None, force=False, thread_ids=None):
    codex_home = default_codex_home(codex_home)
    db_path = db_path or default_db_path(codex_home)
    rows = discover_threads(codex_home=codex_home, limit=limit, thread_ids=thread_ids)
    connection = open_db(db_path)
    indexed = 0
    skipped = 0
    backend = None
    try:
        backend = _prepare_vector_backend(connection, db_path, codex_home, force)
        if backend["rebuild"]:
            force = True
        for thread in rows:
            existing = connection.execute(
                "SELECT updated_at, content_hash FROM threads WHERE id = ?", (thread.id,)
            ).fetchone()
            if existing and not force and int(existing["updated_at"]) == int(thread.updated_at):
                skipped += 1
                continue
            extracted = extract_thread(thread)
            if (
                existing
                and not force
                and int(existing["updated_at"]) == int(thread.updated_at)
                and existing["content_hash"] == extracted.content_hash
            ):
                skipped += 1
                continue
            _insert_thread(connection, backend, extracted)
            indexed += 1
        _meta_set(connection, "last_indexed_at", int(time.time()))
        if backend and backend.get("active"):
            _meta_set(connection, "vector_backend", "qdrant")
            _meta_set(connection, "vector_collection", COLLECTION_NAME)
            _meta_set(connection, "vector_qdrant_path", backend["qdrant_path"])
            _meta_set(
                connection,
                "embedding_provider",
                backend["settings"].effective_provider or backend["settings"].provider_name,
            )
            _meta_set(
                connection,
                "embedding_model",
                backend["settings"].effective_model or backend["settings"].model,
            )
            _meta_set(connection, "embedding_signature", backend["signature"])
            _update_vector_point_count_meta(connection, backend)
        elif backend and backend.get("error"):
            _meta_set(connection, "vector_backend", "qdrant")
            _meta_set(connection, "vector_qdrant_path", backend["qdrant_path"])
        connection.commit()
    finally:
        connection.close()
        if backend and backend.get("client"):
            backend["client"].close()
    return {
        "db_path": db_path,
        "thread_count": len(rows),
        "indexed": indexed,
        "skipped": skipped,
    }


def _make_fts_scores(connection, query, cwd, cap):
    if not _fts_enabled(connection):
        return {}
    tokens = [token for token in query.replace('"', " ").split() if token.strip()]
    if not tokens:
        return {}
    fts_query = " OR ".join('"{}"'.format(token.replace('"', "")) for token in tokens[:10])
    try:
        rows = connection.execute(
            """
            SELECT memory_fts.rowid AS row_id
            FROM memory_fts
            JOIN memory_items ON memory_items.id = memory_fts.rowid
            JOIN threads ON threads.id = memory_items.thread_id
            WHERE memory_fts MATCH ?
              AND (? = '' OR threads.cwd = ? OR threads.cwd LIKE ?)
            ORDER BY bm25(memory_fts)
            LIMIT ?
            """,
            (
                fts_query,
                cwd or "",
                cwd or "",
                (cwd.rstrip("/") + "/%") if cwd else "",
                cap,
            ),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    scores = {}
    total = max(1, len(rows))
    for index, row in enumerate(rows):
        scores[int(row["row_id"])] = 1.0 - (index / float(total))
    return scores


def _make_qdrant_scores(connection, db_path, codex_home, query, limit):
    backend_name = _meta_get(connection, "vector_backend")
    if backend_name != "qdrant":
        return {}
    qdrant_path = _meta_get(connection, "vector_qdrant_path", default_qdrant_path(db_path))
    embed_model = _meta_get(connection, "embedding_model")
    embed_dimensions = _meta_get(connection, "embedding_dimensions")
    embed_provider = _meta_get(connection, "embedding_provider")
    settings = resolve_embedding_settings(
        codex_home=codex_home,
        prefer_model=embed_model,
        prefer_provider=embed_provider,
    )
    settings.dimensions = int(embed_dimensions) if embed_dimensions else settings.dimensions
    try:
        query_vector = embed_texts([query], settings, batch_size=1)[0]
    except EmbeddingError:
        return {}
    try:
        client = open_qdrant(qdrant_path)
    except RuntimeError:
        return {}
    try:
        points = query_vectors(
            client,
            vector=query_vector,
            limit=max(limit * 12, 48),
            collection_name=COLLECTION_NAME,
        )
    finally:
        client.close()
    return dict((int(point.id), float(point.score)) for point in points)


def _iter_candidates(connection, cwd):
    return connection.execute(
        """
        SELECT
            memory_items.id,
            memory_items.thread_id,
            memory_items.item_type,
            memory_items.text,
            memory_items.preview,
            memory_items.vector_json,
            threads.title,
            threads.cwd,
            threads.summary,
            threads.created_at,
            threads.updated_at,
            threads.first_user_message,
            threads.files_json,
            threads.commands_json,
            threads.final_answer
        FROM memory_items
        JOIN threads ON threads.id = memory_items.thread_id
        WHERE (? = '' OR threads.cwd = ? OR threads.cwd LIKE ?)
        ORDER BY threads.updated_at DESC, memory_items.ordinal ASC
        """,
        (
            cwd or "",
            cwd or "",
            (cwd.rstrip("/") + "/%") if cwd else "",
        ),
    )


def _recency_score(updated_at):
    age_days = max(0.0, (time.time() - float(updated_at)) / 86400.0)
    return 1.0 / (1.0 + (age_days / 14.0))


def _cwd_matches(thread_cwd, cwd):
    if not cwd:
        return True
    thread_cwd = thread_cwd or ""
    return thread_cwd == cwd or thread_cwd.startswith(cwd.rstrip("/") + "/")


def _item_type_multiplier(item_type):
    return ITEM_TYPE_SCORE_MULTIPLIER.get(item_type, 1.0)


def _item_type_priority(item_type):
    return ITEM_TYPE_MATCH_PRIORITY.get(item_type, 0)


def _query_prefers_commands(query):
    lowered = (query or "").lower()
    return any(
        token in lowered
        for token in (
            "./",
            "--",
            "git ",
            "pip ",
            "python",
            "npm ",
            "sqlite3 ",
            "rg ",
            "sed ",
            "find ",
        )
    )


def _load_thread_signals(connection, thread_id):
    placeholders = ",".join("?" for _ in SIGNAL_ITEM_TYPES)
    rows = connection.execute(
        """
        SELECT item_type, preview
        FROM memory_items
        WHERE thread_id = ?
          AND item_type IN ({})
        ORDER BY
          CASE item_type
            WHEN 'result' THEN 0
            WHEN 'decision' THEN 1
            WHEN 'observation' THEN 2
            ELSE 9
          END,
          ordinal DESC
        """.format(placeholders),
        (thread_id, *SIGNAL_ITEM_TYPES),
    ).fetchall()
    signals = {}
    for row in rows:
        item_type = row["item_type"]
        if item_type not in signals:
            signals[item_type] = row["preview"]
    return signals


def _finalize_matches(matches, limit=3):
    ordered = []
    seen = set()
    for match in sorted(
        matches,
        key=lambda item: (
            item["score"],
            _item_type_priority(item["item_type"]),
            len(item["preview"]),
        ),
        reverse=True,
    ):
        key = (match["item_type"], match["preview"])
        if key in seen:
            continue
        seen.add(key)
        ordered.append(
            {
                "item_type": match["item_type"],
                "preview": match["preview"],
                "score": round(match["score"], 3),
            }
        )
        if len(ordered) >= limit:
            break
    return ordered


def search_threads(db_path, query, cwd=None, limit=5, codex_home=None):
    codex_home = default_codex_home(codex_home)
    connection = open_db_readonly(db_path)
    query = (query or "").strip()
    if not query:
        return recent_threads(db_path=db_path, cwd=cwd, limit=limit)
    fallback_vector = text_to_vector(query)
    command_intent = _query_prefers_commands(query)
    qdrant_scores = _make_qdrant_scores(connection, db_path, codex_home, query, limit)
    fts_scores = _make_fts_scores(connection, query, cwd, cap=max(25, limit * 8))
    grouped = {}
    fallbacks = []
    try:
        for row in _iter_candidates(connection, cwd):
            candidate_vector = decode_vector(row["vector_json"])
            fallback_semantic = cosine_similarity(fallback_vector, candidate_vector)
            semantic_score = qdrant_scores.get(int(row["id"]), 0.0)
            lexical_score = fts_scores.get(int(row["id"]), 0.0)
            overlap_score = token_overlap(query, row["text"])
            recency = _recency_score(row["updated_at"])
            if qdrant_scores:
                score = (
                    (0.66 * semantic_score)
                    + (0.16 * lexical_score)
                    + (0.08 * overlap_score)
                    + (0.06 * fallback_semantic)
                    + (0.04 * recency)
                )
            else:
                score = (
                    (0.58 * fallback_semantic)
                    + (0.24 * lexical_score)
                    + (0.12 * overlap_score)
                    + (0.06 * recency)
                )
            score *= _item_type_multiplier(row["item_type"])
            if row["item_type"] == "commands" and not command_intent:
                score *= 0.55
            fallbacks.append((score, row))
            if score <= 0.05 and semantic_score <= 0 and lexical_score <= 0 and overlap_score <= 0:
                continue
            entry = grouped.setdefault(
                row["thread_id"],
                {
                    "thread_id": row["thread_id"],
                    "title": row["title"],
                    "cwd": row["cwd"],
                    "summary": row["summary"],
                    "created_at": int(row["created_at"]),
                    "updated_at": int(row["updated_at"]),
                    "first_user_message": row["first_user_message"] or "",
                    "files": json.loads(row["files_json"] or "[]"),
                    "commands": json.loads(row["commands_json"] or "[]"),
                    "final_answer": row["final_answer"] or "",
                    "score": -1.0,
                    "matches": [],
                    "signals": {},
                },
            )
            if score > entry["score"]:
                entry["score"] = score
            entry["matches"].append(
                {
                    "item_type": row["item_type"],
                    "preview": row["preview"],
                    "score": score,
                }
            )
        if not grouped:
            for score, row in sorted(fallbacks, key=lambda item: item[0], reverse=True)[: max(limit * 3, 6)]:
                entry = grouped.setdefault(
                    row["thread_id"],
                    {
                        "thread_id": row["thread_id"],
                        "title": row["title"],
                        "cwd": row["cwd"],
                        "summary": row["summary"],
                        "created_at": int(row["created_at"]),
                        "updated_at": int(row["updated_at"]),
                        "first_user_message": row["first_user_message"] or "",
                        "files": json.loads(row["files_json"] or "[]"),
                        "commands": json.loads(row["commands_json"] or "[]"),
                        "final_answer": row["final_answer"] or "",
                        "score": -1.0,
                        "matches": [],
                        "signals": {},
                    },
                )
                if score > entry["score"]:
                    entry["score"] = score
                entry["matches"].append(
                    {
                        "item_type": row["item_type"],
                        "preview": row["preview"],
                        "score": score,
                    }
                )
        for entry in grouped.values():
            entry["matches"] = _finalize_matches(entry["matches"])
        results = sorted(grouped.values(), key=lambda item: item["score"], reverse=True)
        top_results = results[:limit]
        for entry in top_results:
            entry["signals"] = _load_thread_signals(connection, entry["thread_id"])
        return top_results
    finally:
        connection.close()


def recent_threads(db_path, cwd=None, limit=5):
    connection = open_db_readonly(db_path)
    try:
        rows = connection.execute(
            """
            SELECT id, title, cwd, summary, created_at, updated_at,
                   files_json, commands_json, final_answer, first_user_message
            FROM threads
            WHERE (? = '' OR cwd = ? OR cwd LIKE ?)
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (
                cwd or "",
                cwd or "",
                (cwd.rstrip("/") + "/%") if cwd else "",
                limit,
            ),
        ).fetchall()
        results = []
        for row in rows:
            results.append(
                {
                    "thread_id": row["id"],
                    "title": row["title"],
                    "cwd": row["cwd"],
                    "summary": row["summary"],
                    "created_at": int(row["created_at"]),
                    "updated_at": int(row["updated_at"]),
                    "first_user_message": row["first_user_message"] or "",
                    "files": json.loads(row["files_json"] or "[]"),
                    "commands": json.loads(row["commands_json"] or "[]"),
                    "final_answer": row["final_answer"] or "",
                    "score": 0.0,
                    "matches": [],
                    "signals": _load_thread_signals(connection, row["id"]),
                }
            )
        return results
    finally:
        connection.close()


def resolve_thread(db_path, thread_ref="latest", cwd=None):
    connection = open_db_readonly(db_path)
    try:
        if thread_ref == "latest":
            row = connection.execute(
                """
                SELECT * FROM threads
                WHERE (? = '' OR cwd = ? OR cwd LIKE ?)
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (
                    cwd or "",
                    cwd or "",
                    (cwd.rstrip("/") + "/%") if cwd else "",
                ),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT * FROM threads
                WHERE id = ?
                   OR id LIKE ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (thread_ref, thread_ref + "%"),
            ).fetchone()
        if not row:
            return None
        items = connection.execute(
            """
            SELECT item_type, text, preview
            FROM memory_items
            WHERE thread_id = ?
            ORDER BY ordinal ASC
            """,
            (row["id"],),
        ).fetchall()
        return {
            "thread_id": row["id"],
            "title": row["title"],
            "cwd": row["cwd"],
            "summary": row["summary"],
            "created_at": int(row["created_at"]),
            "updated_at": int(row["updated_at"]),
            "first_user_message": row["first_user_message"],
            "final_answer": row["final_answer"],
            "commands": json.loads(row["commands_json"] or "[]"),
            "files": json.loads(row["files_json"] or "[]"),
            "tool_names": json.loads(row["tool_names_json"] or "[]"),
            "items": [dict(item) for item in items],
            "signals": _load_thread_signals(connection, row["id"]),
            "rollout_path": row["rollout_path"],
        }
    finally:
        connection.close()


def status(db_path):
    connection = open_db_readonly(db_path)
    try:
        thread_count = connection.execute("SELECT COUNT(*) AS count FROM threads").fetchone()["count"]
        item_count = connection.execute("SELECT COUNT(*) AS count FROM memory_items").fetchone()["count"]
        last_indexed = _meta_get(connection, "last_indexed_at")
        vector_backend = _meta_get(connection, "vector_backend")
        embedding_provider = _meta_get(connection, "embedding_provider")
        embedding_model = _meta_get(connection, "embedding_model")
        embedding_dimensions = _meta_get(connection, "embedding_dimensions")
        cached_point_count = _meta_get(connection, "vector_point_count")
        qdrant_path = _meta_get(connection, "vector_qdrant_path", default_qdrant_path(db_path))
        point_count = 0
        vector_lock_error = None
        if vector_backend == "qdrant":
            try:
                client = open_qdrant(qdrant_path)
            except RuntimeError as exc:
                vector_lock_error = str(exc)
                point_count = int(cached_point_count) if cached_point_count else 0
            else:
                try:
                    point_count = collection_point_count(client, COLLECTION_NAME)
                finally:
                    client.close()
        return {
            "db_path": db_path,
            "thread_count": int(thread_count),
            "item_count": int(item_count),
            "fts_enabled": _fts_enabled(connection),
            "last_indexed_at": int(last_indexed) if last_indexed else None,
            "vector_backend": vector_backend or "none",
            "embedding_provider": embedding_provider or "none",
            "embedding_model": embedding_model or "",
            "embedding_dimensions": int(embedding_dimensions) if embedding_dimensions else None,
            "qdrant_path": qdrant_path,
            "vector_points": point_count,
            "vector_lock_error": vector_lock_error,
        }
    finally:
        connection.close()


def ensure_populated(codex_home=None, db_path=None):
    codex_home = default_codex_home(codex_home)
    db_path = db_path or default_db_path(codex_home)
    info = status(db_path)
    if info["thread_count"] > 0:
        return info
    index_threads(codex_home=codex_home, db_path=db_path)
    return status(db_path)


def sync_latest_threads(codex_home=None, db_path=None, limit=3, cwd=None):
    codex_home = default_codex_home(codex_home)
    db_path = db_path or default_db_path(codex_home)
    recent = discover_threads(codex_home=codex_home, limit=max(1, limit))
    if cwd:
        recent = [thread for thread in recent if _cwd_matches(thread.cwd, cwd)]
    connection = open_db_readonly(db_path)
    try:
        stale_ids = []
        for thread in recent:
            row = connection.execute(
                "SELECT updated_at FROM threads WHERE id = ?",
                (thread.id,),
            ).fetchone()
            if not row or int(row["updated_at"]) != int(thread.updated_at):
                stale_ids.append(thread.id)
    finally:
        connection.close()
    payload = {
        "db_path": db_path,
        "checked": len(recent),
        "stale_thread_ids": stale_ids,
        "thread_ids": stale_ids[:],
        "indexed": 0,
        "skipped": len(recent) - len(stale_ids),
    }
    if stale_ids:
        result = index_threads(
            codex_home=codex_home,
            db_path=db_path,
            thread_ids=stale_ids,
        )
        payload["indexed"] = result["indexed"]
        payload["skipped"] = result["skipped"] + (len(recent) - len(stale_ids))
    return payload


def format_timestamp(timestamp):
    if not timestamp:
        return "unknown"
    return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")


def render_search_results(results):
    if not results:
        return "No memory matches found."
    lines = []
    for index, result in enumerate(results, start=1):
        lines.append(
            "{}. [{}] {} ({})".format(
                index,
                result["thread_id"][:8],
                result["title"] or "Untitled thread",
                format_timestamp(result["updated_at"]),
            )
        )
        lines.append("   cwd: {}".format(compact_path(result["cwd"])))
        lines.append("   summary: {}".format(shorten(result["summary"], 260)))
        if result.get("matches"):
            lines.append(
                "   matches: {}".format(
                    " | ".join(
                        "{}: {}".format(match["item_type"], shorten(match["preview"], 120))
                        for match in result["matches"]
                    )
                )
            )
    return "\n".join(lines)


def render_sync(payload):
    lines = [
        "db: {}".format(payload["db_path"]),
        "checked: {}".format(payload["checked"]),
        "indexed: {}".format(payload["indexed"]),
        "skipped: {}".format(payload["skipped"]),
    ]
    if payload.get("thread_ids"):
        lines.append("thread ids: {}".format(", ".join(payload["thread_ids"])))
    return "\n".join(lines)


def render_context(results):
    if not results:
        return "No relevant memory found."
    lines = ["Relevant Codex memory:"]
    for result in results:
        ask = clean_message_text(result.get("first_user_message", "")) or result["title"] or ""
        signals = result.get("signals", {}) or {}
        outcome = signals.get("result", "") or summarize_outcome_text(result.get("final_answer", ""))
        useful_files = prioritize_files(result.get("files", []), limit=4)
        useful_commands = prioritize_commands(result.get("commands", []), limit=3)
        lines.append(
            "- [{}] {} | {}".format(
                result["thread_id"][:8],
                result["title"] or "Untitled thread",
                format_timestamp(result["updated_at"]),
            )
        )
        lines.append("  cwd: {}".format(compact_path(result["cwd"])))
        if ask:
            lines.append("  ask: {}".format(shorten(ask, 180)))
        lines.append("  summary: {}".format(shorten(result["summary"], 220)))
        if outcome:
            lines.append("  state: {}".format(shorten(outcome, 180)))
        if signals.get("decision"):
            lines.append("  decision: {}".format(shorten(signals["decision"], 180)))
        if signals.get("observation"):
            lines.append("  observation: {}".format(shorten(signals["observation"], 180)))
        if useful_files:
            pretty_files = ", ".join(compact_path(path) for path in useful_files)
            lines.append("  useful files: {}".format(pretty_files))
        if useful_commands:
            pretty_commands = "; ".join(shorten(command, 72) for command in useful_commands)
            lines.append("  useful commands: {}".format(pretty_commands))
    return "\n".join(lines)


def render_brief(brief):
    if not brief:
        return "Thread not found."
    lines = [
        "{} [{}]".format(brief["title"] or "Untitled thread", brief["thread_id"][:8]),
        "updated: {}".format(format_timestamp(brief["updated_at"])),
        "cwd: {}".format(compact_path(brief["cwd"])),
        "rollout: {}".format(compact_path(brief["rollout_path"])),
        "",
        "summary:",
        brief["summary"] or "(none)",
    ]
    signals = brief.get("signals", {}) or {}
    if signals.get("decision"):
        lines.extend(["", "decision:", signals["decision"]])
    if signals.get("observation"):
        lines.extend(["", "observation:", signals["observation"]])
    if brief.get("commands"):
        lines.extend(["", "commands:"])
        for command in brief["commands"][:10]:
            lines.append("- {}".format(command))
    if brief.get("files"):
        lines.extend(["", "files:"])
        for path in brief["files"][:12]:
            lines.append("- {}".format(compact_path(path)))
    if brief.get("final_answer"):
        lines.extend(["", "final answer:", brief["final_answer"]])
    return "\n".join(lines)


def render_timeline(brief):
    if not brief:
        return "Thread not found."
    lines = [
        "{} [{}]".format(brief["title"] or "Untitled thread", brief["thread_id"][:8]),
        "updated: {}".format(format_timestamp(brief["updated_at"])),
        "",
        "timeline:",
    ]
    interesting_types = ("user", "decision", "observation", "result", "answer")
    entries = [
        item for item in brief.get("items", []) if item.get("item_type") in interesting_types
    ]
    if not entries:
        lines.append("(no structured memory items)")
        return "\n".join(lines)
    for index, item in enumerate(entries, start=1):
        lines.append(
            "{}. {}: {}".format(
                index,
                item["item_type"],
                shorten(item.get("preview") or item.get("text", ""), 220),
            )
        )
    return "\n".join(lines)
