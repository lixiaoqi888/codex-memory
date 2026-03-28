import hashlib
import json
import math
import re
from typing import Dict, Iterable, List, Sequence, Set


Vector = Dict[int, float]
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+|[\u3400-\u9fff]+")
SPLIT_RE = re.compile(r"[\/_.:-]+")


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def _contains_cjk(text):
    return bool(re.search(r"[\u3400-\u9fff]", text))


def keyword_tokens(text, limit=24):
    seen = set()
    tokens = []
    normalized = normalize_text(text).lower()
    for token in TOKEN_RE.findall(normalized):
        if token and token not in seen:
            seen.add(token)
            tokens.append(token)
        if len(tokens) >= limit:
            break
        if any(ch in token for ch in "/._:-"):
            for piece in SPLIT_RE.split(token):
                if len(piece) >= 2 and piece not in seen:
                    seen.add(piece)
                    tokens.append(piece)
                if len(tokens) >= limit:
                    break
    return tokens


def iter_features(text):
    normalized = normalize_text(text).lower()
    sample = normalized[:1200]
    for token in TOKEN_RE.findall(sample):
        if not token:
            continue
        yield token, 1.0
        if any(ch in token for ch in "/._:-"):
            for piece in SPLIT_RE.split(token):
                if len(piece) >= 2:
                    yield "part:" + piece, 1.1
        if _contains_cjk(token):
            for size in (2, 3):
                if len(token) >= size:
                    for index in range(0, len(token) - size + 1):
                        yield "cjk:" + token[index : index + size], 1.25

    compact = sample.replace(" ", "")[:400]
    for size in (3, 4):
        if len(compact) >= size:
            for index in range(0, len(compact) - size + 1):
                yield "gram:" + compact[index : index + size], 0.18


def _stable_hash(token):
    payload = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(payload, "big")


def text_to_vector(text, dimension=384):
    bins = {}
    for feature, weight in iter_features(text):
        raw = _stable_hash(feature)
        bucket = raw % dimension
        sign = -1.0 if raw & 1 else 1.0
        bins[bucket] = bins.get(bucket, 0.0) + (sign * weight)
    norm = math.sqrt(sum(value * value for value in bins.values()))
    if norm <= 0:
        return {}
    return dict(
        (bucket, value / norm)
        for bucket, value in bins.items()
        if abs(value / norm) >= 1e-6
    )


def encode_vector(vector):
    pairs = []
    for bucket, value in sorted(vector.items()):
        pairs.append([bucket, round(float(value), 6)])
    return json.dumps(pairs, separators=(",", ":"))


def decode_vector(payload):
    if not payload:
        return {}
    pairs = json.loads(payload)
    return dict((int(bucket), float(value)) for bucket, value in pairs)


def cosine_similarity(left, right):
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    total = 0.0
    for bucket, value in left.items():
        total += value * right.get(bucket, 0.0)
    return total


def token_overlap(query, candidate):
    left = set(keyword_tokens(query, limit=18))
    right = set(keyword_tokens(candidate, limit=28))
    if not left or not right:
        return 0.0
    return len(left & right) / float(len(left))


def top_keywords(text, limit=12):
    keywords = []
    for token in keyword_tokens(text, limit=limit * 2):
        if token.startswith("part:"):
            token = token[5:]
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords

