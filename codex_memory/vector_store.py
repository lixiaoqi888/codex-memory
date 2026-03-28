import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+",
)

try:
    from qdrant_client import QdrantClient, models
    QDRANT_IMPORT_ERROR = None
except Exception as exc:
    QdrantClient = None
    models = None
    QDRANT_IMPORT_ERROR = exc


COLLECTION_NAME = "memory_items"


def default_qdrant_path(db_path):
    return os.path.join(os.path.dirname(os.path.abspath(db_path)), "qdrant")


def open_qdrant(qdrant_path):
    if QdrantClient is None:
        raise RuntimeError(
            "qdrant-client is not available: {}".format(QDRANT_IMPORT_ERROR)
        )
    os.makedirs(qdrant_path, exist_ok=True)
    return QdrantClient(path=qdrant_path)


def collection_exists(client, collection_name=COLLECTION_NAME):
    return client.collection_exists(collection_name)


def recreate_collection(client, vector_size, collection_name=COLLECTION_NAME):
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def upsert_points(client, ids, vectors, payloads, collection_name=COLLECTION_NAME):
    points = []
    for point_id, vector, payload in zip(ids, vectors, payloads):
        points.append(
            models.PointStruct(
                id=int(point_id),
                vector=vector,
                payload=payload,
            )
        )
    if points:
        client.upsert(collection_name, points=points)


def delete_points(client, ids, collection_name=COLLECTION_NAME):
    ids = [int(point_id) for point_id in ids if point_id is not None]
    if not ids or not client.collection_exists(collection_name):
        return
    client.delete(collection_name, points_selector=models.PointIdsList(points=ids))


def query_vectors(client, vector, limit, collection_name=COLLECTION_NAME):
    if not client.collection_exists(collection_name):
        return []
    response = client.query_points(
        collection_name,
        query=vector,
        limit=limit,
        with_payload=True,
    )
    return list(response.points)


def collection_point_count(client, collection_name=COLLECTION_NAME):
    if not client.collection_exists(collection_name):
        return 0
    return int(client.count(collection_name, exact=True).count)
