from qdrant_client import QdrantClient
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


def start_qdrant_client():
    try:
        client = QdrantClient(
            url="http://localhost:6333",
            prefer_grpc=True,
        )
        return client
    except Exception as e:
        logger.error(e)


def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]