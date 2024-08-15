import time
import logging
from pprint import pprint
from qdrant_client import QdrantClient, models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils import start_qdrant_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:

    def __init__(self, embedding, collection_name="binary-quantization"):
        self.collection_name = collection_name
        self.embed_model = embedding()
        self.qdrant_client = self._start_qdrant_client()

    def _start_qdrant_client(self):
        client = start_qdrant_client()
        return client

    def search(self, query):
        query_embedding = self.embed_model.get_query_embedding(query)

        start = time.time()

        result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            search_params=models.SearchParams(quantization=models.QuantizationSearchParams(ignore=False, rescore=True, oversampling=2.0)),
            timeout=1000
        )

        end = time.time()
        elapsed_time = end - start

        logger.info(f"Execution Time for Search: {elapsed_time:.4f} seconds")

        return result