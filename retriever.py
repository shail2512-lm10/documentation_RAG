import time
import logging
from pprint import pprint
from qdrant_client import QdrantClient, models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils import start_qdrant_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:

    def __init__(self, embed_model_name="dunzhang/stella_en_1.5B_v5", collection_name="binary-quantization"):
        self.embed_model_name = embed_model_name
        self.collection_name = collection_name
        self.embed_model = self._load_embed_model()
        self.qdrant_client = self._start_qdrant_client()

    def _start_qdrant_client(self):
        client = start_qdrant_client()
        return client

    def _load_embed_model(self):
        embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name, trust_remote_code=True, cache_folder="./hf_cache")
        return embed_model

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