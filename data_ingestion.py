from qdrant_client import QdrantClient, models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
from data_reader import DataPrep
from utils import start_qdrant_client, batch_iterate
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "binary-quantization"
BATCH_SIZE = 1000


def upload_collection_to_vector_store(vectors, payload, client):

    num_rows = len(vectors)

    try:

        if not client.collection_exists(collection_name=COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=8192, distance=models.Distance.COSINE, on_disk=True),
                optimizers_config=models.OptimizersConfigDiff(default_segment_number=5, indexing_threshold=0),
                quantization_config=models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True)),
                shard_number=4
            )
            logger.info(f"Successfully Created '{COLLECTION_NAME}' collection in the Qdrant Vector Store!")

        for batch_payload, batch_vectors in tqdm(
            zip(batch_iterate(payload, BATCH_SIZE), batch_iterate(vectors, BATCH_SIZE)),
            total=num_rows // BATCH_SIZE,
            desc="Ingesting in batches",
        ):
            client.upload_collection(
                collection_name=COLLECTION_NAME,
                vectors=batch_vectors,
                payload=batch_payload,
                parallel=16
            )

        logger.info(f"Successfully Uploaded the vectors and payload to the '{COLLECTION_NAME}' collection!")

    except Exception as e:
        logger.error(e)


def main(ingestion_dir: str, indexing: bool = False):
    
    if indexing == False:

        client = start_qdrant_client()

        dataprep = DataPrep(input_dir=ingestion_dir)
        chunks = dataprep.load_parse_chunk_data()
        vectors, payload = dataprep.vectorize(chunks=chunks)

        upload_collection_to_vector_store(vectors=vectors, payload=payload, client=client)

    if indexing == True:
        client.update_collection(
            collection_name=COLLECTION_NAME,
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=20000
            )
        )

    

if __name__ == "__main__":
    
    main(ingestion_dir="/teamspace/studios/this_studio/langchain", indexing=False)
