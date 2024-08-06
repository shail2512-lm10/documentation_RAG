from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
#from llama_index.core import VectorStoreIndex, StorageContext
# from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.readers.file import UnstructuredReader
from qdrant_client import QdrantClient, models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm

collection_name = "binary-quantization"
batch_size = 1000


def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

client = QdrantClient(
    url="http://localhost:6333",
    prefer_grpc=True,
)


dir_reader = SimpleDirectoryReader(
    input_dir="/teamspace/studios/this_studio/llamaindex-docs",
    file_extractor={".html": UnstructuredReader()},
)
documents = dir_reader.load_data()

node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents=documents)


payload = []
docs = []

for index, text_nodes in enumerate(nodes):
    text_field = text_nodes.text
    payload.append(
        {
            "text": text_field,
            "id": index
        }
    )
    docs.append(text_field)


    
if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE, on_disk=True),
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=5, indexing_threshold=0),
        quantization_config=models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True)),
        shard_number=4
    )

embed_model = HuggingFaceEmbedding(model_name="dunzhang/stella_en_1.5B_v5", trust_remote_code=True, cache_folder="./hf_cache")
doc_vectors = embed_model.get_text_embedding_batch(docs)

print("Finished generating vectors!")
print("Uploading to vector store")

num_rows = len(docs)

for batch_docs, batch_vectors in tqdm(
    zip(batch_iterate(payload, batch_size), batch_iterate(doc_vectors, batch_size)),
    total=num_rows // batch_size,
    desc="Ingesting in batches",
):
    client.upload_collection(
        collection_name=collection_name,
        vectors=batch_vectors,
        payload=batch_docs
    )
