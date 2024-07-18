from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.readers.file import UnstructuredReader
import qdrant_client


load_dotenv()

if __name__ == "__main__":

    dir_reader = SimpleDirectoryReader(
        input_dir="/teamspace/studios/this_studio/llamaindex-docs-tmp",
        file_extractor={".html": UnstructuredReader()},
    )
    documents = dir_reader.load_data()

    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
    #nodes = node_parser.get_nodes_from_documents(documents=documents)
    
