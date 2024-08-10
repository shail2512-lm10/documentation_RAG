from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.file import UnstructuredReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class DataPrep():
    def __init__(self, input_dir: str, embedding_model_name: str = "dunzhang/stella_en_1.5B_v5"):
        self.input_dir = input_dir
        self.embedding_model_name = embedding_model_name
        self.embedding_model = self._setup_embedding()

    def _setup_embedding(self):
        embedding_model = HuggingFaceEmbedding(model_name=self.embedding_model_name, trust_remote_code=True, cache_folder="./hf_cache")
        return embedding_model

    
    def load_parse_chunk_data(self):
        dir_reader = SimpleDirectoryReader(
                        input_dir=self.input_dir,
                        file_extractor={".html": UnstructuredReader()},
                    )
        documents = dir_reader.load_data(show_progress=True, num_workers=4)

        node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
        chunks = node_parser.get_nodes_from_documents(documents=documents, show_progress=True)

        return chunks

    def vectorize(self, chunks):
        payload = []
        docs = []

        for index, text_chunks in enumerate(chunks):
            text_field = text_chunks.text
            payload.append(
                {
                    "text": text_field,
                    "id": index
                }
            )
            docs.append(text_field)

        vectors = self.embedding_model.get_text_embedding_batch(docs, show_progress=True)

        return vectors, payload



