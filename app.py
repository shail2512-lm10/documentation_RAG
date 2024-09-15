import os
import base64
from typing import List
import gc
import random
import time
import uuid
from collections import defaultdict
import torch
from IPython.display import Markdown, display
from huggingface_hub import login
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag import RAG

import streamlit as st


HF_TOKEN = os.getenv("HF_API_KEY")
login(HF_TOKEN)

@st.cache_resource
def setup_llm():
    
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True)
    # bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_use_double_quant=True)

    llm = HuggingFaceLLM(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    # context_window=32000,
    # max_new_tokens=512,
    model_kwargs={"quantization_config": quantization_config},
    # messages_to_prompt=messages_to_prompt,
    # completion_to_prompt=completion_to_prompt,
    device_map="auto")

    return llm

@st.cache_resource
def setup_embedding():
    embed_model = HuggingFaceEmbedding(model_name="dunzhang/stella_en_1.5B_v5", trust_remote_code=True, cache_folder="./hf_cache")
    return embed_model


class QueryEngine:

    def __init__(self):
        self.query_engine = RAG(llm=setup_llm, embedding=setup_embedding)

    def response(self, query):
        response = self.query_engine.query(query=query)
        # return str(response)
        for token in str(response).split(" "):
            yield token + " "
            time.sleep(0.02)

    
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

if "query_engine" not in st.session_state:
    st.session_state.query_engine = QueryEngine()


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


st.header(f"RAG over HuggingFace, LlamaIndex, LangChain docs! 🚀")
st.button("Clear the chat❓ ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        streaming_response = st.session_state.query_engine.response(prompt)

        for chunk in streaming_response:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})