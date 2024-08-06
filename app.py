import os
import base64
from typing import List
import gc
import random
import time
import uuid
from collections import defaultdict

from IPython.display import Markdown, display

from rag import RAG
from retriever import Retriever

import streamlit as st


class QueryEngine:

    def __init__(self):
        self.query_engine = RAG()

    def response(self, query):
        response = self.query_engine.query(query)

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

# Sidebar
st.sidebar.header("What would you like to do❓")
# Add radio buttons for selection

# Initialize or update engine type in session state


col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"RAG over LlamaIndex docs! 🚀")


with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
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