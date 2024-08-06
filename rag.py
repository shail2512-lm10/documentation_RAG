from retriever import Retriever
from prompt_template import qa_prompt_tmpl_str
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig
import torch
from dotenv import load_dotenv
import os
from huggingface_hub import login



load_dotenv()

HF_TOKEN = os.environ.get("HF_API_KEY")
login(HF_TOKEN)

class RAG:

    def __init__(self, llm_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", request_timeout: float = 100.0):
        self.request_timeout = request_timeout
        self.llm_name = llm_name
        self.retriever = Retriever()
        self.llm = self._setup_llm()

    def _setup_llm(self):
        
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True)

        llm = HuggingFaceLLM(model_name=self.llm_name,
        tokenizer_name=self.llm_name,
        context_window=3900,
        max_new_tokens=256,
        model_kwargs={"quantization_config": quantization_config},
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        # messages_to_prompt=messages_to_prompt,
        # completion_to_prompt=completion_to_prompt,
        device_map="auto")

        return llm


    def generate_context(self, query: str) -> str:
        result = self.retriever.search(query=query)
        context = [dict(data) for data in result]
        combined_prompt = []

        for entry in context:
            related_context = entry["payload"]["text"]

            prompt = f"Related Context: {related_context}\n"

            combined_prompt.append(prompt)

        return "\n\n----\n\n".join(combined_prompt)


    def query(self, query: str, streaming: bool = False):
        context = self.generate_context(query=query)
        prompt = qa_prompt_tmpl_str.format(context=context, query=query)

        if streaming:
            response = self.llm.stream_complete(prompt)
        else:
            response = self.llm.complete(prompt)

        return response

