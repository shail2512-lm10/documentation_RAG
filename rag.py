from retriever import Retriever
from prompt_template import qa_prompt_tmpl_str


class RAG:

    def __init__(self, llm, embedding, request_timeout: float = 100.0):
        self.request_timeout = request_timeout
        self.retriever = Retriever(embedding)
        self.llm = llm()


    def generate_context(self, query: str) -> str:
        result = self.retriever.search(query=query)
        context = [dict(data) for data in result]
        combined_prompt = []

        for entry in context:
            related_context = entry["payload"]["text"]

            prompt = f"Context: {related_context}\n"

            combined_prompt.append(prompt)

        return "\n\n----\n\n".join(combined_prompt)


    def query(self, query: str):
        context = self.generate_context(query=query)
        prompt = qa_prompt_tmpl_str.format(context=context, query=query)
        response = self.llm.complete(prompt)

        return response

