# RAG Over Documentation - LLM APP

Try it out yourself in Lightning AI Studios:

<a target="_blank" href="https://lightning.ai/shail251298/studios/documentation-rag">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>

This LLM Application enables Search as well as RAG (Retrieval Augmented Generation) over documentation of HuggingFace (transformers, diffusers), LLamaIndex, and LangCHain libraries. But this can be extended to as many libraries as possible.



https://github.com/user-attachments/assets/dd90466b-415e-4c3b-8434-0591de6afc72



![rag drawio](https://github.com/user-attachments/assets/abd1207b-0ba6-4943-91d7-b56b7c7fd2f5)





Steps to Run:

1. Open this studio
2. Make sure you have HuggingFace Token stored in your studio as a secret. If not follow the below steps and set the token as `HF_API_KEY` secret. https://lightning.ai/docs/overview/Studios/secrets
3. Open the terminal and run: `docker compose up qdrant` this will start a self hosted qdrant vector store. It may take upto 5 minutes to fully setup
4. Run the streamlit app

Click the Streamlit icon and select "New App"

Run the App. Make sure the streamlit command is correct for app.py 
