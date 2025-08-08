# Install needed packages:
# pip install langchain langchain-community sentence-transformers faiss-cpu gradio ollama

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import json, os
import gradio as gr

# ---------- 1) Load JSON data ----------
DIR = "processed_json"
docs = []
for file in os.listdir(DIR):
    if file.endswith(".json"):
        with open(os.path.join(DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            text = data.get("text", "")
            docs.append(text)

# ---------- 2) Create embeddings ----------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------- 3) Build FAISS index ----------
db = FAISS.from_texts(docs, embeddings)

# ---------- 4) Local LLM via Ollama ----------
# Make sure Ollama is running: `ollama serve` (usually runs automatically)
# And that you've pulled your model: `ollama pull mistral`
llm = Ollama(model="mistral", temperature=0.2)

# ---------- 5) RetrievalQA chain ----------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

# ---------- 6) Web interface ----------
def ask_bot(query):
    res = qa(query)
    answer = res["result"]
    sources = [doc.page_content for doc in res["source_documents"]]
    return answer, "\n\n".join(sources)

iface = gr.Interface(
    fn=ask_bot,
    inputs="text",
    outputs=["text", "text"],
    title="Local LCA Proxy Modelling Chatbot",
    description="Ask questions about LCA proxy modelling using local JSON data."
)

iface.launch(server_name="127.0.0.1", server_port=7860)
