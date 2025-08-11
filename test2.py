# Install needed packages:
# pip install langchain langchain-community sentence-transformers faiss-cpu gradio ollama

from langchain_community.embeddings import HuggingFaceEmbeddings

#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema import Document
import json, os
import gradio as gr

# ---------- 1) Load JSON data ----------
DIR = "processed_json"
docs = []
for file in os.listdir(DIR):
    if file.endswith(".json"):
        with open(os.path.join(DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            text = data.get("text_for_embedding", "")
            # Add source metadata with filename or id
            doc = Document(page_content=text, metadata={"source": file})
            docs.append(doc)

# ---------- 2) Create embeddings ----------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
# ---------- 3) Build FAISS index ----------
db = FAISS.from_documents(docs, embeddings)

# ---------- 4) Local LLM via Ollama ----------
# Make sure Ollama is running: `ollama serve` (usually runs automatically)
# And that you've pulled your model: `ollama pull mistral`
llm = Ollama(model="qwen3:0.6b", temperature=0.2)

# ---------- 5) Create RetrievalQAWithSourcesChain explicitly ----------
qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 4}),
    chain_type="stuff",  # or "map_reduce" if you want
    return_source_documents=True,
)

# ---------- 6) Web interface ----------
def ask_bot(query):
    # Retrieve docs explicitly
    retrieved_docs = qa.retriever.get_relevant_documents(query)
    
    print("\n=== Retrieved documents ===")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"Document #{i}:\n{doc.page_content}\n{'-'*40}")
    
    # Show final prompt sent to the LLM
    prompt_template = qa.combine_documents_chain.llm_chain.prompt
    final_prompt = prompt_template.format_prompt(
    question=query,
    summaries=[doc.page_content for doc in retrieved_docs]
              ).to_string()
    print("\n=== Final prompt sent to LLM ===")
    print(final_prompt)
    print("="*60)

    # Run the chain and get answer + sources
    res = qa({"question": query}, return_only_outputs=False)
    answer = res["answer"]
    sources = [doc.page_content for doc in res["source_documents"]]
    
    print("\n=== LLM Answer ===")
    print(answer)
    print("="*60)

    return answer, "\n\n".join(sources)


iface = gr.Interface(
    fn=ask_bot,
    inputs="text",
    outputs=["text", "text"],
    title="Local LCA Proxy Modelling Chatbot - Debug Mode",
    description="Ask questions about LCA proxy modelling using local JSON data."
)

iface.launch(server_name="127.0.0.1", server_port=7860)
