import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Directory containing your JSON files
DIR = "processed_json"

# Load documents from JSON files
docs = []
doc_ids = []
for file in os.listdir(DIR):
    if file.endswith(".json"):
        with open(os.path.join(DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            text = data.get("text_for_embedding") or data.get("text") or ""
            if text.strip():
                docs.append(text)
                doc_ids.append(data.get("id", file))

print(f"Loaded {len(docs)} documents for indexing.")

# Create embeddings (runs on CPU by default)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS index
db = FAISS.from_texts(docs, embeddings)

# Create retriever
retriever = db.as_retriever(search_kwargs={"k": 10})

print("Ready for retrieval testing. Type your query and press Enter (empty input to quit).")

while True:
    query = input("\nEnter query: ").strip()
    if not query:
        break

    results = retriever.get_relevant_documents(query)

    if not results:
        print("No documents retrieved.")
        continue

    print(f"\nTop {len(results)} documents retrieved:")
    for i, doc in enumerate(results, start=1):
        print(f"\nDocument #{i}:")
        print("-" * 40)
        print(doc.page_content)
        print("-" * 40)
