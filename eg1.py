from ollama import Client
import json
import chromadb
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------
# Setup Clients
# ---------------------------
chat_bot = Client(host="http://localhost:11434")
embed_client = Client(host="http://localhost:11434")

# ---------------------------
# Setup ChromaDB
# ---------------------------
chroma_client = chromadb.PersistentClient()
collection = chroma_client.get_or_create_collection(name="articles_demo")

# ---------------------------
# Load Counter
# ---------------------------
counter_file = "counter.txt"
current_count = 0

if os.path.exists(counter_file):
    with open(counter_file, "r") as f:
        current_count = int(f.read().strip())

print(f"Current count: {current_count}")

# ---------------------------
# Text Splitter
# ---------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    separators=['.', '\n']
)

# ---------------------------
# Build / Update Vector DB
# ---------------------------
if os.path.exists("articles.jsonl"):
    with open("articles.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):

            # Skip already processed articles
            if i < current_count:
                continue

            print(f"Adding article {i}")

            article = json.loads(line)
            content = article["content"]
            chunks = [c.strip() for c in splitter.split_text(content) if c.strip()]

            for j, chunk in enumerate(chunks):

                response = embed_client.embed(
                    model="nomic-embed-text",
                    input=f"search_document: {chunk}"
                )

                embedding = response["embeddings"][0]

                collection.add(
                    ids=[f"article_{i}_chunk_{j}"],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "title": article.get("title", ""),
                        "chunk": j
                    }],
                )

            # Update counter AFTER article processed
            current_count = i + 1

    # Save updated counter
    with open(counter_file, "w") as f:
        f.write(str(current_count))

    print("Database updated successfully!")

else:
    print("articles.jsonl not found!")

# ---------------------------
# Chat Loop
# ---------------------------
print("\nType 'exit' to stop the chatbot.\n")

while True:
    user_input = input("Question:")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye")
        break

    # Embed query
    query_embedding = embed_client.embed(
        model="nomic-embed-text",
        input=f"query: {user_input}"
    )["embeddings"][0]

    # Retrieve similar docs
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    retrieved_docs = results["documents"][0]
    context = "\n\n".join(retrieved_docs)

    # Create prompt
    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't know".

Context:
{context}

Question:
{user_input}

Answer:
"""

    # Generate answer
    response = chat_bot.generate(
        model="qwen2.5:3b",
        prompt=prompt,
        options={"temperature": 0.1}
    )

    answer = response["response"].strip()

    print(f"\nBot: {answer}\n")
