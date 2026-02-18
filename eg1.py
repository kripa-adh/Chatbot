from ollama import Client
import json
import chromadb
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama

chat_bot= Client(host=f"http://localhost:11434")

client = chromadb.PersistentClient()
remote_client = Client(host=f"http://localhost:11434")


counter=0
current_count=0
if os.path.exists("counter.txt"):
    with open("counter.txt", "r") as f:     
        current_count = int(f.read().strip())
collection = client.get_or_create_collection(name="articles_demo")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    separators=['.','\n']
)
print(f"Current count: {current_count}")

with open("articles.jsonl", "r", encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < current_count:
            print(f"Skipping article",i,counter)
            continue
        print(f"added new line {i}")
        article = json.loads(line)
        content = article["content"]

        chunks = [c.strip() for c in splitter.split_text(content) if c.strip()]
        for j, chunk in enumerate(chunks):
                print(chunk)
                response = remote_client.embed(model="nomic-embed-text", input=f"search_document: {chunk}")
                embedding = response["embeddings"][0]


        collection.add(
            ids=[f"article_{i}_chunk_{j}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"title": article["title"],"chunk":j}],
        )
counter+=1
with open("counter.txt", "w") as f:
    f.write(str(counter))
print("Database built successfully!")

# query = "what are update about gold?"
# # query = "are there any predicted hindrance for upcoming election ?"
# query_embed = remote_client.embed(model="nomic-embed-text", input=f"query: {query}")["embeddings"][0]
# results = collection.query(query_embeddings=[query_embed], n_results=1)
# print(f"\nQuestion: {query}")
# print(f'\n Title :  {results["documents"][0]} ')
while True:
    user_input = input("How may I assist you? ")
    query_embd=remote_client.embed(model="nomic-embed-text", input=f"query: {user_input}")["embeddings"][0]
    results = collection.query(query_embeddings=[query_embd], n_results=2)
    
    retrieved_docs = results['documents'][0]
    context = "\n\n".join(retrieved_docs)

    # print(context)
    prompt = f"""You are a helpful assistant. Answer the question based on the context provided. Use the information in the context to form your answer. If contextdoes not have enough information just say "I don't know"

    Context: {context}

    Question: {user_input}

    Answer:"""
    print(prompt)
    response = chat_bot.generate(
            model="qwen2.5:3b",
            prompt=prompt,
            options={
                "temperature": 0.1
            }
        )

    answer = response['response']

    print(answer)
