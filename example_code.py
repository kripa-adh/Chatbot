from ollama import Client
import json
import chromadb

client = chromadb.Client()
remote_client = Client(host=f"http://localhost:11434")
collection = client.get_or_create_collection(name="articles_demo")


print("Reading articles.jsonl and generating embeddings...")
with open("articles.jsonl", "r") as f:
    for i, line in enumerate(f):
        article = json.loads(line)
        content = article["content"]

        response = remote_client.embed(model="nomic-embed-text", input=f"search_document: {content}")
        embedding = response["embeddings"][0]

        collection.add(
            ids=[f"article_{i}"],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{"title": article["title"]}],
        )

print("Database built successfully!")


# query = "what are different problems provinces of nepal are facing?"
query = "are there any predicted hindrance for upcoming election ?"
query_embed = remote_client.embed(model="nomic-embed-text", input=f"query: {query}")["embeddings"][0]
results = collection.query(query_embeddings=[query_embed], n_results=1)
print(f"\nQuestion: {query}")
print(f'\n Title : {results["metadatas"][0][0]["title"]} \n {results["documents"][0][0]} ')
