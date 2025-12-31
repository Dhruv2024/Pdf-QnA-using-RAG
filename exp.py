from langchain_ollama import OllamaEmbeddings
# trying out Ollama embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)
text = "LangChain is the framework for building context-aware reasoning applications"
single_vector = embeddings.embed_query(text)
print(str(single_vector)[:100])  # Show the first 100 characters of the vector

