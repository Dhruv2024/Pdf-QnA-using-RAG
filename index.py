from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# for openai embedding
# from langchain_openai import OpenAIEmbeddings

# for gemini embedding
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

#for ollama embedding
from langchain_ollama import OllamaEmbeddings

from langchain_qdrant import QdrantVectorStore
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

pdf_path=Path(__file__).parent / "nodejs.pdf"

# Load this file in python program
loader = PyPDFLoader(pdf_path)
docs=loader.load()

#Split the docs into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
chunks = text_splitter.split_documents(documents=docs)

#Vector Embedding
# embedding_model=OpenAIEmbeddings(
#     model="text-embedding-3-large",
# )
# embedding_model=GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-001"
# )
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
)


vector_store=QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

print("Indexing of documents is completed...")