import truststore
truststore.inject_into_ssl()

import os

import chromadb
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

# Config: קבועים, אפשר לשנות את המודל דרך .env
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "hotel"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50


def load_hotel_documents():
    # Load: טעינת כל קבצי המלון מהתיקייה (שם הקובץ נשמר אוטומטית ב-metadata)
    return SimpleDirectoryReader(DATA_DIR, required_exts=[".md"]).load_data()


def split_into_chunks(documents):
    # Split: חלוקה ל-Chunks לפי משפטים
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.get_nodes_from_documents(documents)


def build_index(nodes):
    # Reset: מחיקת האוסף הקודם, כדי שכל הרצה תבנה את האינדקס מחדש
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)
    collection = chroma.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    # Store: יצירת Embeddings ושמירה ב-Chroma
    storage_context = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=OpenAIEmbedding(model=EMBED_MODEL))
    return collection.count()


if __name__ == "__main__":
    documents = load_hotel_documents()
    print(f"Documents: נטענו {len(documents)} מסמכים")
    nodes = split_into_chunks(documents)
    print(f"Chunks: נוצרו {len(nodes)} קטעים")
    count = build_index(nodes)
    print(f"Done: נשמרו {count} קטעים ב-Chroma")
