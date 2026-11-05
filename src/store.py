import chromadb 
 
from src.config import CHROMA_DIR 
from src.embedder import embed_texts 
 
COLLECTION_NAME = "ragscope" 
 
 
def get_client(): 
    return chromadb.PersistentClient(path=str(CHROMA_DIR)) 
 
 
def get_collection(name=COLLECTION_NAME): 
    return get_client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"}) 
 
 
def reset_collection(name=COLLECTION_NAME): 
    client = get_client() 
    try: 
        client.delete_collection(name) 
    except Exception: 
        pass 
    return get_collection(name) 
 
 
def add_chunks(chunks, collection=None): 
    collection = collection or get_collection() 
    if not chunks: 
        return 0 
    ids = [f"{c['source']}::{c['chunk_id']}" for c in chunks] 
    texts = [c["text"] for c in chunks] 
    metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks] 
    collection.upsert(ids=ids, documents=texts, embeddings=embed_texts(texts), metadatas=metadatas) 
    return len(chunks) 
