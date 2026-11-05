from src.config import settings 
from src.embedder import embed_query 
from src.store import get_collection 
 
 
def retrieve(query, k=None, collection=None): 
    k = k or settings.top_k 
    collection = collection or get_collection() 
    n = min(k, collection.count()) 
    if n == 0: 
        return [] 
    res = collection.query(query_embeddings=[embed_query(query)], n_results=n) 
    docs = res["documents"][0] 
    metas = res["metadatas"][0] 
    dists = res["distances"][0] 
    out = [] 
    for text, meta, dist in zip(docs, metas, dists): 
        out.append({"text": text, "source": meta["source"], "chunk_id": meta["chunk_id"], "score": round(1 - dist, 4)}) 
    return out 
