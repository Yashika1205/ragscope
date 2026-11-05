from src.chunker import chunk_by_sentences, chunk_text 
from src.generator import generate 
from src.retriever import retrieve 
from src.store import add_chunks, reset_collection 
 
 
def build_chunks(docs, strategy="fixed", chunk_size=500, overlap=50): 
    chunks = [] 
    for doc in docs: 
        if strategy == "sentence": 
            pieces = chunk_by_sentences(doc["text"], chunk_size) 
        else: 
            pieces = chunk_text(doc["text"], chunk_size, overlap) 
        for i, piece in enumerate(pieces): 
            chunks.append({"source": doc["source"], "chunk_id": i, "text": piece}) 
    return chunks 
 
 
class RAGPipeline: 
    def __init__(self, strategy="fixed", chunk_size=500, overlap=50, top_k=4, collection_name="ragscope"): 
        self.strategy = strategy 
        self.chunk_size = chunk_size 
        self.overlap = overlap 
        self.top_k = top_k 
        self.collection_name = collection_name 
        self.collection = None 
        self.num_chunks = 0 
 
    def index(self, docs): 
        chunks = build_chunks(docs, self.strategy, self.chunk_size, self.overlap) 
        self.collection = reset_collection(self.collection_name) 
        self.num_chunks = add_chunks(chunks, self.collection) 
        return self.num_chunks 
 
    def ask(self, question): 
        if self.collection is None: 
            raise RuntimeError("Call index(docs) before ask().") 
        contexts = retrieve(question, self.top_k, self.collection) 
        answer = generate(question, contexts) 
        return {"question": question, "answer": answer, "contexts": contexts} 
