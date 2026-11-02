import re 
 
from src.config import settings 
 
 
def chunk_text(text, chunk_size=None, overlap=None): 
    chunk_size = chunk_size or settings.chunk_size 
    overlap = settings.chunk_overlap if overlap is None else overlap 
    if overlap >= chunk_size: 
        raise ValueError("overlap must be smaller than chunk_size") 
    step = chunk_size - overlap 
    chunks = [] 
    for start in range(0, len(text), step): 
        chunk = text[start:start + chunk_size].strip() 
        if chunk: 
            chunks.append(chunk) 
        if start + chunk_size >= len(text): 
            break 
    return chunks 
 
 
def chunk_documents(docs, chunk_size=None, overlap=None): 
    out = [] 
    for doc in docs: 
        for i, chunk in enumerate(chunk_text(doc["text"], chunk_size, overlap)): 
            out.append({"source": doc["source"], "chunk_id": i, "text": chunk}) 
    return out 
 
 
def split_sentences(text): 
    sentences = re.findall(r"[^.!?]+[.!?]*", text) 
    return [s.strip() for s in sentences if s.strip()] 
 
 
def chunk_by_sentences(text, max_chars=None): 
    max_chars = max_chars or settings.chunk_size 
    chunks, current = [], "" 
    for sentence in split_sentences(text): 
        if current and len(current) + len(sentence) + 1 > max_chars: 
            chunks.append(current) 
            current = sentence 
        else: 
            current = (current + " " + sentence).strip() 
    if current: 
        chunks.append(current) 
    return chunks 
