from sentence_transformers import SentenceTransformer 
from src.config import settings 
 
_model = None 
 
 
def get_model(): 
    global _model 
    if _model is None: 
        _model = SentenceTransformer(settings.embedding_model) 
    return _model 
 
 
def embed_texts(texts): 
    return get_model().encode(texts, show_progress_bar=False).tolist() 
 
 
def embed_query(query): 
    return embed_texts([query])[0] 
