import numpy as np 
from src.chunker import split_sentences 
from src.embedder import embed_texts 
 
import json 
 
from src.config import DATA_DIR 
from src.retriever import retrieve 
 
 
def load_questions(path=DATA_DIR / "eval_questions.json"): 
    with open(path, encoding="utf-8") as f: 
        return json.load(f) 
 
 
def is_relevant(text, keywords): 
    text = text.lower() 
    return any(k.lower() in text for k in keywords) 
 
 
def retrieval_metrics(contexts, keywords): 
    flags = [is_relevant(c["text"], keywords) for c in contexts] 
    hit = float(any(flags)) 
    mrr = 0.0 
    for i, flag in enumerate(flags): 
        if flag: 
            mrr = 1 / (i + 1) 
            break 
    precision = sum(flags) / len(flags) if flags else 0.0 
    return {"hit_rate": hit, "mrr": mrr, "context_precision": precision} 
 
 
def evaluate_retrieval(pipeline, questions=None): 
    questions = questions or load_questions() 
    rows = [] 
    for q in questions: 
        contexts = retrieve(q["question"], pipeline.top_k, pipeline.collection) 
        rows.append(retrieval_metrics(contexts, q["keywords"])) 
    return {key: round(sum(r[key] for r in rows) / len(rows), 4) for key in ("hit_rate", "mrr", "context_precision")} 
 
 
def _cos(a, b): 
    a, b = np.array(a), np.array(b) 
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)) 
 
 
def faithfulness_proxy(answer, contexts): 
    sentences = split_sentences(answer) 
    if not sentences or not contexts: 
        return 0.0 
    vecs = embed_texts(sentences + [c["text"] for c in contexts]) 
    s_vecs = vecs[:len(sentences)] 
    c_vecs = vecs[len(sentences):] 
    scores = [max(_cos(s, c) for c in c_vecs) for s in s_vecs] 
    return round(float(np.mean(scores)), 4) 
