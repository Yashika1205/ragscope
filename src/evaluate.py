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
