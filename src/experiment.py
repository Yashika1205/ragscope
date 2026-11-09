import itertools 
 
import pandas as pd 
 
from src.config import RESULTS_DIR 
from src.evaluate import evaluate_pipeline, load_questions 
from src.loader import load_directory 
from src.pipeline import RAGPipeline 
 
GRID = {"strategy": ["fixed", "sentence"], "chunk_size": [200, 400, 800], "top_k": [2, 4]} 
RESULTS_FILE = RESULTS_DIR / "experiment_results.csv" 
 
 
def run_experiments(grid=GRID, docs=None, questions=None): 
    docs = docs or load_directory() 
    questions = questions or load_questions() 
    rows = [] 
    for strategy, size, k in itertools.product(grid["strategy"], grid["chunk_size"], grid["top_k"]): 
        pipe = RAGPipeline(strategy, size, size // 4, k, collection_name="ragscope_exp") 
        n_chunks = pipe.index(docs) 
        metrics = evaluate_pipeline(pipe, questions) 
        rows.append({"strategy": strategy, "chunk_size": size, "top_k": k, "num_chunks": n_chunks, **metrics}) 
    return pd.DataFrame(rows) 
 
 
def save_results(df, path=RESULTS_FILE): 
    path.parent.mkdir(parents=True, exist_ok=True) 
    df.to_csv(path, index=False) 
    return path 
 
 
if __name__ == "__main__": 
    df = run_experiments() 
    path = save_results(df) 
    print(df.sort_values(["hit_rate", "mrr"], ascending=False).to_string(index=False)) 
    print(f"Saved {len(df)} rows to {path}") 
