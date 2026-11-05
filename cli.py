import argparse 
 
from src.loader import load_directory 
from src.pipeline import RAGPipeline 
 
 
def main(): 
    parser = argparse.ArgumentParser(description="RAGScope CLI") 
    parser.add_argument("question") 
    parser.add_argument("--strategy", default="fixed", choices=["fixed", "sentence"]) 
    parser.add_argument("--chunk-size", type=int, default=500) 
    parser.add_argument("--top-k", type=int, default=4) 
    args = parser.parse_args() 
 
    pipe = RAGPipeline(args.strategy, args.chunk_size, args.chunk_size // 10, args.top_k) 
    pipe.index(load_directory()) 
    out = pipe.ask(args.question) 
    print("Answer:", out["answer"]) 
    print("Sources:") 
    for c in out["contexts"]: 
        print(f"  - {c['source']} #{c['chunk_id']} (score {c['score']})") 
 
 
if __name__ == "__main__": 
    main() 
