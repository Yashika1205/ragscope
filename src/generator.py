import os 
 
from src.config import settings 
 
SYSTEM_PROMPT = "You answer questions using only the provided context. If the context does not contain the answer, say you do not know. Keep the answer short." 
 
 
def build_prompt(question, contexts): 
    parts = [] 
    for i, c in enumerate(contexts): 
        parts.append(f"[{i + 1}] ({c['source']}) {c['text']}") 
    context_text = "\n\n".join(parts) 
    return f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:" 
 
 
def generate(question, contexts): 
    if not contexts: 
        return "I do not know." 
    if not settings.llm_api_key: 
        return contexts[0]["text"] 
    from openai import OpenAI 
 
    client = OpenAI(api_key=settings.llm_api_key, base_url=os.getenv("LLM_BASE_URL") or None) 
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_prompt(question, contexts)}] 
    resp = client.chat.completions.create(model=settings.llm_model, temperature=0, messages=messages) 
    return resp.choices[0].message.content.strip() 
