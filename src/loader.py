from pathlib import Path 
from pypdf import PdfReader 
from src.config import DATA_DIR 
 
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"} 
 
 
def load_txt(path): 
    return Path(path).read_text(encoding="utf-8", errors="ignore") 
 
 
def load_pdf(path): 
    reader = PdfReader(str(path)) 
    return "\n".join(page.extract_text() or "" for page in reader.pages) 
 
 
def load_document(path): 
    path = Path(path) 
    ext = path.suffix.lower() 
    if ext == ".pdf": 
        text = load_pdf(path) 
    elif ext in {".txt", ".md"}: 
        text = load_txt(path) 
    else: 
        raise ValueError(f"Unsupported file type: {ext}") 
    return {"source": path.name, "text": text} 
 
 
def load_directory(directory=DATA_DIR): 
    docs = [] 
    for path in sorted(Path(directory).iterdir()): 
        if path.suffix.lower() in SUPPORTED_EXTENSIONS: 
            docs.append(load_document(path)) 
    return docs 
