from pathlib import Path

def load_documents(directory:str):
    doc = []

    for path in Path(directory).glob('*.txt'):

        text = path.read_text(encoding='utf-8')

        doc.append({
            "document_id":path.stem,
            "source":path.name,
            "text":text
        })
    
    return doc