from pathlib import Path
import json
import re

from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, index_path:str = r'data\bm25_index.json'):
        self.index_path = Path(index_path)

        self.document = []
        self.bm25 = None

        if self.index_path.exists():
            if self.index_path.stat().st_size > 0:
                self._load()


    def _tokenize(self, text:str)->list[str]:
        return re.findall(r'\b\w+\b',text.lower())

    def _load(self) -> None:
        try:
            with open(self.index_path,"r",encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(
                "Warning: BM25 index is empty or invalid. "
                "It will be rebuilt during indexing."
            )
            self.documents = []
            self.bm25 = None
            return
        self.documents = data.get("documents",[])
        if not self.documents:
            self.bm25 = None
            return
        tokenized_documents = [self._tokenize(doc["text"]) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_documents)

    def build(self, documents:list[dict])->list[str]:
        if not documents:
            raise ValueError(
                "Cannot build BM25 index from empty documents."
            )
        self.documents = documents
        tokenized_documents = [self._tokenize(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_documents)
        self.index_path.parent.mkdir(parents=True,exist_ok=True)
        with open(self.index_path,"w",encoding="utf-8") as f:
            doc_json={"documents": self.documents}
            json.dump(doc_json,f,ensure_ascii=False,indent=2)

    def search(self, query:str, top_k:int = 20)-> list[dict]:
        if self.bm25 is None:
            raise RuntimeError('BM25 index is not initialized')
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)),key=lambda i: scores[i],reverse=True)[:top_k]
        results=[]
        for rank, idx in enumerate(ranked_indices):
            results.append(
                {
                    **self.documents[idx],
                    "score": float(scores[idx]),
                    "rank": rank + 1,
                    "retrieval_type": "bm25",
                })
        return results


