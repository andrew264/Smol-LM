from typing import List

import torch.nn.functional as F
try:
    from langchain_core.embeddings import Embeddings
    from sentence_transformers import SentenceTransformer
except ImportError:
    Embeddings = object
    SentenceTransformer = None


class HFNomicEmbeddings(Embeddings):
    def __init__(self, size: int = 768, path: str = "nomic-ai/nomic-embed-text-v1.5", device: str = "cpu"):
        if SentenceTransformer is None:
            raise ImportError("Please install sentence-transformers")
        self.model = SentenceTransformer(path, device=device, trust_remote_code=True)
        self.size = size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        task_type = "search_document: "
        texts = [task_type + text for text in texts]
        return self.embed(texts)

    def embed_query(self, text: str) -> List[float]:
        task_type = "search_query: "
        text = task_type + text
        return self.embed([text])[0]

    def embed(self, text: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(text, convert_to_tensor=True)
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = embeddings[:, :self.size]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()
