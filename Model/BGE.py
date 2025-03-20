from typing import List, Tuple

class bge_embedder():
    def __init__(self, path: str = '/home/yujie/models/bge-m3') -> None:
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(model_name_or_path = path, use_fp16=False)

    def get_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        embeddings = self.model.encode(texts, batch_size=1, max_length=8192)['dense_vecs']
        return embeddings
