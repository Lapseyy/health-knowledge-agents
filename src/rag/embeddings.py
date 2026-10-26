"""Local embedding model (ONNX via fastembed): no API key, no rate limits.

The model downloads once (~70 MB for bge-small) into EMBEDDING_CACHE_DIR and
then runs on CPU. The Docker build pre-downloads it so Cloud Run instances
never fetch it at startup.
"""

from langchain_core.embeddings import Embeddings

# BGE models retrieve better when short queries carry this instruction;
# passages are embedded as-is (per the BGE v1.5 model card).
QUERY_INSTRUCTIONS = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str, cache_dir: str | None = None, batch_size: int = 32):
        from fastembed import TextEmbedding  # imported lazily so tests can run without the model

        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.query_instruction = QUERY_INSTRUCTIONS.get(model_name, "")
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.embed([self.query_instruction + text]))).tolist()
