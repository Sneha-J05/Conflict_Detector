"""
Step 5A: Policy Embedder
Embeds rules and article chunks into ChromaDB using sentence-transformers.
Runs PARALLEL to the Knowledge Graph.
"""

import os
import json
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")


class PolicyEmbedder:
    """Embed policy rules and article chunks into ChromaDB."""

    def __init__(self):
        import chromadb
        from chromadb.utils import embedding_functions

        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    def _sanitize_metadata(self, metadata: dict) -> dict:
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                sanitized[key] = json.dumps(value)
            elif isinstance(value, dict):
                sanitized[key] = json.dumps(value)
            elif value is None:
                sanitized[key] = ""
            elif not isinstance(value, (str, int, float, bool)):
                sanitized[key] = str(value)
            else:
                sanitized[key] = value
        return sanitized

    # ------------------------------------------------------------------
    # 5A  Embed rules
    # ------------------------------------------------------------------
    def embed_rules(self, rules_path: str | None = None):
        """Embed each rule as '{entity} {action} {condition} {modality}'."""
        if rules_path is None:
            rules_path = os.path.join(DATA_DIR, "rules.json")

        with open(rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rules = data.get("rules", [])
        if not rules:
            print("[EMBEDDER] No rules to embed.")
            return

        collection = self.client.get_or_create_collection(
            name="policy_rules",
            embedding_function=self.embed_fn,
        )

        ids, documents, metadatas = [], [], []
        for r in rules:
            rid = r.get("rule_id", "")
            if not rid:
                continue
            text = " ".join(filter(None, [
                r.get("entity", ""),
                r.get("action", ""),
                r.get("condition", ""),
                r.get("modality", ""),
            ]))
            weights = r.get("weights", {})
            ids.append(rid)
            documents.append(text)
            metadata = {
                "rule_id": rid,
                "source": r.get("source", ""),
                "article_number": r.get("article_number", 0),
                "modality": r.get("modality", ""),
                "article_weight": weights.get("article", 0.5),
                "source_weight": weights.get("source", 0.5),
                "lex_specialis": str(weights.get("lex_specialis", False)),
            }
            metadata = self._sanitize_metadata(metadata)
            metadatas.append(metadata)

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

        print(f"[EMBEDDER] Embedded {len(ids)} rules -> ChromaDB 'policy_rules'")

    # ------------------------------------------------------------------
    # 5B  Embed article chunks
    # ------------------------------------------------------------------
    def embed_articles(self, corpus_path: str | None = None):
        """Chunk each article (200 tokens, 50 overlap) and embed."""
        if corpus_path is None:
            corpus_path = os.path.join(DATA_DIR, "corpus.json")

        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        collection = self.client.get_or_create_collection(
            name="policy_articles",
            embedding_function=self.embed_fn,
        )

        ids, documents, metadatas = [], [], []
        for article in corpus:
            chunks = self._chunk_text(article["text"],
                                       chunk_size=200, overlap=50)
            for ci, chunk in enumerate(chunks):
                doc_id = f"{article['id']}_chunk_{ci}"
                ids.append(doc_id)
                documents.append(chunk)
                metadata = {
                    "article_id": article["id"],
                    "source": article["source"],
                    "article_number": article["article"],
                    "chunk_index": ci,
                }
                metadata = self._sanitize_metadata(metadata)
                metadatas.append(metadata)

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

        print(f"[EMBEDDER] Embedded {len(ids)} article chunks "
              f"-> ChromaDB 'policy_articles'")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 200,
                    overlap: int = 50) -> list[str]:
        """Split text into overlapping token-based chunks."""
        tokens = text.split()
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunks.append(" ".join(tokens[start:end]))
            start += chunk_size - overlap
        return chunks

    def run(self):
        """Embed both rules and article chunks."""
        self.embed_rules()
        self.embed_articles()


if __name__ == "__main__":
    PolicyEmbedder().run()
