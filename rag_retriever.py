"""
Step 5B: RAG Retriever
Retrieves semantically similar rules and legal context from ChromaDB.
"""

import os
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")


class RAGRetriever:
    """Semantic retrieval over ChromaDB collections."""

    def __init__(self):
        import chromadb
        from chromadb.utils import embedding_functions

        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.rules_col = self.client.get_or_create_collection(
            name="policy_rules",
            embedding_function=self.embed_fn,
        )
        self.articles_col = self.client.get_or_create_collection(
            name="policy_articles",
            embedding_function=self.embed_fn,
        )

    # ------------------------------------------------------------------
    # 5C-1  Similar rules (cross-source)
    # ------------------------------------------------------------------
    def retrieve_similar_rules(self, rule: dict,
                               top_k: int = 5) -> list[dict]:
        """
        Retrieve the top_k most semantically similar rules from a
        different source than the query rule.
        """
        query_text = " ".join(filter(None, [
            rule.get("entity", ""),
            rule.get("action", ""),
            rule.get("condition", ""),
            rule.get("modality", ""),
        ]))
        rule_source = rule.get("source", "")

        results = self.rules_col.query(
            query_texts=[query_text],
            n_results=top_k * 3,  # fetch extra, filter below
        )

        similar = []
        if not results or not results.get("ids"):
            return similar

        for i, rid in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            if meta.get("source", "") == rule_source:
                continue  # skip same-source
            similar.append({
                "rule_id": rid,
                "distance": (results["distances"][0][i]
                             if results.get("distances") else None),
                "metadata": meta,
                "document": (results["documents"][0][i]
                             if results.get("documents") else ""),
            })
            if len(similar) >= top_k:
                break

        return similar

    # ------------------------------------------------------------------
    # 5C-2  Legal context (article chunks)
    # ------------------------------------------------------------------
    def retrieve_legal_context(self, rule_1: dict, rule_2: dict,
                               top_k: int = 3) -> list[dict]:
        """
        Retrieve the top_k most relevant article chunks as grounding
        context for a pair of potentially conflicting rules.
        """
        combined_text = " ".join(filter(None, [
            rule_1.get("raw_text", ""),
            rule_2.get("raw_text", ""),
            rule_1.get("entity", ""), rule_1.get("action", ""),
            rule_2.get("entity", ""), rule_2.get("action", ""),
        ]))

        if not combined_text.strip():
            return []

        results = self.articles_col.query(
            query_texts=[combined_text],
            n_results=top_k,
        )

        contexts = []
        if not results or not results.get("ids"):
            return contexts

        for i, cid in enumerate(results["ids"][0]):
            contexts.append({
                "chunk_id": cid,
                "text": (results["documents"][0][i]
                         if results.get("documents") else ""),
                "metadata": (results["metadatas"][0][i]
                             if results.get("metadatas") else {}),
                "distance": (results["distances"][0][i]
                             if results.get("distances") else None),
            })

        return contexts


if __name__ == "__main__":
    print("[RAG RETRIEVER] Module ready. Use via RAGRetriever class.")
