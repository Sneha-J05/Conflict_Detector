"""
Step 7: RAG + KG Enhanced Conflict Explanation
Generates human-readable explanations grounded in both KG context and RAG context.
"""

import os
import json

from rule_extractor import llm_call, _log_llm_call
from rag_retriever import RAGRetriever

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXPLAIN_SYSTEM = (
    "You are a senior legal policy analyst. Explain policy conflicts clearly "
    "for non-lawyers. Use the provided legal context and concept relationships "
    "to ground your explanation. Be precise and cite the articles."
)

EXPLAIN_USER_TEMPLATE = """CONFLICT ID: {conflict_id}
CONFLICT TYPE: {conflict_type}
SEVERITY SCORE: {composite_score}/1.0
PRECEDENCE: {precedence}

Rule 1 ({rule_1_id} - {source_1}):
{rule_1_raw_text}

Rule 2 ({rule_2_id} - {source_2}):
{rule_2_raw_text}

SHARED LEGAL CONCEPTS (from Knowledge Graph):
{shared_concepts}

RELATED RULES (KG neighbors):
{neighboring_rule_texts}

RETRIEVED LEGAL CONTEXT (RAG):
{rag_context_block}

Explain:
1. What each rule requires (1-2 sentences each)
2. Why they conflict and in what real-world scenario
3. Which rule takes precedence and why (cite lex specialis if relevant)
4. Practical compliance implication

Under 200 words total."""


class ConflictExplainer:
    """Generate LLM explanations for detected conflicts using KG + RAG context."""

    def __init__(self, kg, backend: str = "ollama"):
        """
        Args:
            kg: PolicyKnowledgeGraph instance (with conflict edges).
            backend: LLM backend ('ollama' or 'huggingface').
        """
        self.kg = kg
        self.backend = backend
        self.retriever = RAGRetriever()

    # ------------------------------------------------------------------
    # Context retrieval
    # ------------------------------------------------------------------
    def _get_kg_context(self, rule_id: str) -> list[str]:
        """Get neighbouring rule texts from KG."""
        neighbors = self.kg.get_neighbors(rule_id)
        texts = []
        for nid, data in neighbors:
            ntype = data.get("node_type",
                             self.kg.G.nodes.get(nid, {}).get("node_type", ""))
            if ntype == "rule":
                node_data = self.kg.G.nodes.get(nid, {})
                summary = (f"{nid}: {node_data.get('entity','')} "
                           f"{node_data.get('action','')} "
                           f"({node_data.get('modality','')})")
                texts.append(summary)
        return texts[:5]

    def _get_rag_context(self, rule_1: dict, rule_2: dict) -> list[str]:
        """Retrieve legal context chunks from ChromaDB."""
        try:
            chunks = self.retriever.retrieve_legal_context(rule_1, rule_2, top_k=3)
            return [c.get("text", "") for c in chunks]
        except Exception as e:
            print(f"  [EXPLAINER] RAG retrieval failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Explanation generation
    # ------------------------------------------------------------------
    def explain_conflict(self, conflict: dict) -> str:
        """Generate an explanation for a single conflict."""
        r1 = conflict.get("rule_1", {})
        r2 = conflict.get("rule_2", {})

        # KG context
        r1_id = r1.get("rule_id", "")
        r2_id = r2.get("rule_id", "")
        kg_neighbors_1 = self._get_kg_context(r1_id)
        kg_neighbors_2 = self._get_kg_context(r2_id)
        all_neighbors = list(set(kg_neighbors_1 + kg_neighbors_2))

        # RAG context
        rag_chunks = self._get_rag_context(r1, r2)
        rag_block = "\n".join(
            f"[Context {i+1}]: {chunk}" for i, chunk in enumerate(rag_chunks)
        ) or "[No additional context retrieved]"

        user_prompt = EXPLAIN_USER_TEMPLATE.format(
            conflict_id=conflict.get("conflict_id", ""),
            conflict_type=conflict.get("type", ""),
            composite_score=conflict.get("composite_score", 0),
            precedence=conflict.get("precedence", "ambiguous"),
            rule_1_id=r1_id,
            source_1=r1.get("source", ""),
            rule_1_raw_text=r1.get("raw_text", "(no raw text)")[:500],
            rule_2_id=r2_id,
            source_2=r2.get("source", ""),
            rule_2_raw_text=r2.get("raw_text", "(no raw text)")[:500],
            shared_concepts=", ".join(conflict.get("shared_concepts", [])) or "none",
            neighboring_rule_texts="\n".join(all_neighbors) or "none",
            rag_context_block=rag_block,
        )

        for attempt in range(2):
            try:
                explanation = llm_call(EXPLAIN_SYSTEM, user_prompt, self.backend)
                _log_llm_call("explanation", user_prompt[:300], explanation[:300])
                return explanation.strip()
            except Exception as e:
                print(f"  [EXPLAINER] Attempt {attempt+1} failed "
                      f"({conflict.get('conflict_id','')}): {e}")

        return "(Explanation generation failed)"

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def explain_all(self, conflicts_path: str | None = None):
        """Generate explanations for all conflicts and save."""
        if conflicts_path is None:
            conflicts_path = os.path.join(DATA_DIR, "conflicts_raw.json")

        with open(conflicts_path, "r", encoding="utf-8") as f:
            conflicts = json.load(f)

        print(f"[EXPLAINER] Generating explanations for {len(conflicts)} conflicts …")

        for idx, conflict in enumerate(conflicts, 1):
            print(f"[EXPLAINER] ({idx}/{len(conflicts)}) "
                  f"{conflict.get('conflict_id', '')} …")

            # Get explanation
            explanation = self.explain_conflict(conflict)
            conflict["explanation"] = explanation

            # Attach RAG context text
            r1 = conflict.get("rule_1", {})
            r2 = conflict.get("rule_2", {})
            rag_chunks = self._get_rag_context(r1, r2)
            conflict["rag_context"] = [c for c in rag_chunks]

            # Attach KG neighbor IDs
            r1_id = r1.get("rule_id", "")
            r2_id = r2.get("rule_id", "")
            neighbors = set()
            for nid, _ in self.kg.get_neighbors(r1_id):
                if self.kg.G.nodes.get(nid, {}).get("node_type") == "rule":
                    neighbors.add(nid)
            for nid, _ in self.kg.get_neighbors(r2_id):
                if self.kg.G.nodes.get(nid, {}).get("node_type") == "rule":
                    neighbors.add(nid)
            conflict["kg_neighbors"] = list(neighbors)

        # Save final output
        out_path = os.path.join(DATA_DIR, "conflicts_with_explanations.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(conflicts, f, indent=2, ensure_ascii=False)

        print(f"[EXPLAINER] Done → {out_path}")

    def run(self):
        self.explain_all()
