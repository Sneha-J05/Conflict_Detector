"""
Step 6: RAG + KG Enhanced Conflict Detection
Merges KG structural candidates and RAG semantic candidates,
classifies each pair via Mistral-7B, and enriches the KG.
"""

import os
import json
import datetime

from rule_extractor import llm_call, _extract_json, _log_llm_call
from weight_scorer import WeightScorer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = (
    "You are a legal conflict analyst. Classify the relationship between "
    "two policy rules. Return ONLY valid JSON. No explanation outside JSON."
)

CLASSIFY_USER_TEMPLATE = """Rule 1 ({rule_1_id} - {source_1}):
Entity: {entity_1}
Action: {action_1}
Condition: {condition_1}
Modality: {modality_1}
Lex Specialis: {lex_specialis_1}

Rule 2 ({rule_2_id} - {source_2}):
Entity: {entity_2}
Action: {action_2}
Condition: {condition_2}
Modality: {modality_2}
Lex Specialis: {lex_specialis_2}

Shared Concepts: {shared_concepts}

Classify their relationship. Return JSON:
{{
  "relationship": "<direct_conflict|conditional_conflict|logical_inconsistency|redundancy|exception|no_conflict>",
  "confidence": 0.0,
  "severity": "<direct_conflict|conditional_conflict|logical_inconsistency|redundancy|ambiguity>",
  "reasoning": "one sentence reason",
  "precedence": "<rule_1|rule_2|ambiguous>"
}}"""


class ConflictDetector:
    """Detect conflicts between GDPR and ePrivacy rules using KG + RAG + LLM."""

    def __init__(self, kg, retriever, backend: str = "ollama"):
        """
        Args:
            kg: PolicyKnowledgeGraph instance (already built).
            retriever: RAGRetriever instance.
            backend: LLM backend ('ollama' or 'huggingface').
        """
        self.kg = kg
        self.retriever = retriever
        self.backend = backend
        self.scorer = WeightScorer(backend)
        self.conflicts: list[dict] = []
        self._rules_by_id: dict[str, dict] = {}
        self._conflict_counter = 0

    # ------------------------------------------------------------------
    # Load rules index
    # ------------------------------------------------------------------
    def _load_rules_index(self):
        path = os.path.join(DATA_DIR, "rules.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("rules", []):
            rid = r.get("rule_id", "")
            if rid:
                self._rules_by_id[rid] = r

    # ------------------------------------------------------------------
    # 6A  Candidate generation
    # ------------------------------------------------------------------
    def _generate_candidates(self) -> list[tuple[str, str, list[str]]]:
        """
        Merge KG-based and RAG-based candidates.
        Returns list of (rule_1_id, rule_2_id, shared_concepts).
        """
        seen = set()
        candidates = []

        # Source 1: KG cross-source pairs per concept
        for concept_label in self.kg.get_all_concept_labels():
            pairs = self.kg.get_cross_source_pairs(concept_label)
            for r1, r2 in pairs:
                key = tuple(sorted([r1, r2]))
                if key not in seen:
                    seen.add(key)
                    candidates.append((r1, r2, [concept_label]))
                else:
                    # Append concept to existing candidate
                    for i, (c1, c2, concepts) in enumerate(candidates):
                        if tuple(sorted([c1, c2])) == key:
                            if concept_label not in concepts:
                                candidates[i] = (c1, c2, concepts + [concept_label])
                            break

        # Source 2: RAG semantic similarity
        for rid, rule in self._rules_by_id.items():
            similar = self.retriever.retrieve_similar_rules(rule, top_k=5)
            for s in similar:
                sid = s.get("rule_id", "")
                key = tuple(sorted([rid, sid]))
                if key not in seen and sid in self._rules_by_id:
                    seen.add(key)
                    candidates.append((rid, sid, []))

        print(f"[DETECTOR] Generated {len(candidates)} candidate pairs "
              f"(KG + RAG, deduplicated)")
        return candidates

    # ------------------------------------------------------------------
    # 6B  LLM classification
    # ------------------------------------------------------------------
    def _classify_pair(self, r1_id: str, r2_id: str,
                       shared_concepts: list[str]) -> dict | None:
        """Classify a single pair via Mistral-7B."""
        r1 = self._rules_by_id.get(r1_id, {})
        r2 = self._rules_by_id.get(r2_id, {})

        user_prompt = CLASSIFY_USER_TEMPLATE.format(
            rule_1_id=r1_id,
            source_1=r1.get("source", ""),
            entity_1=r1.get("entity", ""),
            action_1=r1.get("action", ""),
            condition_1=r1.get("condition", ""),
            modality_1=r1.get("modality", ""),
            lex_specialis_1=r1.get("weights", {}).get("lex_specialis", False),
            rule_2_id=r2_id,
            source_2=r2.get("source", ""),
            entity_2=r2.get("entity", ""),
            action_2=r2.get("action", ""),
            condition_2=r2.get("condition", ""),
            modality_2=r2.get("modality", ""),
            lex_specialis_2=r2.get("weights", {}).get("lex_specialis", False),
            shared_concepts=", ".join(shared_concepts) if shared_concepts else "none",
        )

        for attempt in range(2):
            try:
                raw = llm_call(CLASSIFY_SYSTEM, user_prompt, self.backend)
                _log_llm_call("conflict_classification", user_prompt[:300], raw[:300])
                result = _extract_json(raw)
                return result
            except Exception as e:
                print(f"  [DETECTOR] Classify attempt {attempt+1} failed "
                      f"({r1_id} vs {r2_id}): {e}")

        return None

    # ------------------------------------------------------------------
    # 6C  Post-classification processing
    # ------------------------------------------------------------------
    def _process_classification(self, r1_id: str, r2_id: str,
                                 shared_concepts: list[str],
                                 classification: dict):
        """Add conflict to list and KG edge."""
        relationship = classification.get("relationship", "no_conflict")
        confidence = classification.get("confidence", 0.0)

        # Filter
        if relationship == "no_conflict" or confidence < 0.6:
            return

        self._conflict_counter += 1
        conflict_id = f"C{self._conflict_counter:03d}"

        # Severity weight
        severity_type = classification.get("severity", relationship)
        sev_weight = self.scorer.severity_weight(severity_type)

        r1 = self._rules_by_id.get(r1_id, {})
        r2 = self._rules_by_id.get(r2_id, {})
        w1 = r1.get("weights", {})
        w2 = r2.get("weights", {})

        # Composite score uses the max source/article weights of the pair
        source_w = max(w1.get("source", 0.5), w2.get("source", 0.5))
        article_w = max(w1.get("article", 0.5), w2.get("article", 0.5))
        composite = source_w * article_w * sev_weight

        # Add edge to KG
        self.kg.add_conflict_edge(
            r1_id, r2_id,
            conflict_type=relationship,
            confidence=confidence,
            composite_score=composite,
            precedence=classification.get("precedence", "ambiguous"),
            reasoning=classification.get("reasoning", ""),
        )

        conflict = {
            "conflict_id": conflict_id,
            "type": relationship,
            "rule_1": r1,
            "rule_2": r2,
            "shared_concepts": shared_concepts,
            "composite_score": round(composite, 4),
            "confidence": confidence,
            "precedence": classification.get("precedence", "ambiguous"),
            "reasoning": classification.get("reasoning", ""),
            "rag_context_ids": [],
            "explanation": "",
        }
        self.conflicts.append(conflict)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_all(self):
        """Full detection pipeline: candidates → classify → enrich KG."""
        self._load_rules_index()
        candidates = self._generate_candidates()

        for idx, (r1, r2, concepts) in enumerate(candidates, 1):
            print(f"[DETECTOR] ({idx}/{len(candidates)}) {r1} vs {r2} …")
            result = self._classify_pair(r1, r2, concepts)
            if result:
                self._process_classification(r1, r2, concepts, result)

        self._save_conflicts()
        print(f"[DETECTOR] Done — {len(self.conflicts)} conflicts detected")

    def _save_conflicts(self):
        path = os.path.join(DATA_DIR, "conflicts_raw.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.conflicts, f, indent=2, ensure_ascii=False)
        print(f"[DETECTOR] Saved → {path}")

    def get_conflicts_by_severity(self) -> list[dict]:
        return sorted(self.conflicts,
                       key=lambda c: c.get("composite_score", 0),
                       reverse=True)

    def get_conflicts_by_concept(self, concept_label: str) -> list[dict]:
        return [c for c in self.conflicts
                if concept_label in c.get("shared_concepts", [])]

    def get_conflicts_by_article(self, article_id: str) -> list[dict]:
        results = []
        for c in self.conflicts:
            r1_art = f"{c['rule_1'].get('source','')}_Art{c['rule_1'].get('article_number','')}"
            r2_art = f"{c['rule_2'].get('source','')}_Art{c['rule_2'].get('article_number','')}"
            if article_id in (r1_art, r2_art):
                results.append(c)
        return results

    def get_conflict_summary(self) -> dict:
        by_type: dict[str, int] = {}
        scores = []
        for c in self.conflicts:
            t = c.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
            scores.append(c.get("composite_score", 0))
        return {
            "total": len(self.conflicts),
            "by_type": by_type,
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
        }
