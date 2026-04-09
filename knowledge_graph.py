"""
Step 4: Policy Knowledge Graph
Central KG module using NetworkX — nodes for articles, rules, concepts;
edges for CONTAINS, INVOLVES, REFERENCES, CONFLICTS_WITH, etc.
"""

import os
import json
import networkx as nx
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class PolicyKnowledgeGraph:
    """Directed, weighted, attributed Knowledge Graph for policy rules."""

    def __init__(self):
        self.G = nx.DiGraph()

    # ==================================================================
    # 4A  Node construction
    # ==================================================================
    def _add_article_nodes(self, articles: list[dict]):
        """Add ARTICLE-type nodes."""
        for art in articles:
            nid = art["article_id"]
            self.G.add_node(nid, **{
                "node_type": "article",
                "article_id": nid,
                "source": art.get("source", ""),
                "article_number": art.get("article_number", 0),
                "article_type": art.get("article_type", ""),
                "article_weight": (art.get("rules", [{}])[0]
                                   .get("weights", {})
                                   .get("article", 0.5)
                                   if art.get("rules") else 0.5),
                "text_summary": art.get("rules", [{}])[0]
                                .get("raw_text", "")[:200]
                                if art.get("rules") else "",
            })

    def _add_rule_nodes(self, rules: list[dict]):
        """Add RULE-type nodes."""
        for r in rules:
            rid = r.get("rule_id", "")
            if not rid:
                continue
            weights = r.get("weights", {})
            self.G.add_node(rid, **{
                "node_type": "rule",
                "rule_id": rid,
                "entity": r.get("entity", ""),
                "action": r.get("action", ""),
                "condition": r.get("condition", ""),
                "modality": r.get("modality", ""),
                "scope": r.get("scope", ""),
                "source": r.get("source", ""),
                "article_number": r.get("article_number", 0),
                "source_weight": weights.get("source", 0.5),
                "article_weight": weights.get("article", 0.5),
                "composite": weights.get("composite"),
                "lex_specialis": weights.get("lex_specialis", False),
            })

    def _extract_and_add_concept_nodes(self, rules: list[dict]):
        """Extract concepts via spaCy NER + noun phrases, deduplicate with rapidfuzz."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            print("[KG] spaCy model not found — falling back to simple noun extraction")
            nlp = None

        raw_concepts: list[str] = []
        rule_concept_map: dict[str, list[str]] = {}  # rule_id → [concepts]

        for r in rules:
            rid = r.get("rule_id", "")
            text = " ".join(filter(None, [
                r.get("entity", ""),
                r.get("action", ""),
                r.get("condition", ""),
                r.get("scope", ""),
                r.get("raw_text", ""),
            ]))

            concepts_for_rule: list[str] = []
            if nlp:
                doc = nlp(text)
                for ent in doc.ents:
                    concepts_for_rule.append(ent.text.lower().strip())
                for chunk in doc.noun_chunks:
                    concepts_for_rule.append(chunk.text.lower().strip())
            else:
                # Fallback: split on spaces, keep multi-word phrases from entity/action
                for field in ["entity", "action", "condition"]:
                    val = r.get(field, "")
                    if val:
                        concepts_for_rule.append(val.lower().strip())

            raw_concepts.extend(concepts_for_rule)
            rule_concept_map[rid] = concepts_for_rule

        # Deduplicate with rapidfuzz
        unique = self._deduplicate_concepts(raw_concepts)

        # Build canonical mapping  raw → canonical
        canonical_map = {}
        try:
            from rapidfuzz import fuzz
            for raw in set(raw_concepts):
                best, best_score = raw, 0
                for canon in unique:
                    score = fuzz.ratio(raw, canon)
                    if score > best_score:
                        best, best_score = canon, score
                canonical_map[raw] = best
        except ImportError:
            for raw in set(raw_concepts):
                canonical_map[raw] = raw

        # Count frequencies
        freq = Counter()
        for concepts in rule_concept_map.values():
            for c in concepts:
                freq[canonical_map.get(c, c)] += 1

        # Add concept nodes
        for label in unique:
            cid = "concept_" + label.replace(" ", "_")
            self.G.add_node(cid, **{
                "node_type": "concept",
                "concept_id": cid,
                "label": label,
                "frequency": freq.get(label, 0),
            })

        # Store mapping for edge creation
        self._rule_concept_map = {
            rid: list({canonical_map.get(c, c) for c in concepts})
            for rid, concepts in rule_concept_map.items()
        }

    @staticmethod
    def _deduplicate_concepts(concepts: list[str], threshold: int = 85) -> list[str]:
        """Fuzzy-deduplicate concept labels."""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return list(set(concepts))

        unique: list[str] = []
        for c in sorted(set(concepts)):
            if not c:
                continue
            is_dup = False
            for u in unique:
                if fuzz.ratio(c, u) >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(c)
        return unique

    # ==================================================================
    # 4B  Edge construction
    # ==================================================================
    def _add_contains_edges(self, articles: list[dict]):
        """ARTICLE → RULE  (CONTAINS)."""
        for art in articles:
            aid = art["article_id"]
            art_weight = (art.get("rules", [{}])[0]
                          .get("weights", {})
                          .get("article", 0.5)
                          if art.get("rules") else 0.5)
            for r in art.get("rules", []):
                rid = r.get("rule_id", "")
                if rid and self.G.has_node(aid) and self.G.has_node(rid):
                    self.G.add_edge(aid, rid,
                                    edge_type="CONTAINS",
                                    weight=art_weight)

    def _add_involves_edges(self):
        """RULE → CONCEPT  (INVOLVES)."""
        for rid, concepts in self._rule_concept_map.items():
            for label in concepts:
                cid = "concept_" + label.replace(" ", "_")
                if self.G.has_node(rid) and self.G.has_node(cid):
                    self.G.add_edge(rid, cid,
                                    edge_type="INVOLVES",
                                    weight=1.0)

    def _add_reference_edges(self, rules: list[dict]):
        """ARTICLE → ARTICLE  (REFERENCES) from rule references field."""
        for r in rules:
            refs = r.get("references", [])
            if not isinstance(refs, list):
                refs = []
            source_art = f"{r.get('source','')}_Art{r.get('article_number','')}"
            for ref in refs:
                if not isinstance(ref, str):
                    continue
                # Normalise "GDPR_Art7" style
                ref = ref.replace(" ", "_")
                if self.G.has_node(source_art) and self.G.has_node(ref):
                    if not self.G.has_edge(source_art, ref):
                        self.G.add_edge(source_art, ref,
                                        edge_type="REFERENCES",
                                        weight=0.5)

    # ==================================================================
    # Conflict edges (called externally by ConflictDetector)
    # ==================================================================
    def add_conflict_edge(self, rule_1_id: str, rule_2_id: str,
                          conflict_type: str, confidence: float,
                          composite_score: float, **kwargs):
        """Add a CONFLICTS_WITH edge between two rule nodes."""
        if self.G.has_node(rule_1_id) and self.G.has_node(rule_2_id):
            self.G.add_edge(rule_1_id, rule_2_id, **{
                "edge_type": "CONFLICTS_WITH",
                "conflict_type": conflict_type,
                "confidence": confidence,
                "composite_score": composite_score,
                **kwargs,
            })

    # ==================================================================
    # 4D  Query methods
    # ==================================================================
    def get_rules_by_concept(self, concept_label: str) -> list[str]:
        """Return rule_ids that INVOLVE a concept."""
        cid = "concept_" + concept_label.replace(" ", "_")
        if not self.G.has_node(cid):
            return []
        return [
            src for src, _, d in self.G.in_edges(cid, data=True)
            if d.get("edge_type") == "INVOLVES"
        ]

    def get_neighbors(self, node_id: str,
                      edge_type: str | None = None) -> list[tuple]:
        """Get neighbouring nodes, optionally filtered by edge type."""
        result = []
        for _, tgt, d in self.G.out_edges(node_id, data=True):
            if edge_type is None or d.get("edge_type") == edge_type:
                result.append((tgt, d))
        for src, _, d in self.G.in_edges(node_id, data=True):
            if edge_type is None or d.get("edge_type") == edge_type:
                result.append((src, d))
        return result

    def get_cross_source_pairs(self, concept_label: str) -> list[tuple[str, str]]:
        """Return (rule_a, rule_b) pairs from different sources sharing a concept."""
        rule_ids = self.get_rules_by_concept(concept_label)
        pairs = []
        for i, r1 in enumerate(rule_ids):
            for r2 in rule_ids[i + 1:]:
                s1 = self.G.nodes[r1].get("source", "")
                s2 = self.G.nodes[r2].get("source", "")
                if s1 != s2:
                    pairs.append((r1, r2))
        return pairs

    def get_conflict_subgraph(self, rule_id: str) -> nx.DiGraph:
        """Return subgraph around a conflicting rule — its conflicts + shared concepts."""
        nodes = {rule_id}
        for _, tgt, d in self.G.out_edges(rule_id, data=True):
            if d.get("edge_type") in ("CONFLICTS_WITH", "INVOLVES"):
                nodes.add(tgt)
        for src, _, d in self.G.in_edges(rule_id, data=True):
            if d.get("edge_type") in ("CONFLICTS_WITH", "INVOLVES", "CONTAINS"):
                nodes.add(src)
        return self.G.subgraph(nodes).copy()

    def get_high_weight_conflicts(self, threshold: float = 0.7) -> list[dict]:
        """Return conflict edges with composite_score ≥ threshold."""
        results = []
        for u, v, d in self.G.edges(data=True):
            if (d.get("edge_type") == "CONFLICTS_WITH"
                    and d.get("composite_score", 0) >= threshold):
                results.append({"rule_1": u, "rule_2": v, **d})
        return sorted(results, key=lambda x: x.get("composite_score", 0),
                       reverse=True)

    def shortest_path(self, source: str, target: str) -> list[str] | None:
        """Shortest path between two nodes in the KG."""
        try:
            return nx.shortest_path(self.G, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_all_concept_labels(self) -> list[str]:
        """Return all concept labels in the graph."""
        return [
            d.get("label", "")
            for _, d in self.G.nodes(data=True)
            if d.get("node_type") == "concept"
        ]

    # ==================================================================
    # 4C  Build pipeline
    # ==================================================================
    def build(self, rules_path: str | None = None,
              corpus_path: str | None = None):
        """Full KG construction from rules.json (+ optional corpus for text)."""
        if rules_path is None:
            rules_path = os.path.join(DATA_DIR, "rules.json")

        with open(rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        articles = data.get("articles", [])
        rules = data.get("rules", [])

        print(f"[KG] Building graph from {len(articles)} articles, {len(rules)} rules …")

        self._add_article_nodes(articles)
        self._add_rule_nodes(rules)
        self._extract_and_add_concept_nodes(rules)
        self._add_contains_edges(articles)
        self._add_involves_edges()
        self._add_reference_edges(rules)

        print(f"[KG] Graph built — {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")

    # ==================================================================
    # 4E  Export
    # ==================================================================
    def export_graphml(self, path: str | None = None):
        """Export to GraphML for Gephi."""
        if path is None:
            path = os.path.join(DATA_DIR, "knowledge_graph.graphml")
        # NetworkX graphml can't serialise None — replace with ""
        G_clean = self.G.copy()
        import json
        for _, d in G_clean.nodes(data=True):
            for k, v in d.items():
                if v is None:
                    d[k] = ""
                elif isinstance(v, bool):
                    d[k] = str(v)
                elif isinstance(v, (list, dict, tuple)):
                    d[k] = json.dumps(v)
        for _, _, d in G_clean.edges(data=True):
            for k, v in d.items():
                if v is None:
                    d[k] = ""
                elif isinstance(v, bool):
                    d[k] = str(v)
                elif isinstance(v, (list, dict, tuple)):
                    d[k] = json.dumps(v)
        nx.write_graphml(G_clean, path)
        print(f"[KG] Exported GraphML -> {path}")

    def export_json(self, path: str | None = None):
        """Export nodes + edges to JSON."""
        if path is None:
            path = os.path.join(DATA_DIR, "knowledge_graph.json")

        def _serialisable(val):
            if val is None:
                return ""
            return val

        nodes = []
        for nid, d in self.G.nodes(data=True):
            nodes.append({"id": nid, **{k: _serialisable(v) for k, v in d.items()}})

        edges = []
        for u, v, d in self.G.edges(data=True):
            edges.append({"source": u, "target": v,
                          **{k: _serialisable(val) for k, val in d.items()}})

        with open(path, "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes, "edges": edges}, f, indent=2,
                       ensure_ascii=False)
        print(f"[KG] Exported JSON -> {path}")

    def export_html(self, path: str | None = None):
        """Interactive HTML visualisation via pyvis."""
        if path is None:
            path = os.path.join(DATA_DIR, "knowledge_graph.html")
        try:
            from pyvis.network import Network
        except ImportError:
            print("[KG] pyvis not installed — skipping HTML export")
            return

        net = Network(height="900px", width="100%", directed=True,
                      bgcolor="#0a0a0a", font_color="white")
        net.force_atlas_2based()

        colour_map = {
            "article": "#4fc3f7",
            "rule": "#aed581",
            "concept": "#ffb74d",
        }

        for nid, d in self.G.nodes(data=True):
            ntype = d.get("node_type", "")
            label = d.get("label", d.get("rule_id", d.get("article_id", nid)))
            net.add_node(nid, label=str(label)[:40],
                         color=colour_map.get(ntype, "#ccc"),
                         title=json.dumps({k: str(v)[:80] for k, v in d.items()},
                                          indent=1),
                         size=15 + d.get("frequency", 0) * 2)

        for u, v, d in self.G.edges(data=True):
            etype = d.get("edge_type", "")
            colour = "#f44336" if etype == "CONFLICTS_WITH" else "#888"
            net.add_edge(u, v, title=etype, color=colour,
                         width=2 if etype == "CONFLICTS_WITH" else 1)

        net.save_graph(path)
        print(f"[KG] Exported interactive HTML -> {path}")

    def save_all(self):
        """Export all formats."""
        self.export_graphml()
        self.export_json()
        self.export_html()

    def run(self):
        """Entry point: build + export."""
        self.build()
        self.save_all()


if __name__ == "__main__":
    PolicyKnowledgeGraph().run()
