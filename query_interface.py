"""
Standalone Interactive Query Interface for Policy Conflict Detector.

Usage:
  # CLI usage interactive
  python query_interface.py

  # Or pass directly
  python query_interface.py --rule "Users must be notified within 72 hours of a data breach"
"""

import argparse
import sys
import os
import json
import re
import networkx as nx

try:
    from validator import PipelineValidator
except ImportError:
    PipelineValidator = None

try:
    from rule_extractor import llm_call
except ImportError:
    llm_call = None

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
GRAPH_PATH = os.path.join(DATA_DIR, "knowledge_graph.graphml")


class QueryInterface:
    def __init__(self, backend="ollama"):
        self.backend = backend
        self.client = None
        self.collection = None
        self.embed_fn = None
        self.graph = None
        self.concept_nodes = []
        self.rule_dictionary = {}
        
        self.setup()

    def setup(self):
        # 1. Setup ChromaDB
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            self.client = chromadb.PersistentClient(path=CHROMA_DIR)
            self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self.collection = self.client.get_collection(
                name="policy_rules", 
                embedding_function=self.embed_fn
            )
        except Exception as e:
            print(f"[ERROR] Could not load ChromaDB: {e}")
            sys.exit(1)

        # 2. Setup Knowledge Graph
        if os.path.exists(GRAPH_PATH):
            self.graph = nx.read_graphml(GRAPH_PATH)
            # Pre-extract concepts for quick matching
            self.concept_nodes = [
                d.get("label", n).lower() 
                for n, d in self.graph.nodes(data=True) 
                if d.get("node_type") == "concept"
            ]
        else:
            print(f"[WARNING] GraphML not found at {GRAPH_PATH}. Graph overlap matching will be disabled.")
            
        # 3. Cache base rules for quick lookup
        rules_path = os.path.join(DATA_DIR, "rules.json")
        if os.path.exists(rules_path):
            with open(rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for r in data.get("rules", []):
                    if "rule_id" in r:
                        self.rule_dictionary[r["rule_id"]] = r

    def infer_modality(self, statement: str) -> str:
        s = statement.lower()
        if re.search(r'\b(must not|prohibited|shall not|cannot|may not)\b', s):
            return "Prohibition"
        elif re.search(r'\b(must|shall|required|obligated|have to|needs to)\b', s):
            return "Obligation"
        elif re.search(r'\b(may|can|allowed|permitted|authorized|right to)\b', s):
            return "Permission"
        elif re.search(r'\b(except|unless|notwithstanding)\b', s):
            return "Exception"
        return "Undetermined"

    def determine_modality_clash(self, input_modality: str, target_modality: str):
        im = input_modality.lower()
        tm = target_modality.lower()
        
        if (im == "obligation" and tm == "prohibition") or (im == "prohibition" and tm == "obligation"):
            return True, f"{input_modality} vs {target_modality}"
        if (im == "permission" and tm == "prohibition") or (im == "prohibition" and tm == "permission"):
            return True, f"{input_modality} vs {target_modality}"
        
        # If they are compatible, it's a potential overlap, not a hard clash.
        return False, f"{input_modality} vs {target_modality}"

    def check_graph_overlap(self, user_statement: str, retrieved_rule_id: str) -> tuple[bool, str]:
        if not self.graph:
            return False, "Graph unavailable"
            
        # Find which known concepts explicitly appear in the user statement
        s = user_statement.lower()
        matched_concepts = []
        for c in self.concept_nodes:
            if c in s and len(c) > 3: # Ignore tiny concept matches
                matched_concepts.append(c)
                
        if not matched_concepts:
            return False, "No shared entity or scope detected"
            
        # Check if the retrieved rule connects to ANY of the matched concepts
        shared = []
        for n, d in self.graph.nodes(data=True):
            if d.get("node_type") == "concept" and d.get("label", n).lower() in matched_concepts:
                # Does the retrieved rule have an edge to this concept?
                if self.graph.has_edge(retrieved_rule_id, n) or self.graph.has_edge(n, retrieved_rule_id):
                    shared.append(d.get("label", n))
                    
        if shared:
            return True, f"Shared graph concepts: {', '.join(shared)}"
        
        return False, "No shared entity or scope detected"

    def explain_conflict(self, user_rule: str, their_rule: dict) -> tuple[str, str]:
        if not llm_call:
            return "(LLM logic not available)", ""
            
        system_prompt = (
            "You are a legal expert in GDPR and ePrivacy regulations.\n"
            "Explain in 3-5 sentences:\n"
            "1. Why these two rules conflict\n"
            "2. In what specific scenario the conflict would arise\n"
            "3. Which rule takes legal precedence under lex specialis and why\n"
        )
        
        user_prompt = (
            f"A user has proposed the following rule:\n{user_rule}\n\n"
            f"This potentially conflicts with the following existing legal rule:\n"
            f"Rule ID: {their_rule.get('rule_id', 'Unknown')}\n"
            f"Regulation: {their_rule.get('source', 'Unknown')}\n"
            f"Entity: {their_rule.get('entity', 'Unknown')}\n"
            f"Action: {their_rule.get('action', 'Unknown')}\n"
            f"Condition: {their_rule.get('condition', 'Unknown')}\n"
            f"Modality: {their_rule.get('modality', 'Unknown')}\n"
            f"Scope: {their_rule.get('scope', 'Unknown')}\n"
        )
        
        try:
            explanation = llm_call(system_prompt, user_prompt, backend=self.backend)
            
            # Very basic extraction trick just to populate fields if possible, usually we return it as a blob
            parts = explanation.split("3.")
            if len(parts) > 1:
                lex = "3." + parts[1]
                exp = parts[0].strip()
                return exp, lex
            return explanation, "See explanation text."
            
        except Exception as e:
            return f"(LLM Generation Failed: {e})", ""

    def analyze(self, rule_statement: str):
        if PipelineValidator is not None:
            validation = PipelineValidator.validate_query_input(rule_statement)
            if not validation["passed"]:
                return {
                    "input_rule": rule_statement,
                    "inferred_modality": "Unknown",
                    "conflicts": [],
                    "summary": {},
                }
            for issue in validation["issues"]:
                if issue.startswith("WARN:"):
                    print(f"[VALIDATOR] {issue}")

        inferred_modality = self.infer_modality(rule_statement)
        
        # 1. Embed and retrieve top 10 rules
        results = self.collection.query(
            query_texts=[rule_statement],
            n_results=10
        )
        
        if not results['ids'] or not results['ids'][0]:
            return {"input_rule": rule_statement, "inferred_modality": inferred_modality, "conflicts": [], "summary": {}}
            
        ids = results['ids'][0]
        distances = results['distances'][0]
        
        conflicts = []
        confirmed_count = 0
        possible_count = 0
        
        for i, rule_id in enumerate(ids):
            # Distance -> semantic similarity. Distances are usually cosine or L2. Let's invert to similarity (0-1 approx range)
            similarity = max(0, 1.0 - distances[i])
            if similarity < 0.2: # Too low similarity, skip
                continue
                
            their_rule_meta = self.rule_dictionary.get(rule_id, {})
            their_modality = their_rule_meta.get("modality", "Unknown")
            
            is_clash, clash_desc = self.determine_modality_clash(inferred_modality, their_modality)
            is_overlap, overlap_desc = self.check_graph_overlap(rule_statement, rule_id)
            
            if is_clash:
                if is_overlap:
                    c_type = "CONFIRMED CONFLICT"
                    confirmed_count += 1
                else:
                    c_type = "POSSIBLE CONFLICT"
                    possible_count += 1
            else:
                if is_overlap and similarity > 0.6:
                    c_type = "POSSIBLE CONFLICT"
                    possible_count += 1
                else:
                    continue # No tangible conflict
                    
            # Compute a rough 0-100% confidence
            # Semantic similarity scales from 0.4 to 0.8 mostly.
            confidence = int(min(0.99, (similarity * 0.5 + (0.5 if is_overlap else 0.2))) * 100)
            
            their_full_text = " ".join(filter(None, [
                their_rule_meta.get("entity", ""),
                their_rule_meta.get("action", ""),
                their_rule_meta.get("condition", ""),
                their_rule_meta.get("modality", ""),
            ]))
            
            # Fetch explanations
            explanation, lex_resolution = self.explain_conflict(rule_statement, their_rule_meta)
            
            conflicts.append({
                "conflict_id": rule_id,
                "type": "CONFIRMED" if c_type == "CONFIRMED CONFLICT" else "POSSIBLE",
                "confidence": round(confidence / 100.0, 2),
                "their_rule": their_full_text,
                "modality_clash": clash_desc,
                "graph_overlap": overlap_desc,
                "explanation": explanation,
                "lex_specialis_resolution": lex_resolution,
                "regulation": their_rule_meta.get("source", "Unknown"),
                "_raw_type": c_type,
                "_raw_conf": confidence
            })
            
        conflicts = sorted(conflicts, key=lambda x: x["confidence"], reverse=True)
        
        summary = {
            "confirmed": confirmed_count,
            "possible": possible_count,
            "total_checked": len(ids)
        }
        
        return {
            "input_rule": rule_statement,
            "inferred_modality": inferred_modality,
            "conflicts": conflicts,
            "summary": summary
        }

    def print_report(self, analysis_data: dict):
        print("  ========================================")
        print("  CONFLICT ANALYSIS REPORT")
        print(f"  Input Rule: \"{analysis_data['input_rule']}\"")
        print(f"  Inferred Modality: {analysis_data['inferred_modality']}")
        print("  ========================================\n")
        
        if not analysis_data['conflicts']:
            print("  No conflicts detected with existing rules")
        
        for i, c in enumerate(analysis_data['conflicts'], 1):
            exp = str(c['explanation']).replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\n  ').strip()
            lex = str(c['lex_specialis_resolution']).replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\n  ').strip()
            their = str(c['their_rule']).replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').strip()
            
            print(f"  [CONFLICT #{i}] — {c['_raw_type']} (Confidence: {c['_raw_conf']}%)")
            print(f"  Conflicting Rule ID: {c['conflict_id']}")
            print(f"  Regulation: {c.get('regulation', 'Unknown')}")
            print(f"  Their Rule: \"{their}\"")
            print(f"  Modality Clash: Your {c['modality_clash'].split(' vs ')[0]} vs Their {c['modality_clash'].split(' vs ')[1] if ' vs ' in c['modality_clash'] else 'Compatible'}")
            print(f"  Graph Overlap: {c['graph_overlap']}\n")
            print(f"  Explanation:\n  {exp}\n")
            print(f"  Lex Specialis Resolution:\n  {lex}")
            print("  ----------------------------------------\n")

        s = analysis_data['summary']
        print("  ========================================")
        print(f"  SUMMARY: {s['confirmed']} Confirmed Conflicts, {s['possible']} Possible Conflict, {s['total_checked'] - s['confirmed'] - s['possible']} No Conflict")
        print("  ========================================")


def main():
    parser = argparse.ArgumentParser(description="Standalone interactive query interface.")
    parser.add_argument("--rule", type=str, help="Rule text to check")
    parser.add_argument("--backend", type=str, default="ollama", choices=["ollama", "huggingface"], help="LLM backend")
    args = parser.parse_args()
    
    interface = QueryInterface(backend=args.backend)
    
    rule = args.rule
    if not rule:
        try:
            rule = input("Enter a rule to analyze: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)
            
    if not rule:
        print("No rule provided.")
        sys.exit(1)
        
    print(f"\nAnalyzing rule: '{rule}'...\n")
    data = interface.analyze(rule)
    interface.print_report(data)

if __name__ == "__main__":
    main()
