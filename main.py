"""
Step 8: CLI Interface
Master entry point with argparse subcommands for the full pipeline.
"""

import argparse
import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def cmd_scrape(_args):
    """Scrape GDPR and ePrivacy articles."""
    from scraper import PolicyScraper
    PolicyScraper().run()


def cmd_extract(args):
    """Extract rules via LLM."""
    from rule_extractor import RuleExtractor
    RuleExtractor(backend=args.backend).run()


def cmd_score(_args):
    """Run hierarchical weight scoring."""
    from weight_scorer import WeightScorer
    WeightScorer().run()


def cmd_build_kg(_args):
    """Build and export the Policy Knowledge Graph."""
    from knowledge_graph import PolicyKnowledgeGraph
    kg = PolicyKnowledgeGraph()
    kg.run()


def cmd_embed(_args):
    """Build ChromaDB vector store."""
    from embedder import PolicyEmbedder
    PolicyEmbedder().run()


def cmd_detect(args):
    """Run conflict detection (KG + RAG + LLM)."""
    from knowledge_graph import PolicyKnowledgeGraph
    from rag_retriever import RAGRetriever
    from conflict_detector import ConflictDetector

    # Build KG
    kg = PolicyKnowledgeGraph()
    kg.build()

    # Init retriever
    retriever = RAGRetriever()

    # Detect
    detector = ConflictDetector(kg, retriever, backend=args.backend)
    detector.detect_all()

    # Re-export KG with conflict edges
    kg.save_all()

    # Print summary
    summary = detector.get_conflict_summary()
    print("\n=== Conflict Detection Summary ===")
    print(f"Total conflicts: {summary['total']}")
    print(f"By type: {json.dumps(summary['by_type'], indent=2)}")
    print(f"Average composite score: {summary['avg_score']}")


def cmd_explain(args):
    """Generate RAG+KG enhanced explanations."""
    from knowledge_graph import PolicyKnowledgeGraph
    from explainer import ConflictExplainer

    kg = PolicyKnowledgeGraph()
    kg.build()

    explainer = ConflictExplainer(kg, backend=args.backend)
    explainer.run()


def cmd_query(args):
    """Query conflicts by rule ID or concept."""
    conflicts_path = os.path.join(DATA_DIR, "conflicts_with_explanations.json")
    if not os.path.exists(conflicts_path):
        conflicts_path = os.path.join(DATA_DIR, "conflicts_raw.json")
    if not os.path.exists(conflicts_path):
        print("[QUERY] No conflicts file found. Run 'detect' first.")
        return

    with open(conflicts_path, "r", encoding="utf-8") as f:
        conflicts = json.load(f)

    results = conflicts

    if args.rule:
        results = [
            c for c in results
            if (c.get("rule_1", {}).get("rule_id") == args.rule
                or c.get("rule_2", {}).get("rule_id") == args.rule)
        ]

    if args.concept:
        results = [
            c for c in results
            if args.concept.lower() in
               [s.lower() for s in c.get("shared_concepts", [])]
        ]

    # Sort by composite score
    results = sorted(results,
                     key=lambda c: c.get("composite_score", 0),
                     reverse=True)

    if not results:
        print("[QUERY] No matching conflicts found.")
        return

    print(f"\n=== {len(results)} conflict(s) found ===\n")
    for c in results:
        print(f"--- {c.get('conflict_id', '?')} ---")
        print(f"  Type:       {c.get('type', '?')}")
        print(f"  Score:      {c.get('composite_score', 0)}")
        print(f"  Confidence: {c.get('confidence', 0)}")
        print(f"  Rule 1:     {c.get('rule_1', {}).get('rule_id', '?')} "
              f"({c.get('rule_1', {}).get('source', '')})")
        print(f"  Rule 2:     {c.get('rule_2', {}).get('rule_id', '?')} "
              f"({c.get('rule_2', {}).get('source', '')})")
        print(f"  Concepts:   {', '.join(c.get('shared_concepts', []))}")
        print(f"  Precedence: {c.get('precedence', '?')}")
        if c.get("explanation"):
            print(f"  Explanation: {c['explanation'][:200]}…")
        print()


def cmd_run_all(args):
    """Run all steps sequentially."""
    print("=" * 60)
    print("  POLICY CONFLICT DETECTION SYSTEM — FULL PIPELINE")
    print("=" * 60)

    print("\n[1/7] Scraping policies …")
    cmd_scrape(args)

    print("\n[2/7] Extracting rules via LLM …")
    cmd_extract(args)

    print("\n[3/7] Scoring rules …")
    cmd_score(args)

    print("\n[4/7] Building Knowledge Graph …")
    cmd_build_kg(args)

    print("\n[5/7] Embedding into ChromaDB …")
    cmd_embed(args)

    print("\n[6/7] Detecting conflicts …")
    cmd_detect(args)

    print("\n[7/7] Generating explanations …")
    cmd_explain(args)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutputs in: {DATA_DIR}")
    print("  • data/corpus.json")
    print("  • data/rules.json")
    print("  • data/knowledge_graph.graphml")
    print("  • data/knowledge_graph.json")
    print("  • data/knowledge_graph.html")
    print("  • data/conflicts_raw.json")
    print("  • data/conflicts_with_explanations.json")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="policy-conflict-detector",
        description="Policy Conflict Detection System using RAG, "
                    "Knowledge Graph, and Mistral-7B",
    )
    parser.add_argument(
        "--backend", choices=["ollama", "huggingface"],
        default="ollama",
        help="LLM backend to use (default: ollama)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline step")

    # scrape
    subparsers.add_parser("scrape", help="Scrape GDPR & ePrivacy articles")

    # extract
    subparsers.add_parser("extract", help="Extract rules via LLM")

    # score
    subparsers.add_parser("score", help="Run hierarchical weight scoring")

    # build-kg
    subparsers.add_parser("build-kg", help="Build Policy Knowledge Graph")

    # embed
    subparsers.add_parser("embed", help="Embed rules/articles into ChromaDB")

    # detect
    subparsers.add_parser("detect", help="Run conflict detection")

    # explain
    subparsers.add_parser("explain", help="Generate conflict explanations")

    # query
    q = subparsers.add_parser("query", help="Query conflicts")
    q.add_argument("--rule", type=str, default=None,
                   help="Filter by rule ID (e.g. GDPR_Art6_R1)")
    q.add_argument("--concept", type=str, default=None,
                   help="Filter by shared concept (e.g. 'consent')")

    # run-all
    subparsers.add_parser("run-all", help="Run all steps sequentially")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "scrape": cmd_scrape,
        "extract": cmd_extract,
        "score": cmd_score,
        "build-kg": cmd_build_kg,
        "embed": cmd_embed,
        "detect": cmd_detect,
        "explain": cmd_explain,
        "query": cmd_query,
        "run-all": cmd_run_all,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
