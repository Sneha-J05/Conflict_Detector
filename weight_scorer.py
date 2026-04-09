"""
Step 3: Hierarchical Weight Scoring
Assigns source, article-importance, and severity weights to every rule.
"""

import os
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------------------
# Weight maps
# ---------------------------------------------------------------------------

SOURCE_WEIGHTS = {
    "GDPR": 1.0,
    "ePrivacy": 0.85,
}

# GDPR article-importance map (article_number → weight)
GDPR_ARTICLE_MAP = {
    **{n: 0.3 for n in range(1, 5)},       # definitions
    5: 1.0,                                  # principles
    6: 1.0,                                  # lawful basis
    7: 1.0,                                  # consent
    8: 0.85,
    9: 1.0,                                  # special categories
    10: 0.85,
    11: 0.85,
    **{n: 0.85 for n in range(12, 24)},      # data subject rights
    **{n: 0.8 for n in range(24, 44)},       # controller obligations
    **{n: 0.75 for n in range(44, 51)},      # international transfers
    **{n: 0.9 for n in range(51, 85)},       # enforcement
}

EPRIVACY_ARTICLE_MAP = {
    1: 0.3, 2: 0.3,          # scope / definitions
    5: 1.0,                   # confidentiality
    6: 0.9,                   # traffic data
    13: 0.85,                 # unsolicited comms
}

ARTICLE_TYPE_WEIGHTS = {
    "core_obligation": 1.0,
    "fundamental_right": 1.0,
    "enforcement": 0.9,
    "administrative": 0.6,
    "exception": 0.7,
    "definition": 0.3,
}

SEVERITY_WEIGHTS = {
    "direct_conflict": 1.0,
    "conditional_conflict": 0.7,
    "logical_inconsistency": 0.8,
    "redundancy": 0.3,
    "ambiguity": 0.5,
}

# ePrivacy articles that enjoy lex specialis status
LEX_SPECIALIS_ARTICLES = {5, 6, 9, 13}


class WeightScorer:
    """Assigns hierarchical weights to every extracted rule."""

    def __init__(self, backend: str = "ollama"):
        self.backend = backend

    # ------------------------------------------------------------------
    # Dimension A: source priority
    # ------------------------------------------------------------------
    def source_weight(self, rule: dict) -> float:
        return SOURCE_WEIGHTS.get(rule.get("source", ""), 0.5)

    # ------------------------------------------------------------------
    # Dimension B: article importance
    # ------------------------------------------------------------------
    def article_weight(self, rule: dict) -> float:
        source = rule.get("source", "")
        art_num_raw = rule.get("article_number", 0)
        
        if isinstance(art_num_raw, list):
            art_num_raw = art_num_raw[0] if art_num_raw else 0
        try:
            art_num = int(art_num_raw)
        except (ValueError, TypeError):
            art_num = 0

        if source == "GDPR":
            if art_num in GDPR_ARTICLE_MAP:
                return GDPR_ARTICLE_MAP[art_num]
        elif source == "ePrivacy":
            if art_num in EPRIVACY_ARTICLE_MAP:
                return EPRIVACY_ARTICLE_MAP[art_num]

        # Fallback: use article_type
        art_type = rule.get("article_type", "core_obligation")
        return ARTICLE_TYPE_WEIGHTS.get(art_type, 0.5)

    # ------------------------------------------------------------------
    # Dimension C: severity (filled later during conflict detection)
    # ------------------------------------------------------------------
    @staticmethod
    def severity_weight(severity_type: str) -> float:
        return SEVERITY_WEIGHTS.get(severity_type, 0.5)

    # ------------------------------------------------------------------
    # Lex specialis flag
    # ------------------------------------------------------------------
    @staticmethod
    def is_lex_specialis(rule: dict) -> bool:
        if rule.get("source") == "ePrivacy":
            art_num_raw = rule.get("article_number", 0)
            if isinstance(art_num_raw, list):
                art_num_raw = art_num_raw[0] if art_num_raw else 0
            try:
                art_num = int(art_num_raw)
            except (ValueError, TypeError):
                art_num = 0
            return art_num in LEX_SPECIALIS_ARTICLES
        return False

    # ------------------------------------------------------------------
    # Score all rules
    # ------------------------------------------------------------------
    def score_all(self, rules_path: str | None = None):
        """Read rules.json, attach weights, rewrite file."""
        if rules_path is None:
            rules_path = os.path.join(DATA_DIR, "rules.json")

        with open(rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rules = data.get("rules", [])
        for rule in rules:
            sw = self.source_weight(rule)
            aw = self.article_weight(rule)
            rule["weights"] = {
                "source": sw,
                "article": aw,
                "severity": None,       # filled during conflict detection
                "composite": None,       # filled during conflict detection
                "lex_specialis": self.is_lex_specialis(rule),
            }

        # Also update article-level data
        for art in data.get("articles", []):
            for rule in art.get("rules", []):
                rule["source"] = art.get("source", rule.get("source", ""))
                rule["article_number"] = art.get("article_number",
                                                  rule.get("article_number", 0))
                rule["article_type"] = art.get("article_type",
                                                rule.get("article_type", ""))
                sw = self.source_weight(rule)
                aw = self.article_weight(rule)
                rule["weights"] = {
                    "source": sw,
                    "article": aw,
                    "severity": None,
                    "composite": None,
                    "lex_specialis": self.is_lex_specialis(rule),
                }

        with open(rules_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[SCORER] Weights assigned to {len(rules)} rules → {rules_path}")

    def run(self):
        self.score_all()


if __name__ == "__main__":
    WeightScorer().run()
