"""
Validation layer for the Policy Conflict Detection System.
Each method is independently runnable and prints a [VALIDATOR] prefixed summary.
"""

import json
import os
import re
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

LEGAL_KEYWORDS = [
    "data", "process", "consent", "notify", "transfer", "retain",
    "collect", "store", "breach", "access", "right", "obligation",
    "controller", "processor", "subject",
]

GDPR_SHORT_ARTICLES = {"GDPR_Art1", "GDPR_Art2", "GDPR_Art3", "GDPR_Art4"}
EPRIVACY_SHORT_ARTICLES = {"EPRIVACY_Art1", "EPRIVACY_Art2"}
LEGITIMATE_ISOLATED_PREFIXES = ["EPRIVACY_Art12", "EPRIVACY_Art13", "EPRIVACY_Art19"]
LEX_SPECIALIS_ALLOWED_PREFIXES = (
    "EPRIVACY_Art5",
    "EPRIVACY_Art6",
    "EPRIVACY_Art9",
    "EPRIVACY_Art13",
)

VALID_MODALITIES = {"obligation", "permission", "prohibition", "exception"}

VALID_RELATIONSHIPS = {
    "direct_conflict", "conditional_conflict", "logical_inconsistency",
    "redundancy", "exception", "no_conflict",
}

LEX_SPECIALIS_ARTICLES = {5, 6, 9, 13}

RULE_ID_PATTERN = re.compile(r"^(GDPR|EPRIVACY)_Art\d+_R\d+$")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_summary(label, passed, issues):
    print(f"[VALIDATOR] --- {label} ---")
    for issue in issues:
        print(f"[VALIDATOR]   {'⚠' if issue.startswith('WARN:') else '✗'} {issue}")
    if passed:
        print(f"[VALIDATOR] ✓ {label} passed ({len(issues)} warning(s))." if issues else f"[VALIDATOR] ✓ {label} passed.")
    else:
        hard = [i for i in issues if not i.startswith("WARN:")]
        print(f"[VALIDATOR] ✗ {label} FAILED — {len(hard)} error(s), {len(issues) - len(hard)} warning(s).")


class PipelineValidator:

    @staticmethod
    def validate_corpus(corpus_path=None):
        label = "Corpus Validation"
        issues = []

        if corpus_path is None:
            corpus_path = os.path.join(DATA_DIR, "corpus.json")

        try:
            corpus = _load_json(corpus_path)
        except FileNotFoundError:
            print(f"[VALIDATOR] ✗ {label}: file not found → {corpus_path}")
            return {"passed": False, "issues": [f"File not found: {corpus_path}"]}
        except json.JSONDecodeError as e:
            print(f"[VALIDATOR] ✗ {label}: JSON parse error — {e}")
            return {"passed": False, "issues": [f"JSON parse error: {e}"]}

        id_counter = Counter(entry.get("id", "") for entry in corpus)

        for dup_id, count in id_counter.items():
            if count > 1:
                issues.append(f"Duplicate article id: '{dup_id}' appears {count} times")

        present_ids = set(id_counter.keys())

        for n in range(1, 100):
            expected = f"GDPR_Art{n}"
            if expected not in present_ids:
                issues.append(f"Missing GDPR article: {expected}")

        for n in range(1, 23):
            expected = f"EPRIVACY_Art{n}"
            if expected not in present_ids:
                issues.append(f"Missing ePrivacy article: {expected}")

        for entry in corpus:
            text = entry.get("text", "")
            article_id = entry.get("id", "?")
            if not text:
                issues.append(f"Empty text for article: {article_id}")
                continue

            if article_id.startswith("GDPR_"):
                min_len = 100 if article_id in GDPR_SHORT_ARTICLES else 300
            elif article_id.startswith("EPRIVACY_"):
                min_len = 80 if article_id in EPRIVACY_SHORT_ARTICLES else 150
            else:
                min_len = 200

            if len(text) < min_len:
                issues.append(f"Short text ({len(text)} chars, min {min_len}) for article: {article_id}")

        passed = all(not i.startswith("✗") and not i.startswith("Missing") and not i.startswith("Duplicate") and not i.startswith("Empty") and not i.startswith("Short") for i in [])
        hard_issues = [i for i in issues if not i.startswith("WARN:")]
        passed = len(hard_issues) == 0

        _print_summary(label, passed, issues)
        return {"passed": passed, "issues": issues}

    @staticmethod
    def validate_rules(rules_path=None, corpus_path=None):
        label = "Rules Validation"
        issues = []

        if rules_path is None:
            rules_path = os.path.join(DATA_DIR, "rules.json")
        if corpus_path is None:
            corpus_path = os.path.join(DATA_DIR, "corpus.json")

        try:
            rules_data = _load_json(rules_path)
        except FileNotFoundError:
            print(f"[VALIDATOR] ✗ {label}: file not found → {rules_path}")
            return {"passed": False, "issues": [f"File not found: {rules_path}"]}
        except json.JSONDecodeError as e:
            print(f"[VALIDATOR] ✗ {label}: JSON parse error — {e}")
            return {"passed": False, "issues": [f"JSON parse error: {e}"]}

        rules = rules_data.get("rules", [])
        articles = rules_data.get("articles", [])

        required_fields = ["rule_id", "entity", "action", "condition", "modality", "scope", "raw_text"]

        rule_id_counter = Counter(r.get("rule_id", "") for r in rules)
        for dup_id, count in rule_id_counter.items():
            if count > 1:
                issues.append(f"Duplicate rule_id: '{dup_id}' appears {count} times")

        for r in rules:
            rid = r.get("rule_id", "<no id>")

            for field in required_fields:
                val = r.get(field)
                if val is None or (isinstance(val, str) and not val.strip()):
                    issues.append(f"Rule '{rid}': missing or empty field '{field}'")

            modality = r.get("modality", "")
            if modality.lower() not in VALID_MODALITIES:
                issues.append(f"Rule '{rid}': invalid modality '{modality}'")

            if rid != "<no id>" and not RULE_ID_PATTERN.match(rid):
                issues.append(f"Rule '{rid}': rule_id does not match expected format")

            weights = r.get("weights")
            if isinstance(weights, dict):
                source_weight = weights.get("source")
                try:
                    sw = float(source_weight)
                    if not (0 < sw <= 1):
                        issues.append(f"Rule '{rid}': source {sw} out of range (0, 1]")
                    elif sw not in (1.0, 0.85):
                        issues.append(f"Rule '{rid}': source {sw} invalid (allowed: 1.0 or 0.85)")
                except (TypeError, ValueError):
                    issues.append(f"Rule '{rid}': source is not a number: '{source_weight}'")

                article_weight = weights.get("article")
                try:
                    aw = float(article_weight)
                    if not (0 < aw <= 1):
                        issues.append(f"Rule '{rid}': article {aw} out of range (0, 1]")
                except (TypeError, ValueError):
                    issues.append(f"Rule '{rid}': article is not a number: '{article_weight}'")

                composite_score = weights.get("composite")
                if composite_score is not None:
                    try:
                        cs = float(composite_score)
                        if not (0 < cs <= 1):
                            issues.append(f"Rule '{rid}': composite {cs} out of range (0, 1]")
                    except (TypeError, ValueError):
                        issues.append(f"Rule '{rid}': composite is not a number: '{composite_score}'")

                if "lex_specialis" in weights and not isinstance(weights.get("lex_specialis"), bool):
                    issues.append(
                        f"Rule '{rid}': lex_specialis must be boolean when present"
                    )

                lex = weights.get("lex_specialis", False)
                if lex is True:
                    if rid.startswith("GDPR_"):
                        issues.append(f"Rule '{rid}': lex_specialis=True on a GDPR rule (only ePrivacy allowed)")
                    elif rid.startswith("EPRIVACY_") and not rid.startswith(LEX_SPECIALIS_ALLOWED_PREFIXES):
                        issues.append(
                            f"Rule '{rid}': lex_specialis=True on non-qualifying ePrivacy article "
                            f"(allowed: Art5, Art6, Art9, Art13)"
                        )

        try:
            corpus = _load_json(corpus_path)
            corpus_ids = {entry.get("id", "") for entry in corpus}
            articles_with_rules = set()
            for art in articles:
                if art.get("rules"):
                    articles_with_rules.add(art.get("article_id", ""))

            for cid in corpus_ids:
                if cid and cid not in articles_with_rules:
                    issues.append(f"Article '{cid}' has no extracted rules")
        except FileNotFoundError:
            issues.append(f"WARN: corpus.json not found; skipping per-article rule coverage check")
        except json.JSONDecodeError:
            issues.append("WARN: corpus.json could not be parsed; skipping per-article rule coverage check")

        hard_issues = [i for i in issues if not i.startswith("WARN:")]
        passed = len(hard_issues) == 0

        _print_summary(label, passed, issues)
        return {"passed": passed, "issues": issues}

    @staticmethod
    def validate_knowledge_graph(kg_path=None):
        label = "Knowledge Graph Validation"
        issues = []

        if kg_path is None:
            kg_path = os.path.join(DATA_DIR, "knowledge_graph.json")

        try:
            kg = _load_json(kg_path)
        except FileNotFoundError:
            print(f"[VALIDATOR] ✗ {label}: file not found → {kg_path}")
            return {"passed": False, "issues": [f"File not found: {kg_path}"]}
        except json.JSONDecodeError as e:
            print(f"[VALIDATOR] ✗ {label}: JSON parse error — {e}")
            return {"passed": False, "issues": [f"JSON parse error: {e}"]}

        nodes = kg.get("nodes")
        edges = kg.get("edges")

        if not isinstance(nodes, list) or len(nodes) == 0:
            issues.append("'nodes' key is missing or empty")
        if not isinstance(edges, list) or len(edges) == 0:
            issues.append("'edges' key is missing or empty")

        if issues:
            _print_summary(label, False, issues)
            return {"passed": False, "issues": issues}

        node_ids = set()
        for node in nodes:
            nid = node.get("id")
            if not nid:
                issues.append("A node has a null or empty 'id' field")
            else:
                node_ids.add(nid)

        targets_of_contains = set()
        node_edge_counts = Counter()
        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            etype = edge.get("edge_type", "")
            if src:
                node_edge_counts[src] += 1
            if tgt:
                node_edge_counts[tgt] += 1
            if etype == "CONTAINS" and tgt:
                targets_of_contains.add(tgt)

        rule_nodes = {
            n.get("id") for n in nodes
            if n.get("node_type") == "rule" and n.get("id")
        }

        for rid in rule_nodes:
            if rid not in targets_of_contains:
                issues.append(f"Rule node '{rid}' has no CONTAINS edge pointing to it")

        isolated = [nid for nid in node_ids if node_edge_counts.get(nid, 0) == 0]
        if isolated:
            hard_isolated = []
            for nid in isolated:
                if any(nid.startswith(prefix) for prefix in LEGITIMATE_ISOLATED_PREFIXES):
                    issues.append(f"WARN: Isolated node may be legitimate: {nid}")
                else:
                    hard_isolated.append(nid)
            if hard_isolated:
                issues.append(
                    f"{len(hard_isolated)} isolated node(s) detected (no edges): e.g. {hard_isolated[:3]}"
                )

        hard_issues = [i for i in issues if not i.startswith("WARN:")]
        passed = len(hard_issues) == 0

        _print_summary(label, passed, issues)
        return {"passed": passed, "issues": issues}

    @staticmethod
    def validate_conflicts(rules_path=None, conflicts_path=None):
        label = "Conflicts Validation"
        issues = []

        if rules_path is None:
            rules_path = os.path.join(DATA_DIR, "rules.json")
        if conflicts_path is None:
            conflicts_path = os.path.join(DATA_DIR, "conflicts_raw.json")

        try:
            conflicts = _load_json(conflicts_path)
        except FileNotFoundError:
            print(f"[VALIDATOR] ✗ {label}: conflicts file not found → {conflicts_path}")
            return {"passed": False, "issues": [f"File not found: {conflicts_path}"]}
        except json.JSONDecodeError as e:
            return {"passed": False, "issues": [f"JSON parse error (conflicts): {e}"]}

        for idx, conflict in enumerate(conflicts):
            cid = conflict.get("conflict_id", f"<index {idx}>")
            r1 = conflict.get("rule_1", {})
            r2 = conflict.get("rule_2", {})
            r1_id = r1.get("rule_id", "") if isinstance(r1, dict) else ""
            r2_id = r2.get("rule_id", "") if isinstance(r2, dict) else ""
            pair_id = f"{r1_id} vs {r2_id}"

            confidence = conflict.get("confidence")
            if confidence is None:
                issues.append(f"Conflict '{cid}': missing 'confidence' field")
            elif isinstance(confidence, str):
                issues.append(f"Conflict '{cid}': 'confidence' is a string ('{confidence}'), expected float")
            else:
                try:
                    cv = float(confidence)
                    if not (0.0 <= cv <= 1.0):
                        issues.append(f"Conflict '{cid}': confidence {cv} out of range [0.0, 1.0]")
                except (TypeError, ValueError):
                    issues.append(f"Conflict '{cid}': confidence value is not numeric: '{confidence}'")

            relationship = conflict.get("type", "")
            if relationship not in VALID_RELATIONSHIPS:
                issues.append(f"Conflict '{cid}': invalid relationship '{relationship}'")

            src1 = r1.get("source", "") if isinstance(r1, dict) else ""
            src2 = r2.get("source", "") if isinstance(r2, dict) else ""
            if src1 and src2 and src1 == src2:
                issues.append(
                    f"Conflict '{cid}': both rules come from the same source '{src1}' "
                    f"(expected cross-source)"
                )

            composite_score = conflict.get("composite_score")
            composite_value = None
            if composite_score is None:
                issues.append(f"Missing or null composite_score for conflict: {pair_id}")
            else:
                try:
                    composite_value = float(composite_score)
                    if not (0 < composite_value <= 1):
                        issues.append(f"Composite score out of range for conflict: {pair_id} — got {composite_score}")
                except (TypeError, ValueError):
                    issues.append(f"Composite score out of range for conflict: {pair_id} — got {composite_score}")

            weights_1 = r1.get("weights", {}) if isinstance(r1, dict) else {}
            weights_2 = r2.get("weights", {}) if isinstance(r2, dict) else {}
            severity_raw = weights_1.get("severity") if isinstance(weights_1, dict) else None
            severity_value = None
            if severity_raw is not None:
                try:
                    severity_value = float(severity_raw)
                except (TypeError, ValueError):
                    severity_value = None

            if severity_value is not None and 0 < severity_value <= 1:
                if relationship == "direct_conflict" and severity_value < 0.5:
                    issues.append(
                        f"WARN: Severity {severity_value} seems inconsistent with relationship type "
                        f"'{relationship}' for conflict: {r1_id} vs {r2_id}"
                    )
                elif relationship == "logical_inconsistency" and severity_value < 0.4:
                    issues.append(
                        f"WARN: Severity {severity_value} seems inconsistent with relationship type "
                        f"'{relationship}' for conflict: {r1_id} vs {r2_id}"
                    )
                elif relationship == "conditional_conflict" and severity_value < 0.3:
                    issues.append(
                        f"WARN: Severity {severity_value} seems inconsistent with relationship type "
                        f"'{relationship}' for conflict: {r1_id} vs {r2_id}"
                    )
                elif relationship == "redundancy" and severity_value > 0.5:
                    issues.append(
                        f"WARN: Severity {severity_value} seems inconsistent with relationship type "
                        f"'{relationship}' for conflict: {r1_id} vs {r2_id}"
                    )
                elif relationship == "exception" and severity_value > 0.6:
                    issues.append(
                        f"WARN: Severity {severity_value} seems inconsistent with relationship type "
                        f"'{relationship}' for conflict: {r1_id} vs {r2_id}"
                    )
                elif relationship == "no_conflict" and severity_value > 0.3:
                    issues.append(
                        f"WARN: Severity {severity_value} seems inconsistent with relationship type "
                        f"'{relationship}' for conflict: {r1_id} vs {r2_id}"
                    )

            if (
                isinstance(weights_1, dict) and isinstance(weights_2, dict)
                and severity_value is not None and composite_value is not None
                and 0 < severity_value <= 1 and 0 < composite_value <= 1
            ):
                source_weight = weights_1.get("source")
                article_weight = weights_1.get("article")
                try:
                    source_value = float(source_weight)
                    article_value = float(article_weight)
                    expected = source_value * article_value * severity_value
                    if round(abs(expected - composite_value), 4) > 0.01:
                        issues.append(
                            f"Composite score mismatch for conflict {r1_id} vs {r2_id}: "
                            f"expected {expected:.4f}, got {composite_value:.4f}"
                        )
                except (TypeError, ValueError):
                    pass

            composite_score_raw = conflict.get("composite_score")
            try:
                composite_score_value = float(composite_score_raw)
            except (TypeError, ValueError):
                composite_score_value = None

            rule_1 = conflict.get("rule_1", {})
            rule_2 = conflict.get("rule_2", {})
            rule_1_weights = rule_1.get("weights", {}) if isinstance(rule_1, dict) else {}
            rule_2_weights = rule_2.get("weights", {}) if isinstance(rule_2, dict) else {}

            rule_1_source_raw = rule_1_weights.get("source") if isinstance(rule_1_weights, dict) else None
            rule_1_article_raw = rule_1_weights.get("article") if isinstance(rule_1_weights, dict) else None

            if (
                composite_score_value is not None
                and rule_1_source_raw not in (None, 0, 0.0)
                and rule_1_article_raw not in (None, 0, 0.0)
            ):
                try:
                    source_value = float(rule_1_source_raw)
                    article_value = float(rule_1_article_raw)
                except (TypeError, ValueError):
                    source_value = None
                    article_value = None

                if source_value is not None and article_value is not None and source_value != 0 and article_value != 0:
                    derived_severity = round(composite_score_value / (source_value * article_value), 4)

                    if derived_severity <= 0 or derived_severity > 1.0:
                        issues.append(
                            f"Composite score implies invalid severity {derived_severity} for conflict: {r1_id} vs {r2_id}"
                        )
                    else:
                        severity_ranges = {
                            "direct_conflict": (0.7, 1.0),
                            "logical_inconsistency": (0.6, 1.0),
                            "conditional_conflict": (0.4, 0.8),
                            "exception": (0.2, 0.6),
                            "redundancy": (0.1, 0.4),
                            "no_conflict": (0.0, 0.2),
                        }
                        expected_range = severity_ranges.get(relationship)
                        if expected_range is not None:
                            min_sev, max_sev = expected_range
                            if derived_severity < min_sev or derived_severity > max_sev:
                                issues.append(
                                    f"WARN: Derived severity {derived_severity} is outside expected range {min_sev}-{max_sev} "
                                    f"for type '{relationship}': {r1_id} vs {r2_id}"
                                )

                        rule_2_source_raw = rule_2_weights.get("source") if isinstance(rule_2_weights, dict) else None
                        rule_2_article_raw = rule_2_weights.get("article") if isinstance(rule_2_weights, dict) else None

                        if (
                            rule_2_source_raw not in (None, 0, 0.0)
                            and rule_2_article_raw not in (None, 0, 0.0)
                        ):
                            try:
                                rule_2_source_value = float(rule_2_source_raw)
                                rule_2_article_value = float(rule_2_article_raw)
                            except (TypeError, ValueError):
                                rule_2_source_value = None
                                rule_2_article_value = None

                            if (
                                rule_2_source_value is not None
                                and rule_2_article_value is not None
                                and rule_2_source_value != 0
                                and rule_2_article_value != 0
                            ):
                                expected_r2 = round(rule_2_source_value * rule_2_article_value * derived_severity, 4)
                                if abs(composite_score_value - expected_r2) > 0.15:
                                    issues.append(
                                        f"WARN: Composite score may favour rule_1 weights heavily for conflict: {r1_id} vs {r2_id} "
                                        f"— r1-based: {composite_score_value}, r2-based: {expected_r2}"
                                    )

        hard_issues = [i for i in issues if not i.startswith("WARN:")]
        passed = len(hard_issues) == 0

        _print_summary(label, passed, issues)
        return {"passed": passed, "issues": issues}

    @staticmethod
    def validate_explanations(conflicts_path=None, explained_path=None):
        label = "Explanations Validation"
        issues = []

        if conflicts_path is None:
            conflicts_path = os.path.join(DATA_DIR, "conflicts_raw.json")
        if explained_path is None:
            explained_path = os.path.join(DATA_DIR, "conflicts_with_explanations.json")

        try:
            conflicts_raw = _load_json(conflicts_path)
        except FileNotFoundError:
            print(f"[VALIDATOR] ✗ {label}: conflicts_raw.json not found → {conflicts_path}")
            return {"passed": False, "issues": [f"File not found: {conflicts_path}"]}
        except json.JSONDecodeError as e:
            return {"passed": False, "issues": [f"JSON parse error (conflicts_raw): {e}"]}

        try:
            conflicts_explained = _load_json(explained_path)
        except FileNotFoundError:
            print(f"[VALIDATOR] ✗ {label}: conflicts_with_explanations.json not found → {explained_path}")
            return {"passed": False, "issues": [f"File not found: {explained_path}"]}
        except json.JSONDecodeError as e:
            return {"passed": False, "issues": [f"JSON parse error (explained): {e}"]}

        explained_map = {
            entry.get("conflict_id", ""): entry
            for entry in conflicts_explained
        }

        for conflict in conflicts_raw:
            cid = conflict.get("conflict_id", "?")

            if cid not in explained_map:
                issues.append(f"Conflict '{cid}': no matching entry in conflicts_with_explanations.json")
                continue

            explanation = explained_map[cid].get("explanation", "")
            if not explanation:
                issues.append(f"Conflict '{cid}': explanation is null or empty")
            elif len(str(explanation).strip()) < 50:
                issues.append(
                    f"Conflict '{cid}': explanation too short "
                    f"({len(str(explanation).strip())} chars, min 50)"
                )

        hard_issues = [i for i in issues if not i.startswith("WARN:")]
        passed = len(hard_issues) == 0

        _print_summary(label, passed, issues)
        return {"passed": passed, "issues": issues}

    @staticmethod
    def validate_query_input(user_input: str):
        label = "Query Input Validation"
        issues = []

        if not user_input or not user_input.strip():
            issues.append("Input is empty")
            _print_summary(label, False, issues)
            return {"passed": False, "issues": issues}

        if len(user_input.strip()) < 10:
            issues.append("Input too short (min 10 chars)")
            _print_summary(label, False, issues)
            return {"passed": False, "issues": issues}

        lowered = user_input.lower()
        tier_1_keywords = [
            "consent", "process", "controller", "processor", "transfer", "breach",
            "retention", "lawful basis", "data subject", "supervisory authority",
        ]
        tier_2_keywords = [
            "data", "user", "notify", "access", "collect", "store",
            "right", "obligation", "gdpr", "eprivacy", "cookie", "tracking",
        ]
        has_tier_1 = any(kw in lowered for kw in tier_1_keywords)
        has_tier_2 = any(kw in lowered for kw in tier_2_keywords)

        if not has_tier_1 and has_tier_2:
            issues.append(
                "WARN: Query has weak legal signal — consider using specific legal terms like "
                "'consent', 'lawful basis', or 'data subject'"
            )
        elif not has_tier_1 and not has_tier_2:
            issues.append("WARN: No legal keywords detected — query may return irrelevant results")

        hard_issues = [i for i in issues if not i.startswith("WARN:")]
        passed = len(hard_issues) == 0

        _print_summary(label, passed, issues)
        return {"passed": passed, "issues": issues}

    @classmethod
    def run_all(cls):
        print("[VALIDATOR] ========================================")
        print("[VALIDATOR]   FULL PIPELINE VALIDATION")
        print("[VALIDATOR] ========================================\n")

        steps = [
            ("Corpus",     lambda: cls.validate_corpus()),
            ("Rules",      lambda: cls.validate_rules()),
            ("KG",         lambda: cls.validate_knowledge_graph()),
            ("Conflicts",  lambda: cls.validate_conflicts()),
            ("Explanations", lambda: cls.validate_explanations()),
        ]

        results = {}
        for name, fn in steps:
            result = fn()
            results[name] = result
            print()

        print("[VALIDATOR] ========================================")
        print("[VALIDATOR]   SUMMARY")
        print("[VALIDATOR] ========================================")
        all_passed = True
        for name, result in results.items():
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            n_issues = len([i for i in result["issues"] if not i.startswith("WARN:")])
            n_warns = len([i for i in result["issues"] if i.startswith("WARN:")])
            print(f"[VALIDATOR]   {status}  {name:<14}  errors={n_issues}  warnings={n_warns}")
            if not result["passed"]:
                all_passed = False

        print("[VALIDATOR] ========================================")
        if all_passed:
            print("[VALIDATOR] ✓ All pipeline steps passed validation.")
        else:
            print("[VALIDATOR] ⚠ One or more pipeline steps failed. Review issues above.")
        print("[VALIDATOR] ========================================")

        return results


if __name__ == "__main__":
    PipelineValidator.run_all()
