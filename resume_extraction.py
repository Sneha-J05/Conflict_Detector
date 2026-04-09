"""
Resume rule extraction for articles that previously returned 0 rules.
Uses a stricter prompt variant and more aggressive JSON repair.
"""
import json
import os
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------------------
# Stricter prompt — reinforces JSON-only output more explicitly
# ---------------------------------------------------------------------------

STRICT_SYSTEM = (
    "You are a legal rule extractor. "
    "OUTPUT ONLY VALID JSON. "
    "Do NOT write any explanation, prose, or markdown. "
    "Start your response with '{' and end with '}'."
)

STRICT_USER_TEMPLATE = """You MUST respond with ONLY valid JSON and nothing else.

Extract all legal rules from the article below. Use exactly this JSON schema:

{{
  "article_id": "{article_id}",
  "source": "{source}",
  "article_number": {article_number},
  "article_type": "<core_obligation|fundamental_right|enforcement|administrative|definition|exception>",
  "rules": [
    {{
      "rule_id": "{article_id}_R1",
      "entity": "<who the rule applies to>",
      "action": "<what action is required/permitted/prohibited>",
      "condition": "<under what conditions>",
      "modality": "<obligation|permission|prohibition|exception>",
      "scope": "<scope of application>",
      "references": ["<other article IDs mentioned>"],
      "raw_text": "<the original sentence this rule comes from>"
    }}
  ]
}}

IMPORTANT:
- Return ONLY the JSON object. No prose before or after.
- If the article has multiple obligations, add multiple items to the "rules" array.
- If you truly cannot extract any rule, return an empty array: "rules": []
- references should be formatted like ["GDPR_Art7"] — empty list [] if none mentioned.

Article ID: {article_id}
Source: {source}

Article text:
{article_text}"""


def _call_ollama(system: str, user: str, model: str = "mistral") -> str:
    import ollama
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp["message"]["content"]


def _call_huggingface(system: str, user: str,
                      model: str = "mistralai/Mistral-7B-Instruct-v0.2") -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(model)
    resp = client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=1500,
    )
    return resp.choices[0].message.content


def _llm_call(system: str, user: str) -> str:
    """Try Ollama first, fall back to HuggingFace if unavailable."""
    try:
        return _call_ollama(system, user)
    except Exception as ollama_err:
        if "connect" in str(ollama_err).lower() or "connection" in str(ollama_err).lower():
            print("  [INFO] Ollama unavailable — using HuggingFace fallback")
            return _call_huggingface(system, user)
        raise


def _repair_json(text: str) -> str:
    """
    Try multiple strategies to isolate valid JSON from an LLM response
    that may contain prose preambles or markdown fences.
    """
    # Remove markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()

    # If LLM prefixed with prose, find the first '{'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return text  # Return as-is and let json.loads raise


def extract_article(article: dict, retries: int = 3) -> dict | None:
    """Extract rules from a single article with strict prompt and retries."""
    user_prompt = STRICT_USER_TEMPLATE.format(
        article_id=article["id"],
        source=article["source"],
        article_number=article["article"],
        article_text=article["text"][:3000],
    )

    for attempt in range(retries):
        raw = ""
        try:
            raw = _llm_call(STRICT_SYSTEM, user_prompt)
            repaired = _repair_json(raw)
            parsed = json.loads(repaired)

            parsed.setdefault("article_id", article["id"])
            parsed.setdefault("source", article["source"])
            parsed.setdefault("article_number", article["article"])
            parsed.setdefault("article_type", "core_obligation")
            if "rules" not in parsed:
                parsed["rules"] = []

            return parsed

        except Exception as e:
            print(f"    [attempt {attempt + 1}/{retries}] Failed: {e}")
            if attempt == 0 and raw:
                print(f"    Raw output snippet: {repr(raw[:300])}")

    return None


def resume_extraction():
    # Load current state
    with open(os.path.join(DATA_DIR, "rules.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("articles", [])
    successful = [a for a in articles if len(a.get("rules", [])) > 0]
    failed = [a for a in articles if len(a.get("rules", [])) == 0]

    print(f"Current state: {len(successful)} articles WITH rules, {len(failed)} with 0 rules.")
    print(f"Re-extracting {len(failed)} articles...\n")

    # Load corpus for article text
    with open(os.path.join(DATA_DIR, "corpus.json"), "r", encoding="utf-8") as f:
        corpus = json.load(f)
    corpus_dict = {a["id"]: a for a in corpus}

    newly_extracted = []
    fixed_count = 0

    for idx, f_art in enumerate(failed, 1):
        art_id = f_art["article_id"]
        print(f"[{idx}/{len(failed)}] {art_id} ...", end=" ", flush=True)

        full_article = corpus_dict.get(art_id)
        if not full_article:
            print(f"NOT IN CORPUS — skipping")
            newly_extracted.append(f_art)
            continue

        result = extract_article(full_article)
        rule_count = len(result.get("rules", [])) if result else 0

        if result and rule_count > 0:
            newly_extracted.append(result)
            fixed_count += 1
            print(f"OK: {rule_count} rule(s)")
        else:
            print("FAIL: still 0 rules")
            newly_extracted.append(f_art)  # keep the empty placeholder

    # Merge: successful + newly attempted
    final_articles = successful + newly_extracted

    # Flatten rules
    all_rules = []
    for art in final_articles:
        for rule in art.get("rules", []):
            rule["source"] = art.get("source", rule.get("source", ""))
            rule["article_number"] = art.get("article_number", rule.get("article_number"))
            rule["article_type"] = art.get("article_type", "core_obligation")
            # Ensure weights stub exists (used downstream)
            if "weights" not in rule:
                rule["weights"] = {
                    "source": None,
                    "article": None,
                    "severity": None,
                    "composite": None,
                    "lex_specialis": False,
                }
            all_rules.append(rule)

    output = {"articles": final_articles, "rules": all_rules}

    out_path = os.path.join(DATA_DIR, "rules.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"DONE — Fixed {fixed_count}/{len(failed)} previously-empty articles.")
    print(f"Total rules now: {len(all_rules)} from {len(final_articles)} articles.")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    resume_extraction()
