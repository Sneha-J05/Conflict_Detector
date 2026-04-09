"""
Step 2: LLM-based Rule Extraction
Sends each article to Mistral-7B, parses structured rules.
Supports Ollama (local) and HuggingFace InferenceClient backends.
"""

import os
import re
import json
import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------------------
# LLM backend helpers
# ---------------------------------------------------------------------------

def _call_ollama(system: str, user: str, model: str = "mistral") -> str:
    """Call local Ollama server."""
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
    """Call HuggingFace Inference API."""
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


def llm_call(system: str, user: str, backend: str = "ollama") -> str:
    """Unified LLM call with backend selection."""
    if backend == "ollama":
        return _call_ollama(system, user)
    elif backend == "huggingface":
        return _call_huggingface(system, user)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_llm_call(step: str, input_text: str, output_text: str):
    """Append to data/llm_logs.jsonl."""
    log_path = os.path.join(DATA_DIR, "llm_logs.jsonl")
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": step,
        "input": input_text[:500],
        "output": output_text[:500],
        "tokens": len(output_text.split()),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Try to extract JSON from LLM output that may contain markdown fences."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.strip("`").strip()
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


# ---------------------------------------------------------------------------
# Rule Extractor
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a legal rule extractor. Extract structured rules from policy text. "
    "Return ONLY valid JSON. No explanation, no markdown, no preamble."
)

USER_PROMPT_TEMPLATE = """Extract all rules from this article. Return JSON in exactly this format:
{{
  "article_id": "{article_id}",
  "source": "{source}",
  "article_number": {article_number},
  "article_type": "<one of: core_obligation, fundamental_right, enforcement, administrative, definition, exception>",
  "rules": [
    {{
      "rule_id": "{article_id}_R1",
      "entity": "data controller",
      "action": "process personal data",
      "condition": "at least one lawful basis exists",
      "modality": "<one of: obligation, permission, prohibition, exception>",
      "scope": "all processing activities",
      "references": ["GDPR_Art7", "GDPR_Art9"],
      "raw_text": "original sentence here"
    }}
  ]
}}

article_type must be one of:
  core_obligation, fundamental_right, enforcement, administrative, definition, exception

modality must be one of:
  obligation, permission, prohibition, exception

references: list of other articles this rule explicitly mentions
  (e.g. "as referred to in Article 7" → ["GDPR_Art7"])
  Return [] if none.

Article text:
{article_text}"""


class RuleExtractor:
    """Extract structured rules from policy articles via Mistral-7B."""

    def __init__(self, backend: str = "ollama"):
        self.backend = backend
        os.makedirs(DATA_DIR, exist_ok=True)

    def extract_rules_from_article(self, article: dict) -> dict | None:
        """Send one article to the LLM and parse the response."""
        user_prompt = USER_PROMPT_TEMPLATE.format(
            article_id=article["id"],
            source=article["source"],
            article_number=article["article"],
            article_text=article["text"][:3000],  # truncate very long articles
        )

        for attempt in range(2):  # retry once on failure
            try:
                raw = llm_call(SYSTEM_PROMPT, user_prompt, self.backend)
                _log_llm_call("rule_extraction", user_prompt[:300], raw[:300])
                parsed = _extract_json(raw)

                # Basic validation
                if "rules" not in parsed:
                    parsed["rules"] = []
                parsed.setdefault("article_id", article["id"])
                parsed.setdefault("source", article["source"])
                parsed.setdefault("article_number", article["article"])
                parsed.setdefault("article_type", "core_obligation")
                return parsed

            except Exception as e:
                print(f"  [EXTRACTOR] Attempt {attempt+1} failed for "
                      f"{article['id']}: {e}")

        # Return a minimal fallback so the pipeline doesn't break
        return {
            "article_id": article["id"],
            "source": article["source"],
            "article_number": article["article"],
            "article_type": "core_obligation",
            "rules": [],
        }

    def extract_all(self, corpus_path: str | None = None) -> list[dict]:
        """Process every article in corpus.json and save rules.json."""
        if corpus_path is None:
            corpus_path = os.path.join(DATA_DIR, "corpus.json")

        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        all_articles: list[dict] = []
        for idx, article in enumerate(corpus, 1):
            print(f"[EXTRACTOR] ({idx}/{len(corpus)}) {article['id']} …")
            result = self.extract_rules_from_article(article)
            if result:
                all_articles.append(result)

        # Flatten all rules for convenience
        all_rules = []
        for art in all_articles:
            for rule in art.get("rules", []):
                rule["source"] = art["source"]
                rule["article_number"] = art["article_number"]
                rule["article_type"] = art.get("article_type", "core_obligation")
                all_rules.append(rule)

        output = {
            "articles": all_articles,
            "rules": all_rules,
        }

        out_path = os.path.join(DATA_DIR, "rules.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"[EXTRACTOR] Done — {len(all_rules)} rules from "
              f"{len(all_articles)} articles → {out_path}")
        return all_rules

    def run(self):
        """Entry point."""
        self.extract_all()


if __name__ == "__main__":
    RuleExtractor().run()
