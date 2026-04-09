import json

try:
    with open('data/rules.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    articles = data.get('articles', [])
    has_rules = [a for a in articles if len(a.get('rules', [])) > 0]
    no_rules = [a for a in articles if len(a.get('rules', [])) == 0]

    print(f"Total articles: {len(articles)}")
    print(f"Articles with rules: {len(has_rules)}")
    print(f"Articles with 0 rules: {len(no_rules)}")

    sources_with = {}
    for a in has_rules:
        sources_with[a.get('source', '')] = sources_with.get(a.get('source', ''), 0) + 1
    print(f"Sources WITH rules: {sources_with}")

    sources_without = {}
    for a in no_rules:
        sources_without[a.get('source', '')] = sources_without.get(a.get('source', ''), 0) + 1
    print(f"Sources WITHOUT rules: {sources_without}")

except Exception as e:
    print(f"Error reading rules.json: {e}")

print("\n--- LLM Logs Analysis ---")
try:
    failures = 0
    errors_sample = {}
    with open('data/llm_logs.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                log = json.loads(line)
                if log.get('status') == 'error' or 'error' in log:
                    failures += 1
                    err = log.get('error', str(log))
                    # Group by first 50 chars of error
                    short_err = str(err)[:100].replace('\n', ' ')
                    errors_sample[short_err] = errors_sample.get(short_err, 0) + 1
            except:
                pass
    print(f"Total logged errors: {failures}")
    if errors_sample:
        print("Most common errors (up to 100 chars):")
        for err, count in sorted(errors_sample.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {count}x: {err}")
except Exception as e:
    print(f"Error reading llm_logs.jsonl: {e}")
