import json

with open('data/llm_logs.jsonl', 'r', encoding='utf-8') as f:
    logs = [json.loads(line) for line in f if line.strip()]

print(f"Total entries in llm_logs: {len(logs)}")

# Find logs for ePrivacy
epriv_logs = [l for l in logs if 'ePrivacy' in l.get('input', '') or 'EPRIVACY' in l.get('input', '')]
print(f"ePrivacy LLM calls: {len(epriv_logs)}")

if epriv_logs:
    print("\n--- Last ePrivacy LLM Output ---")
    print(epriv_logs[-1].get('output', 'No output'))

# Find logs for GDPR with 0 rules
gdpr_logs = [l for l in logs if 'GDPR' in l.get('input', '')]
empty_responses = [l for l in logs if '"rules": []' in l.get('output', '').replace(' ', '')]
print(f"Logs where LLM explicitly returned empty rules list: {len(empty_responses)}")
if empty_responses:
    print("\n--- Example Empty Return ---")
    print(empty_responses[-1].get('output', ''))
