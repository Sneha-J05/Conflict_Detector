import json

with open("data/rules.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Keep track of seen IDs
seen_ids = set()
new_rules = []
duplicates_found = 0

for rule in data.get("rules", []):
    rid = rule.get("rule_id", "")
    if not rid:
        rid = "Unknown_R"
        
    if rid in seen_ids:
        # Generate new unique ID
        i = 1
        while f"{rid}_{i}" in seen_ids:
            i += 1
        new_rid = f"{rid}_{i}"
        print(f"Renamed duplicate {rid} to {new_rid}")
        rule["rule_id"] = new_rid
        seen_ids.add(new_rid)
        duplicates_found += 1
    else:
        seen_ids.add(rid)
        
    new_rules.append(rule)

if duplicates_found > 0:
    data["rules"] = new_rules
    with open("data/rules.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Fixed {duplicates_found} duplicate IDs.")
else:
    print("No duplicates.")
