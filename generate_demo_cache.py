"""
Offline script to securely pre-compute 8 complex rules through the heavy query interface.
"""
import json
import os
import sys

# Ensure this script works in the right directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from query_interface import QueryInterface

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_FILE = os.path.join(DATA_DIR, "demo_cache.json")

RULES_TO_CACHE = [
    # 5 Conflicts
    "Data controllers are permitted to process special categories of personal data revealing racial or ethnic origin without explicit consent.",
    "Companies can legally store cookies and track user location data indefinitely prior to obtaining informed consent.",
    "Data processors shall not provide the data subject with access to their personal data upon request.",
    "Automated decision-making that produces legal effects concerning the data subject is unconditionally authorized without human intervention.",
    "Personal data may be kept in a form which permits identification of data subjects for longer than is necessary for the purposes for which the personal data are processed.",
    # 3 Non-Conflicts
    "Organizations must notify the supervisory authority within 72 hours of becoming aware of a personal data breach.",
    "Mailing lists may only be used for direct marketing if the recipient has given explicit prior consent.",
    "A data protection officer shall be designated if the core activities consist of large scale regular and systematic monitoring of data subjects."
]

def main():
    print("[CACHE] Initializing strict QueryInterface...")
    interface = QueryInterface(backend="ollama")
    
    cached_data = []
    total = len(RULES_TO_CACHE)
    
    for i, rule in enumerate(RULES_TO_CACHE, 1):
        print(f"\n[CACHE] ({i}/{total}) Pre-computing rule: '{rule}'")
        try:
            res = interface.analyze(rule)
            
            # Remove any raw variables to make the JSON cleaner globally
            for c in res.get('conflicts', []):
                c.pop('_raw_type', None)
                c.pop('_raw_conf', None)
                
            cached_data.append(res)
            print(f"   -> Result cached! Found {res.get('summary', {}).get('confirmed', 0)} confirmed conflicts and {res.get('summary', {}).get('possible', 0)} possible overaps.")
        except Exception as e:
            print(f"   -> [ERROR] Skipping rule due to exception: {e}")
            
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cached_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n[CACHE] Successfully pre-computed {len(cached_data)} rules and generated clean cache: {CACHE_FILE}")

if __name__ == "__main__":
    main()
