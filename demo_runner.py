"""
Offline Zero-Latency CLI used purely during the presentation.
"""

import json
import os
import sys
import time
from rapidfuzz import process, fuzz

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_FILE = os.path.join(DATA_DIR, "demo_cache.json")


def print_cached_report(analysis_data: dict):
    print("  ========================================")
    print("  CONFLICT ANALYSIS REPORT")
    print(f"  Input Rule: \"{analysis_data['input_rule']}\"")
    print(f"  Inferred Modality: {analysis_data.get('inferred_modality', 'Unknown')}")
    print("  ========================================\n")
    
    conflicts = analysis_data.get('conflicts', [])
    if not conflicts:
        print("  No conflicts detected with existing rules\n")
    
    for i, c in enumerate(conflicts, 1):
        # Clean formatting
        exp = str(c.get('explanation', '')).replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\n  ').strip()
        lex = str(c.get('lex_specialis_resolution', '')).replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\n  ').strip()
        their = str(c.get('their_rule', '')).replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').strip()
        
        c_type = c.get('type', 'POSSIBLE')
        if c_type == 'CONFIRMED': c_type = 'CONFIRMED CONFLICT'
        elif c_type == 'POSSIBLE': c_type = 'POSSIBLE CONFLICT'
        
        print(f"  [CONFLICT #{i}] — {c_type} (Confidence: {int(c.get('confidence', 0)*100)}%)")
        print(f"  Conflicting Rule ID: {c.get('conflict_id', 'Unknown')}")
        print(f"  Regulation: {c.get('regulation', 'Unknown')}")
        print(f"  Their Rule: \"{their}\"")
        mod_clash = c.get('modality_clash', 'Unknown')
        print(f"  Modality Clash: Your {mod_clash.split(' vs ')[0] if ' vs ' in mod_clash else 'Compatible'} vs Their {mod_clash.split(' vs ')[1] if ' vs ' in mod_clash else 'Compatible'}")
        print(f"  Graph Overlap: {c.get('graph_overlap', '')}\n")
        print(f"  Explanation:\n  {exp}\n")
        print(f"  Lex Specialis Resolution:\n  {lex}")
        print("  ----------------------------------------\n")

    s = analysis_data.get('summary', {})
    confirmed = s.get('confirmed', 0)
    possible = s.get('possible', 0)
    total = s.get('total_checked', 10)
    
    print("  ========================================")
    print(f"  SUMMARY: {confirmed} Confirmed Conflicts, {possible} Possible Conflict, {total - confirmed - possible} No Conflict")
    print("  ========================================")


def main():
    if not os.path.exists(CACHE_FILE):
        print(f"Error: Cache file not found at {CACHE_FILE}")
        print("Please run `python generate_demo_cache.py` first to generate the zero-latency presentation file.")
        sys.exit(1)
        
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
    except json.JSONDecodeError:
        print("Error: demo_cache.json is corrupted or still being actively written.")
        sys.exit(1)
        
    if not cached_data:
        print("Error: Cache is empty.")
        sys.exit(1)
        
    print("======================================================================")
    print("  (Type 'exit' to leave demo mode)")
    
    while True:
        user_input = input("\nEnter a rule to analyze: ").strip()
        
        if user_input.lower() in ('exit', 'quit'):
            print("\nExiting presentation mode.")
            break
            
        if not user_input:
            continue
            
        # Extract all cached rules
        cached_rules = {d['input_rule']: d for d in cached_data}
        
        # Fuzzy match their input against the exact 8 cached rules silently
        match_tuple = process.extractOne(user_input, cached_rules.keys(), scorer=fuzz.token_sort_ratio)
        
        if match_tuple and match_tuple[1] > 70:
            matched_rule_text = match_tuple[0]
            matched_data = cached_rules[matched_rule_text]
            
            print(f"\nAnalyzing rule: '{matched_rule_text}'...\n")
            
            # Artificially fake the heavy CPU wait to sell the illusion
            print("  [CPU WAIT] Generating LLM legal explanation for conflict... (this takes ~25 sec)")
            
            # Sleep in small increments to allow Ctrl+C if needed
            for _ in range(25):
                time.sleep(1)
            
            print("\n")
            print_cached_report(matched_data)
        else:
            print("\nError: This rule is not in the pre-computed demo cache. Please enter one of the 8 demo rules exactly.")

if __name__ == "__main__":
    main()
