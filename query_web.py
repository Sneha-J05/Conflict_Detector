"""
Web Interface wrapper for the Policy Conflict Detector Query Engine.

Usage:
  python query_web.py
"""

from flask import Flask, request, jsonify
from query_interface import QueryInterface
import sys
import os

app = Flask(__name__)
interface = None

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'rule' not in data:
        return jsonify({"error": "Missing 'rule' in JSON body"}), 400
        
    rule = data['rule']
    print(f"[WEB] Analyzing rule: '{rule}'")
    
    result = interface.analyze(rule)
    
    # Strip Internal properties out of the dictionary responses before sending
    for c in result.get('conflicts', []):
        c.pop('_raw_type', None)
        c.pop('_raw_conf', None)
        
    return jsonify(result), 200

if __name__ == "__main__":
    try:
        print("Initializing Query Interface... this may take a moment.")
        interface = QueryInterface()
        print("Interface ready.")
    except Exception as e:
        print(f"Failed to initialize interface: {e}")
        sys.exit(1)
        
    app.run(host="0.0.0.0", port=5000, debug=False)
