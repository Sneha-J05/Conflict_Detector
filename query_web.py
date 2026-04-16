"""
Web Interface wrapper for the Policy Conflict Detector Query Engine.

Usage:
  python query_web.py
"""

from flask import Flask, request, jsonify, make_response
from query_interface import QueryInterface
import sys
import os
import json

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

    for c in result.get('conflicts', []):
        c.pop('_raw_type', None)
        c.pop('_raw_conf', None)

    return jsonify(result), 200


@app.route('/validate', methods=['GET'])
def validate():
    conflicts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "conflicts_raw.json")
    if not os.path.exists(conflicts_path):
        error_response = app.response_class(
            response=json.dumps({"error": "Conflicts data not found"}, ensure_ascii=False),
            status=404,
            mimetype='application/json; charset=utf-8'
        )
        error_response.headers['Access-Control-Allow-Origin'] = '*'
        return error_response

    try:
        with open(conflicts_path, "r", encoding="utf-8") as f:
            conflicts = json.load(f)

        if not isinstance(conflicts, list):
            conflicts = []

        total_conflicts = len(conflicts)
        composite_values = []
        confidence_values = []
        score_distribution = {"high": 0, "medium": 0, "low": 0}
        severity_consistency = {"consistent": 0, "inconsistent": 0}
        by_type = {}

        for conflict in conflicts:
            ctype = conflict.get("type", "")
            by_type[ctype] = by_type.get(ctype, 0) + 1

            composite_value = None
            confidence_value = None

            try:
                composite_raw = conflict.get("composite_score")
                if composite_raw is not None:
                    composite_value = float(composite_raw)
                    composite_values.append(composite_value)
                    if composite_value >= 0.65:
                        score_distribution["high"] += 1
                    elif composite_value >= 0.4:
                        score_distribution["medium"] += 1
                    else:
                        score_distribution["low"] += 1
            except (TypeError, ValueError):
                composite_value = None

            try:
                confidence_raw = conflict.get("confidence")
                if confidence_raw is not None:
                    confidence_value = float(confidence_raw)
                    confidence_values.append(confidence_value)
            except (TypeError, ValueError):
                confidence_value = None

            is_consistent = True
            if ctype == "direct_conflict":
                is_consistent = confidence_value is not None and confidence_value >= 0.7
            elif ctype == "conditional_conflict":
                is_consistent = confidence_value is not None and confidence_value >= 0.5
            elif ctype == "redundancy":
                is_consistent = composite_value is not None and composite_value <= 0.5
            elif ctype == "exception":
                is_consistent = composite_value is not None and composite_value <= 0.6
            elif ctype == "logical_inconsistency":
                is_consistent = confidence_value is not None and confidence_value >= 0.6

            if is_consistent:
                severity_consistency["consistent"] += 1
            else:
                severity_consistency["inconsistent"] += 1

        avg_composite_score = round(sum(composite_values) / len(composite_values), 2) if composite_values else 0.0
        avg_confidence = round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else 0.0

        result = {
            "total_conflicts": total_conflicts,
            "avg_composite_score": avg_composite_score,
            "avg_confidence": avg_confidence,
            "score_distribution": score_distribution,
            "severity_consistency": severity_consistency,
            "by_type": by_type,
        }
        response = app.response_class(
            response=json.dumps(result, ensure_ascii=False),
            status=200,
            mimetype='application/json; charset=utf-8'
        )
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except UnicodeEncodeError:
        error_response = app.response_class(
            response=json.dumps({"error": "Encoding error while generating validation response"}, ensure_ascii=False),
            status=500,
            mimetype='application/json; charset=utf-8'
        )
        error_response.headers['Access-Control-Allow-Origin'] = '*'
        return error_response


if __name__ == "__main__":
    try:
        print("Initializing Query Interface... this may take a moment.")
        interface = QueryInterface()
        print("Interface ready.")
    except Exception as e:
        print(f"Failed to initialize interface: {e}")
        sys.exit(1)

    app.run(host="0.0.0.0", port=5000, debug=False)
