# Automated Conflict Detection in Bureaucratic Rule Systems Using Intelligent Analysis

## Overview

This project presents an automated system for detecting conflicts in regulatory and bureaucratic rule systems using Large Language Models (LLMs), semantic embeddings, and knowledge graphs.

It processes legal documents, extracts structured rules, identifies contradictions across policies, and generates human-readable explanations to support regulatory compliance.

---

## Problem Statement

Regulatory frameworks such as GDPR and Privacy Directives are complex and often overlap. When multiple policies interact, they may produce:

- Contradictions
- Ambiguities
- Compliance conflicts

Existing systems analyze rules in isolation and fail to capture cross-document interactions.

---

## Features

- Automated rule extraction using LLMs
- Semantic similarity-based conflict detection
- Knowledge graph-based validation
- Explainable conflict reasoning
- Scalable pipeline for large document sets
- Interactive query interface

---

## System Architecture

The system follows a modular pipeline:

1. **Policy Scraping** – Collects and preprocesses legal documents
2. **Rule Extraction** – Extracts structured rule components
3. **Weight Scoring** – Assigns importance based on legal precedence
4. **Knowledge Graph Construction** – Builds relationships between rules
5. **Semantic Embedding** – Converts rules into vector representations
6. **Conflict Detection** – Identifies and validates conflicts
7. **Explanation Generation** – Produces human-readable explanations

---

## Tech Stack

- Python
- Large Language Models (LLMs)
- ChromaDB (vector database)
- NetworkX (knowledge graphs)
- Flask / Web Interface (for interaction)

---

## Results

**Processed:**
- 144 GDPR Articles
- 144 Privacy Directive Articles

**Extracted:**
- 508 structured rules

**Successfully identified:**
- Data retention conflicts
- Cookie consent inconsistencies
- Ambiguous policy overlaps

**Achieved high accuracy with no false conflict classifications in evaluated samples.**

---

## Project Structure
project-root/
│
├── data/                 # Raw and processed documents
├── pipeline/             # Core modules
│   ├── scraper.py
│   ├── rule_extractor.py
│   ├── weight_scorer.py
│   ├── graph_builder.py
│   ├── embedder.py
│   ├── conflict_detector.py
│   └── explainer.py
│
├── interface/            # Query and UI layer
│   ├── query_interface.py
│   └── demo_runner.py
│
├── docs/
│   └── ieee_draft.pdf
│
├── requirements.txt
└── README.md

---

## Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

---

## Usage

### Run the Pipeline

```bash
python pipeline/main.py
```

### Run Demo Interface

```bash
python interface/demo_runner.py
```

### Example Workflow

1. Input legal documents
2. Extract structured rules
3. Generate embeddings
4. Detect conflicts
5. View explanations in UI

---

## Limitations

- Depends on LLM accuracy
- CPU-based inference may be slow
- Limited handling of highly ambiguous legal text

---

## Future Work

- Improve real-time performance
- Enhance conflict classification models
- Extend to additional regulatory domains
- Integrate live compliance monitoring

---

## Contributors

- Sneha J
- Snisha Veluvolu
- Ritu Reddy P

---

## License
This project is for academic and research purposes.

This project is for academic and research purposes.
