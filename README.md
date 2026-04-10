### High Level Architecture Diagram
```mermaid
flowchart LR

A[GDPR Regulation] --> B[Scraper]
A2[ePrivacy Regulation] --> B

B --> C[Rule Extractor]
C --> D[Weight Scorer]
D --> E[Knowledge Graph Builder]
E --> F[Embedder]

F --> G[Conflict Detector]
G --> H[Explainer]

H --> I[Corpus JSON]
H --> J[ChromaDB]
H --> K[Knowledge Graph File]
H --> L[Rules JSON]

J --> M[Vector Search]
K --> N[Knowledge Graph Check]

M --> O[Conflict Classification]
N --> O

O --> P[LLM Explanation]

P --> Q[CLI Interface]
P --> R[Web Interface]
