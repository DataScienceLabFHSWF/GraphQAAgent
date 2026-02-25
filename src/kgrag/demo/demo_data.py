"""Pre-loaded demo scenarios and expected behaviour.

Each scenario groups related questions that showcase specific GraphQA
capabilities (multi-hop reasoning, law graph traversal, entity lookup, etc.).
"""

from __future__ import annotations

DEMO_SCENARIOS: list[dict] = [
    {
        "title": "Nuclear Decommissioning Legal Framework",
        "description": "Explore the legal landscape governing nuclear facility decommissioning in Germany.",
        "questions": [
            "Welche Gesetze regeln den Rückbau von Kernkraftwerken in Deutschland?",
            "Which paragraphs of the AtG cover decommissioning permits?",
            "What is the relationship between AtG and StrlSchG for decommissioning?",
        ],
        "expected_features": [
            "multi-hop reasoning",
            "law graph traversal",
            "cross-references",
        ],
    },
    {
        "title": "Waste Management Chain",
        "description": "Trace the waste processing pipeline through legal and technical concepts.",
        "questions": [
            "Which types of radioactive waste are distinguished in the StrlSchG?",
            "What processes are required for waste disposal according to KrWG?",
        ],
        "expected_features": [
            "entity enumeration",
            "process chain",
            "legal provisions",
        ],
    },
    {
        "title": "Facility-Specific Queries",
        "description": "Query specific facilities and the permits they require.",
        "questions": [
            "Which nuclear facilities are mentioned in the knowledge graph?",
            "What permits does a nuclear facility need for decommissioning?",
        ],
        "expected_features": [
            "entity lookup",
            "relation traversal",
            "type hierarchy",
        ],
    },
    {
        "title": "Ontology Exploration",
        "description": "Test ontology-aware query expansion and class hierarchy navigation.",
        "questions": [
            "What types of entities exist in the domain ontology?",
            "Which ontology classes are related to radiation protection?",
        ],
        "expected_features": [
            "ontology expansion",
            "class hierarchy",
            "SPARQL integration",
        ],
    },
]
