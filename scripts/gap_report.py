"""Simple demo: run a question and print any ontology gaps detected.

Usage:
    python scripts/gap_report.py "Why is FuelRod important?"

This script assumes your services (Neo4j, Qdrant, Fuseki, Ollama) are
running and the environment variables in ``.env`` are configured. The
underlying retriever is ``AgenticGraphRAG``; after a retrieval the
``_detected_gaps`` list is examined and any proposals are generated via
``OntologyProposalGenerator``.
"""

import sys
import asyncio

from kgrag.connectors.neo4j import Neo4jConnector
from kgrag.connectors.qdrant import QdrantConnector
from kgrag.connectors.langchain_ollama_provider import LangChainOllamaProvider
from kgrag.core.config import Neo4jConfig, QdrantConfig, OllamaConfig
from kgrag.retrieval.agentic_rag import AgenticGraphRAG
from kgrag.core.domain import DomainConfig
from kgrag.retrieval.ontology_context import OntologyContext
from kgrag.retrieval.active_ontology import OntologyProposalGenerator
from kgrag.core.models import QAQuery


def load_connectors():
    neo4j_cfg = Neo4jConfig.load()
    qdrant_cfg = QdrantConfig.load()
    ollama_cfg = OllamaConfig.load()

    neo4j = Neo4jConnector(neo4j_cfg)
    qdrant = QdrantConnector(qdrant_cfg)
    ollama = LangChainOllamaProvider(ollama_cfg)
    return neo4j, qdrant, ollama


async def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "What is FuelRod?"

    neo4j, qdrant, ollama = load_connectors()
    ontology = await OntologyContext.load()  # from Fuseki
    domain = DomainConfig.load()

    retriever = AgenticGraphRAG(
        neo4j=neo4j,
        neo4j_config=Neo4jConfig.load(),
        qdrant=qdrant,
        ollama=ollama,
        ontology_context=ontology,
        domain_config=domain,
    )

    query = QAQuery(raw_question=question)
    contexts = await retriever.retrieve(query)

    if retriever._detected_gaps:
        print("== Detected ontology gaps ==")
        generator = OntologyProposalGenerator(ontology)
        for gap in retriever._detected_gaps:
            proposal = generator.generate_proposal(gap)
            print("- gap:", gap.gap_type, gap.query_context)
            print("  proposal type:", proposal.proposal_type)
            print(proposal.turtle_fragment)
    else:
        print("No ontology gaps detected for this question.")


if __name__ == "__main__":
    asyncio.run(main())
