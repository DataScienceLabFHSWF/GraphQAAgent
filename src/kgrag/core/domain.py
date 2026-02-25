"""Domain configuration loader — single YAML drives all domain-specific text.

To adapt the KG-RAG Agent to a new domain, edit ``config/domain.yaml``.
All prompts, vocabulary hints, label mappings, and example data are loaded
from there so the Python code stays domain-neutral.

Usage::

    from kgrag.core.domain import DomainConfig

    domain = DomainConfig.load()           # loads config/domain.yaml
    domain = DomainConfig.load("path.yaml") # or a custom path

    prompt = domain.render_prompt("agentic_system",
        ontology_summary="...",
        neo4j_schema="...",
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "domain.yaml"


@dataclass
class VocabularyEntry:
    """Single term ↔ translation pair."""

    term: str
    translation: str


@dataclass
class DomainConfig:
    """Parsed domain configuration loaded from YAML.

    Fields mirror the ``domain:`` and ``prompts:`` sections in the YAML.
    Call :meth:`render_prompt` to get a fully-interpolated prompt string.
    """

    # ── Domain metadata ────────────────────────────────────────────────
    name: str = "Knowledge Graph"
    description: str = ""
    language: str = "en"
    vocabulary: list[VocabularyEntry] = field(default_factory=list)
    data_model_notes: str = ""
    neo4j_label_mapping: dict[str, str] = field(default_factory=dict)
    cypher_patterns: list[str] = field(default_factory=list)
    example_entity_types: list[str] = field(default_factory=list)
    demo_questions: list[str] = field(default_factory=list)

    # ── Prompt templates (raw, with {placeholders}) ────────────────────
    prompts: dict[str, str] = field(default_factory=dict)

    # -- Loading --------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "DomainConfig":
        """Load domain config from a YAML file.

        Resolution order:
        1. Explicit *path* argument
        2. ``KGRAG_DOMAIN_CONFIG`` environment variable
        3. ``config/domain.yaml`` relative to the repo root
        """
        if path is None:
            path = os.environ.get("KGRAG_DOMAIN_CONFIG", str(_DEFAULT_CONFIG_PATH))
        path = Path(path)

        if not path.exists():
            logger.warning("domain_config.not_found", path=str(path))
            return cls()  # sensible defaults

        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        if not isinstance(raw, dict):
            logger.warning("domain_config.invalid_yaml", path=str(path))
            return cls()

        domain_raw = raw.get("domain", {})
        prompts_raw = raw.get("prompts", {})

        vocab = [
            VocabularyEntry(term=v["term"], translation=v["translation"])
            for v in domain_raw.get("vocabulary", [])
            if isinstance(v, dict) and "term" in v and "translation" in v
        ]

        cfg = cls(
            name=domain_raw.get("name", "Knowledge Graph"),
            description=domain_raw.get("description", ""),
            language=domain_raw.get("language", "en"),
            vocabulary=vocab,
            data_model_notes=domain_raw.get("data_model_notes", "").strip(),
            neo4j_label_mapping=domain_raw.get("neo4j_label_mapping", {}),
            cypher_patterns=domain_raw.get("cypher_patterns", []),
            example_entity_types=domain_raw.get("example_entity_types", []),
            demo_questions=domain_raw.get("demo_questions", []),
            prompts={k: v.strip() for k, v in prompts_raw.items()},
        )
        logger.info(
            "domain_config.loaded",
            path=str(path),
            domain=cfg.name,
            prompts=list(cfg.prompts.keys()),
            vocab_entries=len(cfg.vocabulary),
        )
        return cfg

    # -- Prompt rendering -----------------------------------------------

    @property
    def vocabulary_block(self) -> str:
        """Format the vocabulary list for prompt injection."""
        if not self.vocabulary:
            return ""
        lines = [f"DOMAIN VOCABULARY (the data is primarily in {self.language.upper()}):"]
        for v in self.vocabulary:
            lines.append(f"  {v.term} = {v.translation}")
        return "\n".join(lines)

    @property
    def example_types_str(self) -> str:
        """Quoted, comma-separated example entity types."""
        if not self.example_entity_types:
            return '"Entity", "Concept"'
        return ", ".join(f'"{t}"' for t in self.example_entity_types)

    def render_prompt(
        self,
        prompt_key: str,
        **kwargs: Any,
    ) -> str:
        """Render a named prompt template with domain + caller variables.

        Standard domain variables are always available:
        - ``{domain_name}``
        - ``{vocabulary_block}``
        - ``{data_model_notes}``
        - ``{example_types}``

        Callers pass additional variables (e.g. ``ontology_summary``,
        ``neo4j_schema``) as keyword arguments.

        Unknown placeholders are left as-is (useful for prompts that still
        contain LangChain ``{schema}`` / ``{question}`` variables).
        """
        template = self.prompts.get(prompt_key, "")
        if not template:
            logger.warning("domain_config.missing_prompt", key=prompt_key)
            return ""

        variables: dict[str, str] = {
            "domain_name": self.name,
            "vocabulary_block": self.vocabulary_block,
            "data_model_notes": self.data_model_notes,
            "example_types": self.example_types_str,
        }
        variables.update(kwargs)

        # Safe format: only replace known keys, leave others untouched
        result = template
        for key, value in variables.items():
            result = result.replace("{" + key + "}", str(value))

        return result

    def get_prompt_raw(self, prompt_key: str) -> str:
        """Get the raw template string without rendering."""
        return self.prompts.get(prompt_key, "")
