"""HITL (Human-in-the-Loop) module — feedback, KG versioning, change tracking.

This module handles:

1. **KG Versioning** — temporal properties + change event log for tracking
   all HITL-driven mutations to the knowledge graph.
2. **Change Proposals** — structured review workflow for expert corrections.
3. **n10s (neosemantics) integration** — RDF export of KG snapshots and
   SHACL validation of proposed changes.
4. **Gap Detection** — identifies low-confidence or unanswerable questions
   that signal missing knowledge.

See ``DESIGN_NOTES`` below for the versioning strategy discussion.

DESIGN NOTES — KG Versioning Strategy
======================================

**Neosemantics (n10s)** is *not* a versioning system. It is an RDF ↔ Neo4j
bridge that handles:
- Lossless RDF import/export (Turtle, JSON-LD, RDF/XML)
- Ontology import (``n10s.onto.import.fetch``)
- SHACL validation (``n10s.validation.shacl.validate``)
- Named graph support (RDF contexts → Neo4j labels)

However, n10s is *very useful* in a HITL versioning pipeline:

1. **SHACL validation** — before accepting a change proposal, validate it
   against the domain ontology's SHACL shapes.
2. **RDF snapshot export** — export a validated KG state to Turtle for
   archival / diff.
3. **Named graphs** — map provenance contexts (who changed what and when)
   to RDF named graphs.

**Recommended versioning approach** (implemented below as stubs):

- **Temporal properties** on every node/edge: ``_version``, ``_valid_from``,
  ``_valid_to``, ``_modified_by``, ``_change_event_id``.
- **ChangeEvent nodes** in Neo4j — an audit trail of every mutation with
  before/after snapshots.
- **Soft deletes** — retired triples get ``_valid_to`` set rather than being
  removed.
- **Branch-like workflow** — changes go through ``proposed → validated →
  accepted`` states before modifying the live graph.

Alternative approaches considered:
- Neo4j multi-database (one DB per version) — too heavy for incremental edits.
- Git-for-graphs (e.g. TerminusDB) — better for full-graph versioning but
  adds infra complexity.
"""
