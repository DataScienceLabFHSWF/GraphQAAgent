"""CLI entry point."""

from __future__ import annotations

import sys


def main() -> None:
    """Entry point for the ``kgrag`` command."""
    print("KG-RAG Agent CLI — use scripts/run_qa.py for interactive QA")
    print("  or: uvicorn kgrag.api.server:app --host 0.0.0.0 --port 8080")
    sys.exit(0)


if __name__ == "__main__":
    main()
