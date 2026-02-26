"""Tests for the strategies endpoint and CORS configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

from kgrag.api.routes import STRATEGY_INFO, StrategyDetail, StrategiesResponse
from kgrag.api.server import _cors_origins, _DEFAULT_CORS_ORIGINS


class TestStrategyInfo:
    """Validate the strategy catalogue."""

    def test_all_strategies_present(self) -> None:
        expected = {"vector_only", "graph_only", "hybrid", "cypher", "agentic", "hybrid_sota"}
        assert set(STRATEGY_INFO.keys()) == expected

    def test_each_strategy_has_display_name_and_description(self) -> None:
        for sid, info in STRATEGY_INFO.items():
            assert "display_name" in info, f"{sid} missing display_name"
            assert "description" in info, f"{sid} missing description"
            assert len(info["display_name"]) > 0
            assert len(info["description"]) > 10

    def test_strategy_detail_model(self) -> None:
        s = StrategyDetail(id="hybrid", display_name="Hybrid Fusion", description="RRF fusion")
        assert s.id == "hybrid"
        d = s.model_dump()
        assert d["display_name"] == "Hybrid Fusion"

    def test_strategies_response_model(self) -> None:
        items = [
            StrategyDetail(id="hybrid", display_name="Hybrid", description="desc"),
        ]
        resp = StrategiesResponse(strategies=items, default="hybrid")
        assert resp.default == "hybrid"
        assert len(resp.strategies) == 1


class TestCORSConfiguration:
    """Validate configurable CORS origins."""

    def test_default_origins(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # Remove the var if present
            os.environ.pop("KGRAG_CORS_ORIGINS", None)
            origins = _cors_origins()
            assert origins == _DEFAULT_CORS_ORIGINS
            assert "http://localhost:3000" in origins

    def test_custom_origins(self) -> None:
        with patch.dict(os.environ, {"KGRAG_CORS_ORIGINS": "http://example.com,http://app.io"}):
            origins = _cors_origins()
            assert origins == ["http://example.com", "http://app.io"]

    def test_custom_origins_strips_whitespace(self) -> None:
        with patch.dict(os.environ, {"KGRAG_CORS_ORIGINS": " http://a.com , http://b.com "}):
            origins = _cors_origins()
            assert origins == ["http://a.com", "http://b.com"]

    def test_empty_env_var_falls_back(self) -> None:
        with patch.dict(os.environ, {"KGRAG_CORS_ORIGINS": ""}):
            origins = _cors_origins()
            assert origins == _DEFAULT_CORS_ORIGINS

    def test_wildcard_origin(self) -> None:
        with patch.dict(os.environ, {"KGRAG_CORS_ORIGINS": "*"}):
            origins = _cors_origins()
            assert origins == ["*"]
