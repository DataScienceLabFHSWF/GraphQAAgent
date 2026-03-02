"""Unit tests for HITL-specific API endpoints."""

from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

from kgrag.api.server import app
from kgrag.api.hitl_schemas import QAResult, LowConfidenceReport, FeedbackRequest


@pytest.fixture
def client():
    return TestClient(app)


def test_report_low_confidence_filters_and_forwards(monkeypatch, client):
    called = {}

    def fake_forward(results):
        called['args'] = results
        return {'status': 'received', 'gaps_detected': 1}

    monkeypatch.setattr('kgrag.api.dependencies.forward_to_kgbuilder', fake_forward)

    qa = QAResult(question='q', answer='a', confidence=0.3, session_id='s')
    payload = LowConfidenceReport(qa_results=[qa], threshold=0.5)
    resp = client.post('/api/v1/hitl/report-low-confidence', json=payload.model_dump())
    assert resp.status_code == 200
    data = resp.json()
    assert data['status'] == 'received'
    assert data['gaps_detected'] == 1
    # forwarded arguments should match qa list
    assert isinstance(called.get('args'), list)


def test_report_low_confidence_returns_when_none(monkeypatch, client):
    qa = QAResult(question='q', answer='a', confidence=0.9, session_id='s')
    payload = LowConfidenceReport(qa_results=[qa], threshold=0.5)
    resp = client.post('/api/v1/hitl/report-low-confidence', json=payload.model_dump())
    assert resp.status_code == 200
    assert 'No results below threshold' in resp.json()['message']


def test_feedback_endpoint(client):
    req = FeedbackRequest(session_id='s', turn_index=0, correction='fix', feedback_type='correction')
    resp = client.post('/api/v1/feedback', json=req.model_dump())
    assert resp.status_code == 200
    assert resp.json()['status'] == 'accepted'
    assert resp.json()['feedback_id'].startswith('fb_')
