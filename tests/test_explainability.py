"""Tests for the explainability module (RAG + rule-based fallback)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from explainability.rag_explainer import (
    ClaimExplainer,
    _format_claim_summary,
    _retrieval_query,
    _rule_based_explanation,
)

HIGH_RISK = {
    "PREMIUM_AMOUNT": 150.0,
    "CLAIM_AMOUNT": 52000.0,
    "AGE": 28,
    "TENURE": 8,
    "NO_OF_FAMILY_MEMBERS": 1,
    "INCIDENT_HOUR_OF_THE_DAY": 3,
    "claim_to_premium_ratio": 346.67,
    "is_high_claim": 1,
    "days_loss_to_report": 35,
    "days_report_to_txn": 1,
    "policy_age_days": 45,
    "is_night_incident": 1,
    "is_major_loss": 1,
    "no_police_report": 1,
    "no_injury_high_claim": 1,
    "risk_encoded": 2,
}

LOW_RISK = {
    "PREMIUM_AMOUNT": 500.0,
    "CLAIM_AMOUNT": 1000.0,
    "AGE": 45,
    "TENURE": 60,
    "NO_OF_FAMILY_MEMBERS": 3,
    "INCIDENT_HOUR_OF_THE_DAY": 14,
    "claim_to_premium_ratio": 2.0,
    "is_high_claim": 0,
    "days_loss_to_report": 3,
    "days_report_to_txn": 2,
    "policy_age_days": 1800,
    "is_night_incident": 0,
    "is_major_loss": 0,
    "no_police_report": 0,
    "no_injury_high_claim": 0,
    "risk_encoded": 0,
}


# ── _rule_based_explanation ───────────────────────────────────────────────────

def test_rule_based_returns_string():
    result = _rule_based_explanation(HIGH_RISK, "HIGH", 0.87)
    assert isinstance(result, str) and len(result) > 20


def test_rule_based_contains_risk_level():
    assert "HIGH" in _rule_based_explanation(HIGH_RISK, "HIGH", 0.87)
    assert "LOW" in _rule_based_explanation(LOW_RISK, "LOW", 0.10)


def test_rule_based_mentions_ratio_when_extreme():
    result = _rule_based_explanation(HIGH_RISK, "HIGH", 0.87)
    assert "ratio" in result.lower()


def test_rule_based_no_reasons_still_returns_string():
    # All flags off — should fall back to the generic message
    neutral = {k: 0 for k in HIGH_RISK}
    neutral["claim_to_premium_ratio"] = 3.0
    neutral["policy_age_days"] = 500
    result = _rule_based_explanation(neutral, "LOW", 0.15)
    assert isinstance(result, str) and len(result) > 0


# ── _retrieval_query ──────────────────────────────────────────────────────────

def test_retrieval_query_non_empty():
    assert len(_retrieval_query(HIGH_RISK)) > 0
    assert len(_retrieval_query(LOW_RISK)) > 0


def test_retrieval_query_high_risk_has_keywords():
    query = _retrieval_query(HIGH_RISK)
    fraud_keywords = {"ratio", "fraude", "nocturno", "policial", "lesiones", "pérdida", "reciente"}
    assert any(kw in query for kw in fraud_keywords)


def test_retrieval_query_neutral_returns_fallback():
    neutral = {k: 0 for k in HIGH_RISK}
    neutral["claim_to_premium_ratio"] = 2.0
    neutral["policy_age_days"] = 500
    query = _retrieval_query(neutral)
    # Should return the generic fallback string
    assert "fraude" in query or "análisis" in query


# ── _format_claim_summary ─────────────────────────────────────────────────────

def test_format_claim_summary_contains_labels():
    summary = _format_claim_summary(HIGH_RISK)
    assert "Prima pagada" in summary
    assert "Monto reclamado" in summary
    assert "Incidente nocturno" in summary


def test_format_claim_summary_boolean_fields_humanized():
    summary = _format_claim_summary(HIGH_RISK)
    assert "Sí" in summary  # at least one flag is 1


# ── ClaimExplainer ────────────────────────────────────────────────────────────

def _make_mock_vectorstore():
    mock_doc = MagicMock()
    mock_doc.page_content = "claim_to_premium_ratio: ratio alto es señal de fraude."
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc]
    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever
    return mock_vs


def test_explainer_fallback_when_no_api_key():
    """Without ANTHROPIC_API_KEY the chain is None and explain() uses rule-based."""
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        explainer = ClaimExplainer(_make_mock_vectorstore())
        assert explainer._chain is None
        result = explainer.explain(HIGH_RISK, "HIGH", 0.87)
        assert isinstance(result, str) and len(result) > 0


def test_explainer_uses_chain_when_api_key_set():
    """With ANTHROPIC_API_KEY the chain is invoked and its output is returned."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Explicación generada por LLM."

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-key"}):
        with patch("explainability.rag_explainer.ChatAnthropic"):
            explainer = ClaimExplainer(_make_mock_vectorstore())
            explainer._chain = mock_chain

            result = explainer.explain(HIGH_RISK, "HIGH", 0.87)

    assert result == "Explicación generada por LLM."
    mock_chain.invoke.assert_called_once()


def test_explainer_invoke_passes_correct_keys():
    """chain.invoke receives query, claim_summary, risk_level and probability."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "ok"

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-key"}):
        with patch("explainability.rag_explainer.ChatAnthropic"):
            explainer = ClaimExplainer(_make_mock_vectorstore())
            explainer._chain = mock_chain
            explainer.explain(HIGH_RISK, "HIGH", 0.87)

    call_kwargs = mock_chain.invoke.call_args[0][0]
    assert "query" in call_kwargs
    assert "claim_summary" in call_kwargs
    assert "risk_level" in call_kwargs
    assert call_kwargs["risk_level"] == "HIGH"
    assert call_kwargs["probability"] == pytest.approx(0.87)
