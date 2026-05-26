"""
LangChain RAG chain for claim explainability.

Answers "¿por qué se marcó esta claim?" using:
  - FAISS knowledge base (feature descriptions + MLflow importances)
  - Claude Haiku as the generation model
  - Structured claim features as context

Falls back to a rule-based explanation if ANTHROPIC_API_KEY is not set.
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

EXPLANATION_PROMPT = ChatPromptTemplate.from_template(
    """Eres un analista experto en fraude de seguros. Explica por qué el modelo de ML \
marcó esta claim como {risk_level}.

CONOCIMIENTO DEL MODELO Y FEATURES:
{context}

DATOS DE LA CLAIM:
{claim_summary}

RESULTADO: {risk_level} — probabilidad de fraude: {probability_pct}

En 3-4 puntos concretos, explica:
- Qué features específicas de esta claim activaron la alerta
- Por qué esos valores son sospechosos según los patrones conocidos
- Si hay factores atenuantes, mencionarlos brevemente

Responde en español, sin jerga técnica de ML. El lector es un analista de negocios."""
)


def _format_claim_summary(features: dict) -> str:
    labels = {
        "PREMIUM_AMOUNT": "Prima pagada",
        "CLAIM_AMOUNT": "Monto reclamado",
        "AGE": "Edad del asegurado",
        "TENURE": "Antigüedad como cliente (meses)",
        "NO_OF_FAMILY_MEMBERS": "Miembros del grupo familiar",
        "INCIDENT_HOUR_OF_THE_DAY": "Hora del incidente",
        "claim_to_premium_ratio": "Ratio claim/prima",
        "is_high_claim": "Claim de alto monto",
        "days_loss_to_report": "Días entre incidente y reporte",
        "days_report_to_txn": "Días entre reporte y transacción",
        "policy_age_days": "Antigüedad de la póliza (días)",
        "is_night_incident": "Incidente nocturno (00h–05h)",
        "is_major_loss": "Pérdida mayor (Major Loss)",
        "no_police_report": "Sin reporte policial",
        "no_injury_high_claim": "Sin lesiones + claim alto",
        "risk_encoded": "Segmento de riesgo (0=L, 1=M, 2=H)",
    }
    lines = []
    for key, val in features.items():
        label = labels.get(key, key)
        if isinstance(val, float):
            lines.append(f"  {label}: {val:.4f}")
        elif isinstance(val, int) and key in ("is_night_incident", "is_major_loss", "no_police_report", "no_injury_high_claim", "is_high_claim"):
            lines.append(f"  {label}: {'Sí' if val else 'No'}")
        else:
            lines.append(f"  {label}: {val}")
    return "\n".join(lines)


def _retrieval_query(features: dict) -> str:
    tags = []
    if features.get("claim_to_premium_ratio", 0) > 10:
        tags.append("ratio claim prima muy alto fraude")
    if features.get("is_night_incident", 0):
        tags.append("incidente nocturno madrugada")
    if features.get("no_police_report", 0):
        tags.append("sin reporte policial sospechoso")
    if features.get("no_injury_high_claim", 0):
        tags.append("sin lesiones claim alto inconsistencia")
    if features.get("is_major_loss", 0):
        tags.append("pérdida mayor monto alto")
    if features.get("days_loss_to_report", 0) > 30:
        tags.append("reporte tardío fraude premeditado")
    if features.get("risk_encoded", 0) == 2:
        tags.append("segmento riesgo alto historial")
    if features.get("policy_age_days", 999) < 90:
        tags.append("póliza nueva fraude entrada")
    return " ".join(tags) if tags else "fraude seguros detección features análisis"


def _rule_based_explanation(features: dict, risk_level: str, probability: float) -> str:
    """Fallback when no LLM is available."""
    reasons = []

    ratio = features.get("claim_to_premium_ratio", 0)
    if ratio > 100:
        reasons.append(f"ratio claim/prima extremo ({ratio:.0f}x — umbral normal: <10x)")
    elif ratio > 10:
        reasons.append(f"ratio claim/prima elevado ({ratio:.1f}x — umbral normal: <10x)")

    if features.get("is_night_incident", 0):
        hour = features.get("INCIDENT_HOUR_OF_THE_DAY", "?")
        reasons.append(f"incidente en horario nocturno ({hour}h — menor probabilidad de testigos)")

    if features.get("no_police_report", 0) and features.get("is_major_loss", 0):
        reasons.append("pérdida mayor sin reporte policial (combinación de alto riesgo)")
    elif features.get("no_police_report", 0):
        reasons.append("ausencia de reporte policial")

    if features.get("no_injury_high_claim", 0):
        reasons.append("claim de alto monto sin lesiones reportadas (inconsistencia)")

    days = features.get("days_loss_to_report", 0)
    if days > 30:
        reasons.append(f"reporte tardío ({days} días después del incidente)")

    if features.get("policy_age_days", 999) < 90:
        reasons.append(f"póliza reciente ({features['policy_age_days']} días) — patrón de fraude de entrada")

    if not reasons:
        reasons.append("combinación de múltiples features con valores en rango de riesgo")

    reason_text = "; ".join(reasons)
    return (
        f"Claim marcada como {risk_level} (probabilidad: {probability:.1%}). "
        f"Factores principales: {reason_text}. "
        f"(Explicación automática — configure ANTHROPIC_API_KEY para análisis detallado con IA.)"
    )


class ClaimExplainer:
    def __init__(self, vectorstore: FAISS):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        self._chain = self._build_chain()

    def _build_chain(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0.2,
            max_tokens=512,
        )

        def retrieve_context(inputs: dict) -> str:
            docs = self.retriever.invoke(inputs["query"])
            return "\n\n".join(doc.page_content for doc in docs)

        return (
            RunnablePassthrough()
            | {
                "context": RunnableLambda(retrieve_context),
                "claim_summary": RunnableLambda(lambda x: x["claim_summary"]),
                "risk_level": RunnableLambda(lambda x: x["risk_level"]),
                "probability_pct": RunnableLambda(lambda x: f"{x['probability']:.1%}"),
            }
            | EXPLANATION_PROMPT
            | llm
            | StrOutputParser()
        )

    def explain(self, features: dict, risk_level: str, probability: float) -> str:
        if self._chain is None:
            return _rule_based_explanation(features, risk_level, probability)

        return self._chain.invoke({
            "query": _retrieval_query(features),
            "claim_summary": _format_claim_summary(features),
            "risk_level": risk_level,
            "probability": probability,
        })
