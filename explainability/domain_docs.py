"""
Static knowledge base documents: feature descriptions + fraud patterns.
These are indexed into FAISS alongside dynamic MLflow feature importances.
"""

from langchain_core.documents import Document

FEATURE_DOCS = [
    Document(
        page_content=(
            "claim_to_premium_ratio: Ratio entre el monto reclamado y la prima pagada. "
            "Es el indicador más potente de fraude en el modelo. Un asegurado que paga $100 "
            "de prima y reclama $50,000 tiene un ratio de 500x, lo cual es altamente sospechoso. "
            "Rango normal: 0.5x–5x. Sospechoso: >10x. Fraude probable: >100x."
        ),
        metadata={"feature": "claim_to_premium_ratio", "type": "feature_description"},
    ),
    Document(
        page_content=(
            "is_night_incident: Flag que indica si el incidente ocurrió entre las 00:00 y las 05:59. "
            "Los incidentes nocturnos tienen menor probabilidad de testigos y mayor dificultad de "
            "verificación. Históricamente correlacionan con fraude en seguros de vehículos y hogar. "
            "Un claim nocturno sin reporte policial es especialmente sospechoso."
        ),
        metadata={"feature": "is_night_incident", "type": "feature_description"},
    ),
    Document(
        page_content=(
            "no_police_report: El asegurado no presentó reporte policial pese al incidente. "
            "La ausencia de reporte policial es una de las señales más comunes en fraude de seguros. "
            "Un incident grave (Major Loss) sin reporte policial genera una puntuación de riesgo alta. "
            "Combinado con claim alto y sin lesiones, es un patrón clásico de fraude."
        ),
        metadata={"feature": "no_police_report", "type": "feature_description"},
    ),
    Document(
        page_content=(
            "no_injury_high_claim: Combinación de sin lesiones (ANY_INJURY=0) y claim de alto monto. "
            "Reclamar daños materiales muy altos sin reportar lesiones físicas es un patrón de fraude "
            "documentado. En accidentes reales con daños severos, las lesiones son frecuentes. "
            "Este feature captura la inconsistencia entre severidad del daño y ausencia de lesiones."
        ),
        metadata={"feature": "no_injury_high_claim", "type": "feature_description"},
    ),
    Document(
        page_content=(
            "days_loss_to_report: Días entre la fecha del incidente (LOSS_DT) y la fecha de reporte (REPORT_DT). "
            "Reportes tardíos (>30 días) son sospechosos porque sugieren que el asegurado construyó "
            "una historia antes de reportar. Reportes inmediatos (<1 día) en incidentes nocturnos "
            "también son señal de alerta. Rango normal: 1–14 días."
        ),
        metadata={"feature": "days_loss_to_report", "type": "feature_description"},
    ),
    Document(
        page_content=(
            "policy_age_days: Antigüedad de la póliza al momento del claim (LOSS_DT - POLICY_EFF_DT). "
            "Claims en pólizas muy nuevas (<90 días) son señal de fraude premeditado: el asegurado "
            "contrata la póliza con intención de reclamar. Claims en pólizas muy antiguas (>5 años) "
            "sin historial previo son generalmente legítimos."
        ),
        metadata={"feature": "policy_age_days", "type": "feature_description"},
    ),
    Document(
        page_content=(
            "is_major_loss: Indica si el incidente fue clasificado como 'Major Loss' (pérdida mayor). "
            "Los claims de pérdida total o muy alta tienen mayor escrutinio. En fraude organizado, "
            "los perpetradores tienden a reclamar el máximo posible. Este feature combinado con "
            "ausencia de reporte policial y reporte tardío es un patrón de alto riesgo."
        ),
        metadata={"feature": "is_major_loss", "type": "feature_description"},
    ),
    Document(
        page_content=(
            "risk_encoded: Segmento de riesgo del asegurado (L=0, M=1, H=2). "
            "Los asegurados en segmento H (alto riesgo) tienen historial que justifica mayor vigilancia. "
            "Un asegurado de riesgo alto con claim de monto elevado activa múltiples reglas de detección. "
            "Este feature proviene de la segmentación actuarial de la compañía."
        ),
        metadata={"feature": "risk_encoded", "type": "feature_description"},
    ),
    Document(
        page_content=(
            "is_high_claim: Flag para claims en el cuartil superior de montos (>percentil 75). "
            "Reclamos de alto monto reciben revisión automática. No es indicador de fraude por sí solo, "
            "pero amplifica el riesgo de otros features. Un claim alto + sin reporte policial + "
            "sin lesiones es la combinación de mayor peso en el modelo."
        ),
        metadata={"feature": "is_high_claim", "type": "feature_description"},
    ),
    Document(
        page_content=(
            "TENURE y AGE: Antigüedad del cliente y edad del asegurado. "
            "Clientes nuevos (TENURE < 12 meses) con claims altos son más sospechosos. "
            "La combinación de cliente nuevo + póliza nueva + claim alto es el patrón de 'fraude de entrada'. "
            "Asegurados jóvenes (<25) con pérdidas mayores también tienen mayor tasa de fraude histórica."
        ),
        metadata={"feature": "tenure_age", "type": "feature_description"},
    ),
    Document(
        page_content=(
            "Patrones de fraude en seguros: Los patrones más comunes detectados por el modelo son: "
            "(1) Fraude de entrada: póliza nueva + claim inmediato + monto alto. "
            "(2) Fraude nocturno: incidente de madrugada + sin testigos + sin reporte policial. "
            "(3) Fraude de inflación: daños reales pero monto inflado artificialmente (ratio alto). "
            "(4) Fraude organizado: agente y asegurado coordinados, múltiples claims similares. "
            "El modelo XGBoost detecta combinaciones de estos patrones, no features aislados."
        ),
        metadata={"feature": "fraud_patterns", "type": "domain_knowledge"},
    ),
    Document(
        page_content=(
            "Cómo interpreta el modelo XGBoost las features: El modelo fue entrenado con XGBoost "
            "usando scale_pos_weight para manejar el desbalance 95/5 (legítimos/fraude). "
            "Una probabilidad >0.75 indica HIGH risk, 0.50–0.75 MEDIUM risk, <0.50 LOW risk. "
            "Las feature importances del modelo muestran qué variables usa más el árbol de decisión. "
            "claim_to_premium_ratio y no_injury_high_claim son consistentemente los más importantes."
        ),
        metadata={"feature": "model_context", "type": "model_description"},
    ),
]
