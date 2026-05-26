"""
Locust load test for the Insurance Fraud Detection API.

Validates HPA autoscaling behaviour under realistic traffic patterns.

Usage:
    # Headless — ramp to 50 users over 30s, run for 3 minutes:
    locust -f load_testing/locustfile.py \
        --headless -u 50 -r 30 --run-time 3m \
        --host http://$(minikube ip):30080

    # Web UI:
    locust -f load_testing/locustfile.py --host http://localhost:8000

Watch autoscaling in a separate terminal:
    kubectl get hpa fraud-detection-hpa --watch
    kubectl get pods -l app=fraud-detection-api --watch
"""

import random

from locust import HttpUser, between, task

# ── Payloads ──────────────────────────────────────────────────────────────────

LOW_RISK_CLAIM = {
    "premium_amount": 500.0,
    "claim_amount": 1200.0,
    "age": 42,
    "tenure": 60,
    "no_of_family_members": 3,
    "incident_hour_of_the_day": 14,
    "incident_severity": "Minor Loss",
    "any_injury": 0,
    "police_report_available": 1,
    "risk_segmentation": "L",
    "loss_dt": "2024-03-10",
    "report_dt": "2024-03-11",
    "txn_date_time": "2024-03-12 09:00:00",
    "policy_eff_dt": "2020-01-01",
}

HIGH_RISK_CLAIM = {
    "premium_amount": 150.0,
    "claim_amount": 52000.0,
    "age": 28,
    "tenure": 8,
    "no_of_family_members": 1,
    "incident_hour_of_the_day": 3,
    "incident_severity": "Major Loss",
    "any_injury": 0,
    "police_report_available": 0,
    "risk_segmentation": "H",
    "loss_dt": "2024-03-01",
    "report_dt": "2024-04-05",
    "txn_date_time": "2024-04-06 02:00:00",
    "policy_eff_dt": "2024-01-15",
}

MEDIUM_RISK_CLAIM = {
    "premium_amount": 300.0,
    "claim_amount": 8500.0,
    "age": 35,
    "tenure": 24,
    "no_of_family_members": 2,
    "incident_hour_of_the_day": 22,
    "incident_severity": "Major Loss",
    "any_injury": 1,
    "police_report_available": 0,
    "risk_segmentation": "M",
    "loss_dt": "2024-02-20",
    "report_dt": "2024-02-28",
    "txn_date_time": "2024-03-01 15:00:00",
    "policy_eff_dt": "2021-06-01",
}


def _randomize_claim(base: dict) -> dict:
    """Add small random variation so requests are not byte-identical."""
    claim = base.copy()
    claim["premium_amount"] = round(base["premium_amount"] * random.uniform(0.9, 1.1), 2)
    claim["claim_amount"] = round(base["claim_amount"] * random.uniform(0.95, 1.05), 2)
    claim["age"] = base["age"] + random.randint(-2, 2)
    return claim


# ── User behaviour ────────────────────────────────────────────────────────────

class FraudDetectionUser(HttpUser):
    """
    Simulates analyst traffic hitting the fraud API.
    Task weights reflect real usage: bulk prediction >> explain (expensive).
    """

    wait_time = between(0.5, 2.0)

    @task(10)
    def predict_low_risk(self):
        self.client.post(
            "/predict",
            json=_randomize_claim(LOW_RISK_CLAIM),
            name="/predict [low-risk]",
        )

    @task(8)
    def predict_high_risk(self):
        self.client.post(
            "/predict",
            json=_randomize_claim(HIGH_RISK_CLAIM),
            name="/predict [high-risk]",
        )

    @task(5)
    def predict_medium_risk(self):
        self.client.post(
            "/predict",
            json=_randomize_claim(MEDIUM_RISK_CLAIM),
            name="/predict [medium-risk]",
        )

    @task(2)
    def explain_high_risk(self):
        """Heavier task — hits RAG explainer."""
        self.client.post(
            "/explain",
            json=_randomize_claim(HIGH_RISK_CLAIM),
            name="/explain [high-risk]",
        )

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")


class SpikeUser(HttpUser):
    """
    Aggressive user for spike testing — drives HPA scale-up quickly.
    Spawn with: locust ... --tags spike
    """

    wait_time = between(0.1, 0.3)

    @task
    def predict_burst(self):
        self.client.post(
            "/predict",
            json=_randomize_claim(HIGH_RISK_CLAIM),
            name="/predict [spike]",
        )
