from locust import HttpUser, between, task


class InferXUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task(3)
    def infer_stable(self):
        self.client.post(
            "/v1/inference",
            json={
                "model": "mobilenet_v3_small",
                "latency_budget_ms": 100,
                "accuracy_priority": "balanced",
            },
        )

    @task(1)
    def infer_canary(self):
        self.client.post(
            "/v1/inference",
            json={
                "model": "resnet18",
                "latency_budget_ms": 200,
                "accuracy_priority": "high",
            },
        )
