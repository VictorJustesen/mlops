import random

from locust import HttpUser, between, task


class APIUser(HttpUser):
    wait_time = between(1, 2)  # Simulate user waiting 1-2 seconds between tasks

    @task(1)
    def index(self):
        self.client.get("/")

    @task(3)
    def predict(self):
        # Generate random input for [Price, Load, Production] with sequence length of 168
        dummy_data = [
            [random.uniform(0, 100), random.uniform(100, 500), random.uniform(50, 200)]
            for _ in range(168)
        ]

        self.client.post("/predict", json={"features": dummy_data})
