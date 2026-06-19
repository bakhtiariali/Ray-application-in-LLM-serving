from autoscaling.metrics import TokenMetrics
from autoscaling.predictor import MovingAveragePredictor
from autoscaling.scaling_policy import ScalingPolicy

class AutoscalingController:

    def __init__(self):
        self.metrics = TokenMetrics()
        self.predictor = MovingAveragePredictor()
        self.policy = ScalingPolicy()

    def record(self, prompt, output):
        self.metrics.record_request(prompt, output)

        current_rate = self.metrics.tokens_per_second()
        predicted_rate = self.predictor.update(current_rate)

        replicas = self.policy.compute_replicas(predicted_rate)

        return {
            "current_tokens_per_sec": current_rate,
            "predicted_tokens_per_sec": predicted_rate,
            "recommended_replicas": replicas
        }
