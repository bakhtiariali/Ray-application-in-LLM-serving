import math

class ScalingPolicy:

    def __init__(self,
                 capacity_per_replica=200,   # tokens/sec per replica (empirical)
                 min_replicas=1,
                 max_replicas=4,
                 safety_margin=1.2):

        self.capacity = capacity_per_replica
        self.min = min_replicas
        self.max = max_replicas
        self.margin = safety_margin

    def compute_replicas(self, predicted_tokens_per_sec):

        required = (
            predicted_tokens_per_sec * self.margin
        ) / self.capacity

        replicas = math.ceil(required)

        return max(self.min, min(self.max, replicas))
