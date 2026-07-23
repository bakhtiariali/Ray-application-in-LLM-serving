# autoscaling/predictor.py
from typing import Optional


class AsymmetricEMA:
    """
    Exponential moving average with different alphas for increases and decreases.
    - alpha_up: how fast we react when load increases (scale-up).
    - alpha_down: how slow we react when load decreases (scale-down).

    When fed 0.0 on idle ticks, the EMA decays naturally via alpha_down,
    enabling correct scale-down behavior in time-driven control loops.
    """
    def __init__(self, alpha_up: float = 0.7, alpha_down: float = 0.3):
        assert 0.0 < alpha_up <= 1.0
        assert 0.0 < alpha_down <= 1.0
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
        self._value: Optional[float] = None

    def update(self, current: float) -> float:
        """
        Update EMA with current value and return the new EMA.
        """
        if self._value is None:
            self._value = current
            return self._value

        if current > self._value:
            alpha = self.alpha_up
        else:
            alpha = self.alpha_down

        self._value = alpha * current + (1.0 - alpha) * self._value
        return self._value

    def predict(self) -> float:
        return self._value if self._value is not None else 0.0
