class MovingAveragePredictor:

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.ema = None

    def update(self, value):
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
        return self.ema

    def predict(self):
        return self.ema if self.ema is not None else 0.0
