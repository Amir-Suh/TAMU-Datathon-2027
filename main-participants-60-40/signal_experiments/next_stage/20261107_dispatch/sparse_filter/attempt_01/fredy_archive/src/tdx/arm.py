"""Minimal arm interface."""
class Arm:
    def fit(self, rnd, spec):
        raise NotImplementedError

    def predict(self, split):
        raise NotImplementedError

    def save(self, directory):
        raise NotImplementedError

    @classmethod
    def load(cls, directory):
        raise NotImplementedError
