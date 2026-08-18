import random

from .base_model import BaseModel


class RandomModel(BaseModel):
    """Baseline mais fraco: ordenacao puramente aleatoria (Card 7)."""

    def __init__(self, random_state: int = 42):
        self.name = "random"
        self.random_state = random_state

    def fit(self, train_data=None, val_data=None) -> None:
        pass

    def predict(self, valid_apps: list, n: int = 1) -> list:
        rng = random.Random(self.random_state)
        return rng.sample(valid_apps, min(n, len(valid_apps)))
