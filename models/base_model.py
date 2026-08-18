from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Contrato comum a todos os modelos de recomendacao (Card 6)."""

    name: str

    @abstractmethod
    def fit(self, train_data, val_data) -> None:
        """Treina o modelo. No-op para modelos sem fase de treino."""

    @abstractmethod
    def predict(self, valid_apps: list, n: int) -> list:
        """Retorna os `n` app_packages de `valid_apps` mais recomendados,
        ordenados do mais para o menos relevante."""
