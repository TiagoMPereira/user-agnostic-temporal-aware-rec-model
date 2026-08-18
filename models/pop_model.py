import random
from datetime import date, datetime, timedelta

import polars as pl

from .base_model import BaseModel


class POPModel(BaseModel):
    """Recomendacao por popularidade dentro de uma janela temporal (Card 8).

    O popularity_matrix e uma matriz densa data x item onde o
    valor na linha `d` e a contagem cumulativa de interacoes de cada item
    ANTES de `d` (exclusive). Isso garante ausencia de vazamento temporal
    por construcao: usar a linha de `reference_date` ja exclui reviews do
    proprio dia. Para uma janela [reference_date - window, reference_date],
    o score e obtido pela diferenca entre a linha de `reference_date` e a
    linha de `reference_date - window`.
    """

    def __init__(self, window: int | None = None):
        self.window = window
        self.name = f"pop_{window}d" if window is not None else "pop_all"

    def fit(self, train_data=None, val_data=None) -> None:
        pass

    def predict(
        self,
        valid_apps: list,
        n: int,
        popularity_matrix: pl.DataFrame,
        reference_date: date | datetime | str,
        date_col: str = "formated_date",
    ) -> list:
        matrix = self._with_date_column(popularity_matrix, date_col)
        reference_date = self._to_date(reference_date)

        scores = self._scores(matrix, valid_apps, reference_date, date_col)

        rng = random.Random(42)
        tiebreak = {app: rng.random() for app in valid_apps}
        ranked = sorted(
            valid_apps, key=lambda app: (-scores.get(app, 0), tiebreak[app])
        )

        return ranked[:n]

    def _scores(
        self, matrix: pl.DataFrame, valid_apps: list, reference_date: date, date_col: str
    ) -> dict:
        cum_until = self._cumulative_at(matrix, reference_date, date_col)

        if self.window is None:
            return {app: cum_until.get(app, 0) for app in valid_apps}

        window_start = reference_date - timedelta(days=self.window)
        cum_before_window = self._cumulative_at(matrix, window_start, date_col)

        return {
            app: cum_until.get(app, 0) - cum_before_window.get(app, 0)
            for app in valid_apps
        }

    @staticmethod
    def _cumulative_at(matrix: pl.DataFrame, target_date: date, date_col: str) -> dict:
        row = matrix.filter(pl.col(date_col) <= target_date).tail(1)
        if row.is_empty():
            return {}
        return row.drop(date_col).row(0, named=True)

    @staticmethod
    def _with_date_column(matrix: pl.DataFrame, date_col: str) -> pl.DataFrame:
        if matrix.schema[date_col] == pl.Date:
            return matrix
        return matrix.with_columns(pl.col(date_col).str.to_date())

    @staticmethod
    def _to_date(value: date | datetime | str) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            return date.fromisoformat(value)
        return value
