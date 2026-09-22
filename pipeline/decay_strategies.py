"""Estrategias de decaimento para a matriz de popularidade.

Cada estrategia e o "conector" do pipeline: recebe a grade de contagens
diarias brutas (`pivot_daily_counts`, ja existente em
utils/decay_popularity_matrix.py -- a mesma base usada hoje por decayPop e
dechypPop) e seus proprios hiperparametros, e devolve a matriz densa
data x item (acumulado EXCLUSIVE por linha: a linha `d` soma o que
aconteceu ANTES de `d`, nunca em `d` ou depois -- mesmo contrato de
utils.popularity_matrix.build_popularity_matrix e
utils.decay_popularity_matrix.decay_from_daily_counts).

O corte de recencia (janela W) tambem e responsabilidade de cada
estrategia, via `windowed_scores`, porque o algoritmo eficiente para
cortar a janela difere por familia:

  - `none` / `exponential`: TELESCOPICO -- a soma anterior a janela pode
    ser subtraida algebricamente da soma completa, reescalada por um fator
    que so depende do gap (dias) entre as duas datas de corte (fator 1
    para `none`, exp(-lambda*gap) para `exponential` -- derivacao completa
    em modelos_popularidade.md, secao 3, e prova numerica historica em
    optimize_recent_decay_pop.py). As duas reaproveitam
    `_telescoping_windowed_scores` abaixo.

  - `hyperbolic`: NAO telescopico -- 1/(1+lambda*gap) nao se decompoe
    dessa forma (ver docstring de
    utils.decay_popularity_matrix.hyperbolic_decay_from_daily_counts).
    `windowed_scores` fica como esqueleto (NotImplementedError) ate ganhar
    uma implementacao propria (soma mascarada na matriz de pesos densa --
    mesma ideia da mascara `ti < t` que hyperbolic_decay_from_daily_counts
    ja usa, so adicionando `ti >= t-window`). Nenhum outro componente do
    pipeline precisa mudar quando isso for implementado: e so o corpo
    deste metodo. `build_matrix` (usada quando window=None) ja funciona
    hoje, entao dechypPop sem janela -- o unico modo que
    optimize_dechyp_pop.py/predict_dechyp_pop.py (scripts antigos) sempre
    suportaram -- continua disponivel.
"""

from abc import ABC, abstractmethod

import numpy as np
import polars as pl

from utils.decay_popularity_matrix import (
    decay_from_daily_counts,
    hyperbolic_decay_from_daily_counts,
)

# (kind, low, high) -- usado pelo objective do Optuna para montar o
# search space de cada hiperparametro (ver optimize_popularity.py).
HyperparamSpace = tuple[str, float, float]


class DecayStrategy(ABC):
    """Contrato comum a toda familia de decaimento."""

    name: str
    param_space: dict[str, HyperparamSpace] = {}
    supports_window: bool = True

    @abstractmethod
    def build_matrix(self, daily_counts: pl.DataFrame, date_col: str, **params) -> pl.DataFrame:
        """Matriz densa data x item, acumulado EXCLUSIVE por linha."""

    @abstractmethod
    def windowed_scores(
        self,
        matrix_values: np.ndarray,
        matrix_dates: np.ndarray,
        idx_until: np.ndarray,
        idx_before: np.ndarray | None,
        **params,
    ) -> np.ndarray:
        """Score (n_interacoes, n_items) considerando so a janela
        [d_before, d_until). `idx_before=None` equivale a window=None (sem
        corte -- retorna a linha `idx_until` de matrix_values direto).
        `idx_until`/`idx_before` sao indices de linha ja resolvidos pelo
        pipeline via busca binaria (searchsorted) sobre `matrix_dates` --
        nao dependem da estrategia, so das datas de referencia e de W."""


def _rows_at(matrix_values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Linhas de `matrix_values` em `idx`, com linha zero onde idx < 0
    (data de referencia anterior a qualquer dado presente na matriz)."""
    n_items = matrix_values.shape[1]
    zero_row = np.zeros(n_items, dtype=matrix_values.dtype)
    valid = idx >= 0
    return np.where(valid[:, None], matrix_values[np.clip(idx, 0, None)], zero_row)


def _telescoping_windowed_scores(
    matrix_values: np.ndarray,
    matrix_dates: np.ndarray,
    idx_until: np.ndarray,
    idx_before: np.ndarray | None,
    decay_factor,
) -> np.ndarray:
    """score = M(idx_until) - decay_factor(gap) * M(idx_before).

    Valido para qualquer familia telescopica (ver docstring do modulo).
    `decay_factor` recebe o gap real (dias, array) entre as linhas
    `idx_until` e `idx_before` e devolve o fator multiplicativo que
    reescala a soma acumulada ate `idx_before` para a mesma referencia de
    `idx_until`.
    """
    row_until = _rows_at(matrix_values, idx_until)
    if idx_before is None:
        return row_until

    row_before = _rows_at(matrix_values, idx_before)

    valid_before = idx_before >= 0
    gap_days = np.zeros(len(idx_before), dtype=np.int64)
    gap_days[valid_before] = (
        (matrix_dates[idx_until[valid_before]] - matrix_dates[idx_before[valid_before]])
        .astype("timedelta64[D]")
        .astype(np.int64)
    )

    factor = np.zeros(len(idx_before), dtype=np.float64)
    factor[valid_before] = decay_factor(gap_days[valid_before])

    return row_until - row_before * factor[:, None]


class NoDecayStrategy(DecayStrategy):
    """recentPop: sem ponderacao -- cada interacao conta 1, dentro (ou
    fora) da janela. Matematicamente e o caso degenerado `lambda=0` de
    `exponential` (ver modelos_popularidade.md), mas implementada a parte,
    sem hiperparametro, para nao gastar trials do Optuna testando
    lambda=0 e para expor um nome de estrategia sem parametros (nenhum
    `decay_params` a preencher)."""

    name = "none"
    param_space: dict[str, HyperparamSpace] = {}

    def build_matrix(self, daily_counts: pl.DataFrame, date_col: str, **params) -> pl.DataFrame:
        item_cols = [c for c in daily_counts.columns if c != date_col]
        # numpy, nao uma expressao pl.col(...).cum_sum() por coluna: com o
        # catalogo real (~10 mil itens), 10 mil expressoes individuais no
        # plano de execucao do polars sao muito mais lentas do que um unico
        # np.cumsum vetorizado sobre a matriz densa inteira.
        values = daily_counts.select(item_cols).to_numpy()
        exclusive = np.zeros_like(values)
        exclusive[1:] = np.cumsum(values[:-1], axis=0)
        result = pl.DataFrame(exclusive, schema=item_cols)
        return result.insert_column(0, daily_counts[date_col])

    def windowed_scores(
        self,
        matrix_values: np.ndarray,
        matrix_dates: np.ndarray,
        idx_until: np.ndarray,
        idx_before: np.ndarray | None,
        **params,
    ) -> np.ndarray:
        return _telescoping_windowed_scores(
            matrix_values,
            matrix_dates,
            idx_until,
            idx_before,
            decay_factor=lambda gap: np.ones_like(gap, dtype=np.float64),
        )


class ExponentialDecayStrategy(DecayStrategy):
    """decayPop / recentDecayPop: peso exp(-lambda*(t-ti)) por interacao."""

    name = "exponential"
    param_space: dict[str, HyperparamSpace] = {"lambda_": ("float", 0.001, 1.0)}

    def build_matrix(self, daily_counts: pl.DataFrame, date_col: str, *, lambda_: float, **params) -> pl.DataFrame:
        return decay_from_daily_counts(daily_counts, lambda_, date_col)

    def windowed_scores(
        self,
        matrix_values: np.ndarray,
        matrix_dates: np.ndarray,
        idx_until: np.ndarray,
        idx_before: np.ndarray | None,
        *,
        lambda_: float,
        **params,
    ) -> np.ndarray:
        return _telescoping_windowed_scores(
            matrix_values,
            matrix_dates,
            idx_until,
            idx_before,
            decay_factor=lambda gap: np.exp(-lambda_ * gap),
        )


class HyperbolicDecayStrategy(DecayStrategy):
    """dechypPop: peso 1/(1+lambda*(t-ti)) por interacao.

    ESQUELETO: ver docstring do modulo. `build_matrix` (window=None) ja
    funciona; `windowed_scores` com corte de janela ainda nao esta
    implementado (`supports_window=False`) porque o decaimento hiperbolico
    nao e telescopico -- nao da pra reaproveitar
    `_telescoping_windowed_scores`.
    """

    name = "hyperbolic"
    param_space: dict[str, HyperparamSpace] = {"lambda_": ("float", 0.001, 1.0)}
    supports_window = False

    def build_matrix(self, daily_counts: pl.DataFrame, date_col: str, *, lambda_: float, **params) -> pl.DataFrame:
        return hyperbolic_decay_from_daily_counts(daily_counts, lambda_, date_col)

    def windowed_scores(
        self,
        matrix_values: np.ndarray,
        matrix_dates: np.ndarray,
        idx_until: np.ndarray,
        idx_before: np.ndarray | None,
        *,
        lambda_: float,
        **params,
    ) -> np.ndarray:
        if idx_before is not None:
            raise NotImplementedError(
                "HyperbolicDecayStrategy ainda nao suporta corte de janela "
                "(recencia): o decaimento hiperbolico nao e telescopico, "
                "entao a subtracao algebrica usada por none/exponential nao "
                "se aplica aqui (ver docstring do modulo para o caminho de "
                "implementacao futura). Use PopularityConfig(window=None)."
            )
        return _rows_at(matrix_values, idx_until)


DECAY_REGISTRY: dict[str, DecayStrategy] = {
    "none": NoDecayStrategy(),
    "exponential": ExponentialDecayStrategy(),
    "hyperbolic": HyperbolicDecayStrategy(),
}
