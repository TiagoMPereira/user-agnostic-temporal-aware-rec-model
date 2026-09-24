"""Estrategias de decaimento para a matriz de popularidade.

Cada estrategia e o "conector" do pipeline: recebe a grade de contagens
diarias brutas (`pivot_daily_counts`, ja existente em
utils/decay_popularity_matrix.py -- a mesma base usada hoje por decayPop e
dechypPop) e seus proprios hiperparametros, e devolve a matriz densa
data x item (acumulado EXCLUSIVE por linha: a linha `d` soma o que
aconteceu ANTES de `d`, nunca em `d` ou depois -- mesmo contrato de
utils.popularity_matrix.build_popularity_matrix e
utils.decay_popularity_matrix.decay_from_daily_counts).

IMPORTANTE (memoria): `score_row` extrai NO MAXIMO DUAS linhas de
`matrix_values` por chamada (idx_until, idx_before) -- nunca a matriz
inteira `(n_interacoes, n_itens)`. Com o catalogo real (~10 mil itens) e
~700 mil interacoes de validacao/teste, uma matriz `(n_interacoes,
n_itens)` materializada de uma vez e ~53 GB -- inviavel na maioria das
maquinas (e foi exatamente o que uma versao anterior deste modulo fazia,
via uma funcao `windowed_scores` "vetorizada" que indexava
`matrix_values[idx_array]` para todas as interacoes de uma vez). Por isso
`score_row` e chamada uma vez por interacao, dentro do laco de
`pipeline.PopularityPipeline._rank_and_format` -- mesmo perfil de memoria
dos scripts antigos (predict_pop.py e companhia), que sempre extraiam uma
linha por vez dentro do laco por usuario.

O que PODE ser vetorizado com seguranca (arrays de ESCALARES por usuario,
nao de linhas -- nunca mais que alguns MB mesmo em escala real) e feito
uma vez por trial em `decay_factors`, fora do laco por usuario.

O corte de recencia (janela W) e responsabilidade de cada estrategia
porque o algoritmo eficiente para cortar a janela difere por familia:

  - `none` / `exponential`: TELESCOPICO -- a soma anterior a janela pode
    ser subtraida algebricamente da soma completa, reescalada por um fator
    que so depende do gap (dias) entre as duas datas de corte (fator 1
    para `none`, exp(-lambda*gap) para `exponential` -- derivacao completa
    em modelos_popularidade.md, secao 3, e prova numerica historica em
    optimize_recent_decay_pop.py). As duas reaproveitam
    `_telescoping_score_row`/`_telescoping_factors` abaixo.

  - `hyperbolic`: NAO telescopico -- 1/(1+lambda*gap) nao se decompoe
    dessa forma (ver docstring de
    utils.decay_popularity_matrix.hyperbolic_decay_from_daily_counts).
    `score_row` com janela fica como esqueleto (NotImplementedError) ate
    ganhar uma implementacao propria (soma mascarada na matriz de pesos
    densa -- mesma ideia da mascara `ti < t` que
    hyperbolic_decay_from_daily_counts ja usa, so adicionando
    `ti >= t-window`). Nenhum outro componente do pipeline precisa mudar
    quando isso for implementado: e so o corpo deste metodo. `build_matrix`
    (usada quando window=None) ja funciona hoje, entao dechypPop sem
    janela -- o unico modo que optimize_dechyp_pop.py/predict_dechyp_pop.py
    (scripts antigos) sempre suportaram -- continua disponivel.
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
    def score_row(
        self,
        matrix_values: np.ndarray,
        idx_until: int,
        idx_before: int | None,
        factor: float | None,
        **params,
    ) -> np.ndarray:
        """Score (n_itens,) de UMA interacao -- extrai no maximo 2 linhas
        de `matrix_values` (idx_until, idx_before), nunca a matriz inteira
        (ver docstring do modulo). `idx_before=None` equivale a
        window=None. `factor` e o fator de reescala ja resolvido por
        `decay_factors` (None quando idx_before e None ou a familia nao e
        telescopica)."""

    def decay_factors(
        self,
        matrix_dates: np.ndarray,
        idx_until: np.ndarray,
        idx_before: np.ndarray,
        **params,
    ) -> np.ndarray:
        """Vetorizado, UMA VEZ por trial (nao por usuario): fator de
        reescala por interacao, como array de ESCALARES -- barato mesmo em
        escala real (so indices/datas, nunca linhas da matriz). So
        chamada quando `window != None` e `supports_window=True`; a
        implementacao default assume familia nao-telescopica."""
        raise NotImplementedError(f"'{self.name}' nao implementa decay_factors (ver supports_window)")


def _row_at(matrix_values: np.ndarray, idx: int) -> np.ndarray:
    """Uma linha de `matrix_values` (view, sem copia) -- linha zero se
    idx < 0 (data de referencia anterior a qualquer dado na matriz)."""
    if idx < 0:
        return np.zeros(matrix_values.shape[1], dtype=matrix_values.dtype)
    return matrix_values[idx]


def _telescoping_factors(matrix_dates, idx_until, idx_before, decay_factor_fn) -> np.ndarray:
    """Fator de reescala por interacao (array de ESCALARES, nao de
    linhas): score = M(idx_until) - factor * M(idx_before), valido para
    qualquer familia telescopica (ver docstring do modulo). `decay_factor_fn`
    recebe o gap real (dias, array) entre as linhas `idx_until` e
    `idx_before` e devolve o fator multiplicativo correspondente."""
    factor = np.zeros(len(idx_before), dtype=np.float64)
    valid = idx_before >= 0
    if np.any(valid):
        gap_days = (
            (matrix_dates[idx_until[valid]] - matrix_dates[idx_before[valid]])
            .astype("timedelta64[D]")
            .astype(np.int64)
        )
        factor[valid] = decay_factor_fn(gap_days)
    return factor


def _telescoping_score_row(matrix_values: np.ndarray, idx_until: int, idx_before: int | None, factor: float | None) -> np.ndarray:
    row_until = _row_at(matrix_values, idx_until)
    if idx_before is None or idx_before < 0:
        return row_until
    row_before = _row_at(matrix_values, idx_before)
    return row_until - row_before * factor


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

    def decay_factors(self, matrix_dates, idx_until, idx_before, **params) -> np.ndarray:
        return _telescoping_factors(
            matrix_dates, idx_until, idx_before, lambda gap: np.ones_like(gap, dtype=np.float64)
        )

    def score_row(self, matrix_values, idx_until, idx_before, factor, **params) -> np.ndarray:
        return _telescoping_score_row(matrix_values, idx_until, idx_before, factor)


class ExponentialDecayStrategy(DecayStrategy):
    """decayPop / recentDecayPop: peso exp(-lambda*(t-ti)) por interacao."""

    name = "exponential"
    param_space: dict[str, HyperparamSpace] = {"lambda_": ("float", 0.001, 1.0)}

    def build_matrix(self, daily_counts: pl.DataFrame, date_col: str, *, lambda_: float, **params) -> pl.DataFrame:
        return decay_from_daily_counts(daily_counts, lambda_, date_col)

    def decay_factors(self, matrix_dates, idx_until, idx_before, *, lambda_: float, **params) -> np.ndarray:
        return _telescoping_factors(matrix_dates, idx_until, idx_before, lambda gap: np.exp(-lambda_ * gap))

    def score_row(self, matrix_values, idx_until, idx_before, factor, *, lambda_: float, **params) -> np.ndarray:
        return _telescoping_score_row(matrix_values, idx_until, idx_before, factor)


class HyperbolicDecayStrategy(DecayStrategy):
    """dechypPop: peso 1/(1+lambda*(t-ti)) por interacao.

    ESQUELETO: ver docstring do modulo. `build_matrix` (window=None) ja
    funciona; `score_row` com corte de janela ainda nao esta implementado
    (`supports_window=False`) porque o decaimento hiperbolico nao e
    telescopico -- nao da pra reaproveitar `_telescoping_score_row`.
    """

    name = "hyperbolic"
    param_space: dict[str, HyperparamSpace] = {"lambda_": ("float", 0.001, 1.0)}
    supports_window = False

    def build_matrix(self, daily_counts: pl.DataFrame, date_col: str, *, lambda_: float, **params) -> pl.DataFrame:
        return hyperbolic_decay_from_daily_counts(daily_counts, lambda_, date_col)

    def score_row(self, matrix_values, idx_until, idx_before, factor, *, lambda_: float, **params) -> np.ndarray:
        if idx_before is not None:
            raise NotImplementedError(
                "HyperbolicDecayStrategy ainda nao suporta corte de janela "
                "(recencia): o decaimento hiperbolico nao e telescopico, "
                "entao a subtracao algebrica usada por none/exponential nao "
                "se aplica aqui (ver docstring do modulo para o caminho de "
                "implementacao futura). Use PopularityConfig(window=None)."
            )
        return _row_at(matrix_values, idx_until)


DECAY_REGISTRY: dict[str, DecayStrategy] = {
    "none": NoDecayStrategy(),
    "exponential": ExponentialDecayStrategy(),
    "hyperbolic": HyperbolicDecayStrategy(),
}
