"""Orquestrador do pipeline unificado de popularidade.

Compoe as pecas independentes -- decaimento (`pipeline.decay_strategies`),
corte de recencia (metodo `windowed_scores` de cada estrategia) e
ranking/desempate (`models.pop_utils`, Card 8, inalterado) -- atras de um
unico contrato (`PopularityConfig` + `PopularityPipeline`), usado tanto
pela otimizacao (Optuna, split="val") quanto pela predicao final
(split="test").

Usar a MESMA funcao (`PopularityPipeline.score`) nos dois casos elimina o
risco que existia nos scripts antigos: cada par optimize_*.py/predict_*.py
reimplementava a formula de pontuacao duas vezes (uma para validacao, outra
para teste), sem garantia estrutural de que as duas ficariam iguais.

Reaproveita, sem duplicar:
  - `utils.decay_popularity_matrix.pivot_daily_counts` como grade unica de
    contagens brutas -- base de qualquer estrategia de decaimento;
  - `models.pop_utils._rank_top_n` para ranking e desempate (Card 8);
  - `pipeline.decay_strategies.DECAY_REGISTRY` para matriz e corte de janela.
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from models.pop_utils import _rank_top_n
from utils.decay_popularity_matrix import pivot_daily_counts

from .decay_strategies import DECAY_REGISTRY, DecayStrategy

DATE_COL_DEFAULT = "formated_date"

# Splits (Card 5, leave-one-out) que precedem cronologicamente cada split
# avaliado -- "ja consumido pelo usuario" para uma linha de `val` e todo o
# treino; para uma linha de `test` e treino + val (a linha de val e sempre
# a penultima interacao do usuario, cronologicamente anterior ao teste).
_PRECEDING_SPLITS: dict[str, list[str]] = {
    "val": ["train"],
    "test": ["train", "val"],
}


@dataclass(frozen=True)
class PopularityConfig:
    """Um ponto no espaco de busca: qual decaimento, com quais
    hiperparametros, e qual corte de recencia."""

    decay: str
    decay_params: dict = field(default_factory=dict)
    window: int | None = None
    n_recs: int = 50

    @property
    def strategy(self) -> DecayStrategy:
        try:
            return DECAY_REGISTRY[self.decay]
        except KeyError:
            raise ValueError(
                f"Decaimento '{self.decay}' desconhecido. Opcoes: {list(DECAY_REGISTRY)}"
            ) from None


@dataclass
class PreparedContext:
    """Tudo que NAO depende de (decay, decay_params, window): resolvido
    uma unica vez por split, reaproveitado por qualquer `PopularityConfig`
    (equivalente ao `prepare()` duplicado 4x nos scripts antigos)."""

    daily_counts: pl.DataFrame
    date_col: str
    matrix_dates: np.ndarray
    n_items: int
    catalog_native: list
    app_dtype: pl.DataType
    uids: list
    timestamps: list
    consumed_lists: list
    ref_dates: np.ndarray
    idx_until: np.ndarray
    ground_truth: pl.DataFrame
    # Cache da matriz construida por strategy.build_matrix, por (decay,
    # decay_params) -- a matriz NAO depende de `window` (so o corte de
    # janela depende), entao uma busca que varia so `window` com `decay`
    # fixo (ex.: --decay none --window auto, ou qualquer --decay com
    # lambda_ fixo) reconstruia a MESMA matriz do zero a cada trial sem
    # isso. Vive no ctx (nao no pipeline) para ser descartado junto com
    # ele -- um novo prepare() comeca com cache vazio.
    _matrix_cache: dict = field(default_factory=dict, repr=False, compare=False)


def _to_date_series(df: pl.DataFrame, date_col: str) -> pl.Series:
    if df.schema[date_col] == pl.Date:
        return df[date_col]
    return df[date_col].str.to_date()


class PopularityPipeline:
    """Orquestrador unico: `prepare()` uma vez por split, `score()` uma vez
    por config (por trial do Optuna, ou uma unica vez para a predicao
    final via `predict()`)."""

    def __init__(self, n_recs: int = 50, date_col: str = DATE_COL_DEFAULT):
        self.n_recs = n_recs
        self.date_col = date_col

    def prepare(self, df: pl.DataFrame, raw_df: pl.DataFrame, split: str) -> PreparedContext:
        if split not in _PRECEDING_SPLITS:
            raise ValueError(f"split '{split}' desconhecido. Opcoes: {list(_PRECEDING_SPLITS)}")
        date_col = self.date_col

        daily_counts = pivot_daily_counts(raw_df, date_col=date_col).sort(date_col)
        if daily_counts.schema[date_col] != pl.Date:
            daily_counts = daily_counts.with_columns(pl.col(date_col).str.to_date())
        assert daily_counts[date_col].is_sorted(), (
            "daily_counts precisa estar ordenado por data ascendente -- o "
            "searchsorted usado para idx_until/idx_before e as recorrencias "
            "de cada estrategia de decaimento assumem isso"
        )

        catalog = [c for c in daily_counts.columns if c != date_col]  # ja ordenado (pivot_daily_counts)
        n_items = len(catalog)
        matrix_dates = daily_counts[date_col].to_numpy()  # datetime64[D], ascendente

        app_dtype = df.schema["app_package"]
        catalog_native = pl.Series(catalog, dtype=pl.Utf8).cast(app_dtype).to_list()
        df = df.with_columns(
            pl.col("app_package").cast(pl.Utf8).cast(pl.Enum(catalog)).to_physical().alias("code")
        )

        preceding = _PRECEDING_SPLITS[split]
        print(f"Agregando apps consumidos por usuario (splits {preceding})...")
        consumed = (
            df.filter(pl.col("split").is_in(preceding))
            .group_by("uid")
            .agg(pl.col("code").alias("consumed_codes"))
        )
        target_df = df.filter(pl.col("split") == split).join(consumed, on="uid", how="left")

        print(f"Extraindo ground truth de {split}...")
        ground_truth = target_df.select(
            pl.col("uid"),
            pl.col("app_package"),
            pl.col(date_col).alias("timestamp"),
        )

        print("Pre-calculando indices de data (vetorizado)...")
        ref_dates = _to_date_series(target_df, date_col).to_numpy()
        idx_until = np.searchsorted(matrix_dates, ref_dates, side="right") - 1

        return PreparedContext(
            daily_counts=daily_counts,
            date_col=date_col,
            matrix_dates=matrix_dates,
            n_items=n_items,
            catalog_native=catalog_native,
            app_dtype=app_dtype,
            uids=target_df["uid"].to_list(),
            timestamps=target_df[date_col].to_list(),
            consumed_lists=target_df["consumed_codes"].to_list(),
            ref_dates=ref_dates,
            idx_until=idx_until,
            ground_truth=ground_truth,
        )

    def score(self, config: PopularityConfig, ctx: PreparedContext) -> pl.DataFrame:
        """Gera as predicoes para uma config especifica, reaproveitando
        tudo que `prepare()` ja calculou. Mesma logica de ranking/desempate
        de todos os scripts antigos (`_rank_top_n`, models/pop_utils.py)."""
        strategy = config.strategy
        matrix_values = self._build_matrix_cached(strategy, config, ctx)

        idx_before = None
        if config.window is not None:
            if not strategy.supports_window:
                raise ValueError(
                    f"Estrategia '{strategy.name}' ainda nao suporta corte de "
                    "janela (recencia). Use PopularityConfig(window=None)."
                )
            window_start = ctx.ref_dates - np.timedelta64(config.window, "D")
            idx_before = np.searchsorted(ctx.matrix_dates, window_start, side="right") - 1

        scores_matrix = strategy.windowed_scores(
            matrix_values, ctx.matrix_dates, ctx.idx_until, idx_before, **config.decay_params
        )

        return self._rank_and_format(scores_matrix, matrix_values, ctx, config.n_recs)

    def predict(self, config: PopularityConfig, df: pl.DataFrame, raw_df: pl.DataFrame) -> pl.DataFrame:
        """Atalho `prepare(split="test")` + `score()` para gerar as
        predicoes finais fora do laco do Optuna."""
        ctx = self.prepare(df, raw_df, split="test")
        return self.score(config, ctx)

    def _build_matrix_cached(
        self, strategy: DecayStrategy, config: PopularityConfig, ctx: PreparedContext
    ) -> np.ndarray:
        """`build_matrix` depende so de (decay, decay_params) -- nao de
        `window`. Sem cache, uma busca que fixa `decay` (ex.: --decay none,
        onde decay_params e sempre {}) e varia so `window` reconstruiria a
        MESMA matriz a cada trial do Optuna, o custo dominante do laco de
        otimizacao. A chave inclui `config.decay` (nao so os params) para
        nao colidir entre estrategias diferentes que por acaso tenham os
        mesmos nomes de hiperparametro."""
        cache_key = (config.decay, tuple(sorted(config.decay_params.items())))
        cached = ctx._matrix_cache.get(cache_key)
        if cached is not None:
            return cached

        matrix = strategy.build_matrix(ctx.daily_counts, ctx.date_col, **config.decay_params)
        matrix_values = matrix.drop(ctx.date_col).to_numpy().astype(np.float64)
        ctx._matrix_cache[cache_key] = matrix_values
        return matrix_values

    def _rank_and_format(
        self,
        scores_matrix: np.ndarray,
        matrix_values: np.ndarray,
        ctx: PreparedContext,
        n_recs: int,
    ) -> pl.DataFrame:
        rec_cols = [f"rec{j:03d}" for j in range(n_recs)]
        schema_cols = ["uid", "timestamp", *rec_cols]

        mask = np.zeros(ctx.n_items, dtype=bool)  # reutilizada: setada e desfeita a cada usuario
        rows: list = []

        for i in range(len(ctx.uids)):
            idx_until = ctx.idx_until[i]
            scores = scores_matrix[i]

            consumed_codes = ctx.consumed_lists[i]
            if consumed_codes:
                idx_arr = np.asarray(consumed_codes, dtype=np.int64)
                mask[idx_arr] = True

            candidate_idx = np.flatnonzero(~mask)
            top_idx = _rank_top_n(candidate_idx, scores, matrix_values, idx_until, n_recs)

            if consumed_codes:
                mask[idx_arr] = False

            preds = [ctx.catalog_native[c] for c in top_idx]
            preds += [None] * (n_recs - len(preds))
            rows.append((ctx.uids[i], ctx.timestamps[i], *preds))

        dtypes = {"uid": pl.Utf8, "timestamp": pl.Utf8, **{c: ctx.app_dtype for c in rec_cols}}
        if not rows:
            return pl.DataFrame({c: [] for c in schema_cols}, schema=dtypes)
        columns = dict(zip(schema_cols, zip(*rows)))
        return pl.DataFrame(columns, schema=dtypes)
