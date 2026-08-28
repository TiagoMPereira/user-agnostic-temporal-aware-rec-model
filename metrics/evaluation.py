import polars as pl

from .interaction_metrics import METRIC_COLUMNS, all_metric_exprs
from .rank import N_RECS, compute_rank_expr

_STAT_FUNCS = {
    "mean": lambda col: col.mean(),
    "median": lambda col: col.median(),
    "min": lambda col: col.min(),
    "max": lambda col: col.max(),
    "q25": lambda col: col.quantile(0.25, interpolation="linear"),
    "q75": lambda col: col.quantile(0.75, interpolation="linear"),
    "p95": lambda col: col.quantile(0.95, interpolation="linear"),
    "p99": lambda col: col.quantile(0.99, interpolation="linear"),
}


def compute_rank(
    ground_truth_path: str,
    predictions_path: str,
) -> pl.LazyFrame:
    """
    Realiza o join, calcula o rank e todas as metricas por interacao.

    Faz um inner join (lazy) entre `ground_truth_path`
    (uid, timestamp, app_package) e `predictions_path`
    (uid, timestamp, rec000..rec249) pela chave composta (uid, timestamp).
    Registros sem correspondencia nos dois lados sao descartados
    silenciosamente. Para cada linha resultante calcula o rank do
    app_package na lista de recomendacoes (vetorizado, via
    pl.min_horizontal -- ver metrics.rank) e as metricas por interacao
    (HR@K, NDCG@K, MRR).

    Retorna um LazyFrame com colunas:
    [uid, timestamp, rank, hr_at_1, hr_at_5, hr_at_10, hr_at_15, hr_at_20,
     ndcg_at_5, ndcg_at_10, ndcg_at_15, ndcg_at_20, mrr]
    """
    ground_truth = pl.scan_parquet(ground_truth_path).select(
        "uid", "timestamp", "app_package"
    )
    predictions = pl.scan_parquet(predictions_path)

    joined = ground_truth.join(predictions, on=["uid", "timestamp"], how="inner")

    with_rank = joined.with_columns(compute_rank_expr(n_recs=N_RECS))
    with_metrics = with_rank.with_columns(all_metric_exprs())

    return with_metrics.select("uid", "timestamp", "rank", *METRIC_COLUMNS)


def compute_summary(interaction_metrics: pl.LazyFrame) -> pl.DataFrame:
    """
    Calcula as estatisticas resumo sobre as metricas por interacao.

    Com o split leave-one-out (Card 5), cada usuario contribui com
    exatamente 1 interacao de teste, entao a agregacao por usuario
    seria a identidade -- por isso as estatisticas sao calculadas
    direto sobre `interaction_metrics`, sem estagio intermediario de
    agregacao por uid.

    Calcula mean, median, min, max, q25, q75, p95 e p99 de cada
    metrica.

    Retorna um DataFrame com 8 linhas, na ordem exata acima, e colunas
    [statistic, hr_at_1, ..., mrr]. A coluna `statistic` identifica
    cada linha e e usada por `save_to_excel` como rotulo de linha da
    planilha.
    """
    metrics = interaction_metrics.select(
        [pl.col(c).cast(pl.Float64) for c in METRIC_COLUMNS]
    ).collect()

    rows = [
        metrics.select(
            [fn(pl.col(c)).alias(c) for c in METRIC_COLUMNS]
        ).with_columns(pl.lit(stat).alias("statistic"))
        for stat, fn in _STAT_FUNCS.items()
    ]

    return pl.concat(rows).select("statistic", *METRIC_COLUMNS)
