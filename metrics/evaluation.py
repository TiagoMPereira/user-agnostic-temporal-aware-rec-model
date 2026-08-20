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


def aggregate_per_user(interaction_metrics: pl.LazyFrame) -> pl.DataFrame:
    """
    Agrega as metricas por usuario (Estagio 1 da macro-agregacao).

    Para cada uid, calcula a media de cada metrica sobre todas as suas
    interacoes de teste -- nunca a media direta sobre as linhas brutas
    (isso seria agregacao micro, com peso maior para usuarios com mais
    interacoes).

    Retorna um DataFrame com uma linha por uid e colunas
    [uid, hr_at_1, hr_at_5, hr_at_10, hr_at_15, hr_at_20, ndcg_at_5,
    ndcg_at_10, ndcg_at_15, ndcg_at_20, mrr].
    """
    per_user = (
        interaction_metrics.group_by("uid")
        .agg([pl.col(c).mean() for c in METRIC_COLUMNS])
        .select("uid", *METRIC_COLUMNS)
    )
    return per_user.collect()


def compute_summary(per_user_metrics: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula as estatisticas resumo (Estagio 2 da macro-agregacao).

    Calcula mean, median, min, max, q25, q75, p95 e p99 de cada metrica
    sobre os valores ja agregados por usuario em `per_user_metrics`
    (Estagio 1) -- nunca sobre as interacoes brutas.

    Retorna um DataFrame com 8 linhas, na ordem exata acima, e colunas
    [statistic, hr_at_1, ..., mrr] (mesma ordem de metricas de
    `per_user_metrics`, sem uid). A coluna `statistic` identifica cada
    linha e e usada por `save_to_excel` como rotulo de linha da planilha.
    """
    metric_cols = [c for c in per_user_metrics.columns if c != "uid"]

    rows = [
        per_user_metrics.select(
            [fn(pl.col(c)).alias(c) for c in metric_cols]
        ).with_columns(pl.lit(stat).alias("statistic"))
        for stat, fn in _STAT_FUNCS.items()
    ]

    return pl.concat(rows).select("statistic", *metric_cols)
