import polars as pl

from .rank import MISS_RANK

HR_K_VALUES = (1, 5, 10, 15, 20)
NDCG_K_VALUES = (5, 10, 15, 20)

METRIC_COLUMNS = (
    [f"hr_at_{k}" for k in HR_K_VALUES]
    + [f"ndcg_at_{k}" for k in NDCG_K_VALUES]
    + ["mrr"]
)


def hr_at_k(k: int, rank_col: str = "rank") -> pl.Expr:
    """HR@K: 1 se o rank estiver nas top-K posicoes, 0 caso contrario."""
    return (pl.col(rank_col) <= k).cast(pl.Int8).alias(f"hr_at_{k}")


def ndcg_at_k(k: int, rank_col: str = "rank") -> pl.Expr:
    """NDCG@K: 1/log2(rank+1) se o rank estiver nas top-K posicoes, 0 caso
    contrario. Com ground truth unico por (uid, timestamp), IDCG = 1 e
    portanto DCG = NDCG.
    """
    gain = pl.lit(1.0) / (pl.col(rank_col) + 1).log(base=2)
    return (
        pl.when(pl.col(rank_col) <= k).then(gain).otherwise(0.0).alias(f"ndcg_at_{k}")
    )


def mrr_expr(rank_col: str = "rank", miss_rank: int = MISS_RANK) -> pl.Expr:
    """MRR: 1/rank se for hit (rank < miss_rank), 0 se for miss
    (rank == miss_rank). Misses recebem 0, nao 1/miss_rank -- a
    penalizacao por miss e total.
    """
    return (
        pl.when(pl.col(rank_col) < miss_rank)
        .then(pl.lit(1.0) / pl.col(rank_col))
        .otherwise(0.0)
        .alias("mrr")
    )


def all_metric_exprs(rank_col: str = "rank") -> list[pl.Expr]:
    """Todas as expressoes de metricas por interacao, na ordem de saida
    esperada: HR@K, NDCG@K, MRR."""
    return (
        [hr_at_k(k, rank_col) for k in HR_K_VALUES]
        + [ndcg_at_k(k, rank_col) for k in NDCG_K_VALUES]
        + [mrr_expr(rank_col)]
    )
