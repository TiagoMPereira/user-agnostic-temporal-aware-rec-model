import polars as pl

N_RECS = 250
MISS_RANK = N_RECS + 1  # 251: app_package nao aparece em nenhuma das N_RECS posicoes


def compute_rank_expr(app_col: str = "app_package", n_recs: int = N_RECS) -> pl.Expr:
    """Expressao vetorizada do rank 1-indexed de `app_col` entre as colunas
    rec000..rec{n_recs-1}, ou n_recs+1 se o app nao aparecer em nenhuma
    posicao.

    Implementada via pl.min_horizontal sobre `n_recs` expressoes
    condicionais (uma por coluna de recomendacao) -- sem UDFs Python,
    viavel em datasets de milhoes de linhas.
    """
    miss_rank = n_recs + 1
    position_exprs = [
        pl.when(pl.col(app_col) == pl.col(f"rec{i:03d}"))
        .then(pl.lit(i + 1, dtype=pl.UInt16))
        .otherwise(pl.lit(miss_rank, dtype=pl.UInt16))
        for i in range(n_recs)
    ]
    return pl.min_horizontal(position_exprs).alias("rank")
