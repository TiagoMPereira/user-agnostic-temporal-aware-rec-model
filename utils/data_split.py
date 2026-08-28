import polars as pl


def add_data_split(
    df: pl.DataFrame,
    uid_col: str = "uid",
    rank_col: str = "interaction_rank",
) -> pl.DataFrame:
    """Adiciona `split` (train/val/test) no estilo leave-one-out por usuario.

    Para cada usuario com N interacoes, ordenadas por `rank_col`:
      - train -> posicoes 1..N-2 (todas as interacoes ate t-2)
      - val   -> posicao N-1 (t-1)
      - test  -> posicao N (t)

    Cada usuario contribui com exatamente 1 interacao de val e 1 de
    test (as demais vao para train). Requer que `rank_col` ja exista em
    df.
    """
    if rank_col not in df.columns:
        raise ValueError(
            f"Coluna '{rank_col}' nao encontrada. "
            "Execute add_running_mean antes de add_data_split."
        )

    n_interactions = pl.col(uid_col).len().over(uid_col)
    train_end = n_interactions - 2
    val_end = n_interactions - 1

    return df.with_columns(
        pl.when(pl.col(rank_col) <= train_end)
        .then(pl.lit("train"))
        .when(pl.col(rank_col) <= val_end)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )
