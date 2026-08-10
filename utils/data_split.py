import polars as pl


def add_data_split(
    df: pl.DataFrame,
    uid_col: str = "uid",
    rank_col: str = "interaction_rank",
) -> pl.DataFrame:
    """Adiciona `split` (train/val/test), 70/15/15 por usuario.

    Para cada usuario com N interacoes, ordenadas por `rank_col`:
      - train -> posicoes 1..floor(0.70*N)
      - val   -> posicoes floor(0.70*N)+1..floor(0.85*N)
      - test  -> posicoes floor(0.85*N)+1..N

    Requer que `rank_col` ja exista em df.
    """
    if rank_col not in df.columns:
        raise ValueError(
            f"Coluna '{rank_col}' nao encontrada. "
            "Execute add_running_mean antes de add_data_split."
        )

    n_interactions = pl.col(uid_col).len().over(uid_col)
    train_end = (n_interactions * 0.70).floor()
    val_end = (n_interactions * 0.85).floor()

    return df.with_columns(
        pl.when(pl.col(rank_col) <= train_end)
        .then(pl.lit("train"))
        .when(pl.col(rank_col) <= val_end)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )
