import polars as pl


def add_centered_rating(
    df: pl.DataFrame,
    rating_col: str = "rating",
    running_mean_col: str = "running_mean",
) -> pl.DataFrame:
    """Adiciona `centered_rating` = rating - running_mean.

    Requer que `running_mean_col` ja exista em df.
    NaN se propaga naturalmente na primeira interacao de cada usuario.
    """
    if running_mean_col not in df.columns:
        raise ValueError(
            f"Coluna '{running_mean_col}' nao encontrada. "
            "Execute add_running_mean antes de add_centered_rating."
        )

    return df.with_columns(
        (pl.col(rating_col) - pl.col(running_mean_col)).alias("centered_rating")
    )
