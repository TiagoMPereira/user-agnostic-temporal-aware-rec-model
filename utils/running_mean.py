import polars as pl


def add_running_mean(
    df: pl.DataFrame,
    uid_col: str = "uid",
    rating_col: str = "rating",
    date_col: str = "formated_date",
    tiebreak_col: str = "app_package",
) -> pl.DataFrame:
    """Adiciona `interaction_rank` e `running_mean`.

    running_mean(k) = media dos ratings anteriores (1..k-1) do usuario,
    ordenados cronologicamente por (date_col, tiebreak_col).
    A primeira interacao de cada usuario recebe running_mean = null.
    """
    df = df.sort([uid_col, date_col, tiebreak_col])

    df = df.with_columns(
        pl.col(rating_col).cum_count().over(uid_col).alias("interaction_rank")
    )

    prior_sum = pl.col(rating_col).cum_sum().over(uid_col) - pl.col(rating_col)
    prior_count = pl.col("interaction_rank") - 1

    df = df.with_columns(
        pl.when(prior_count == 0)
        .then(None)
        .otherwise(prior_sum / prior_count)
        .alias("running_mean")
    )

    return df
