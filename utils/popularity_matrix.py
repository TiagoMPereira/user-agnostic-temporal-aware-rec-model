import polars as pl


def build_popularity_matrix(
    df: pl.DataFrame,
    item_col: str = "app_package",
    date_col: str = "formated_date",
) -> pl.DataFrame:
    """Matriz densa data x item de popularidade cumulativa.

    Cada linha e uma data, cada coluna um item. O valor da celula
    (data d, item i) e o numero de interacoes do item i ocorridas
    ANTES de d (exclui interacoes do proprio dia d). Usa todas as
    interacoes do dataset, independente do rating.
    """
    daily = (
        df.group_by([item_col, date_col])
        .agg(pl.len().alias("daily_count"))
        .sort([item_col, date_col])
    )

    daily = daily.with_columns(
        pl.col("daily_count").cum_sum().over(item_col).alias("cum_inclusive")
    )

    wide = daily.pivot(
        on=item_col,
        index=date_col,
        values="cum_inclusive",
        aggregate_function="first",
    ).sort(date_col)

    item_cols = sorted(c for c in wide.columns if c != date_col)
    wide = wide.select([date_col, *item_cols])

    value_cols = pl.exclude(date_col)
    wide = wide.with_columns(value_cols.fill_null(strategy="forward"))
    wide = wide.with_columns(value_cols.fill_null(0))
    # desloca uma linha para excluir as interacoes do proprio dia
    wide = wide.with_columns(value_cols.shift(1))
    wide = wide.with_columns(value_cols.fill_null(0))

    return wide
