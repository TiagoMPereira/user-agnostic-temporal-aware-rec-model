import numpy as np
import polars as pl


def pivot_daily_counts(
    df: pl.DataFrame,
    item_col: str = "app_package",
    date_col: str = "formated_date",
) -> pl.DataFrame:
    """Matriz densa data x item com a contagem BRUTA (nao cumulativa) de
    interacoes de cada item em cada data (0 quando o item nao teve
    interacao naquela data). Independe de qualquer fator de decaimento --
    usada como base por `decay_from_daily_counts`, para nao repetir o
    group_by/pivot a cada valor de `lambda` testado.
    """
    daily = (
        df.group_by([item_col, date_col])
        .agg(pl.len().alias("daily_count"))
        .sort([item_col, date_col])
    )

    wide = daily.pivot(
        on=item_col,
        index=date_col,
        values="daily_count",
        aggregate_function="first",
    ).sort(date_col)

    item_cols = sorted(c for c in wide.columns if c != date_col)
    wide = wide.select([date_col, *item_cols])
    return wide.with_columns(pl.exclude(date_col).fill_null(0))


def decay_from_daily_counts(
    daily_counts: pl.DataFrame,
    lambda_: float,
    date_col: str = "formated_date",
) -> pl.DataFrame:
    """Aplica decaimento exponencial sobre a matriz de contagens brutas
    (`pivot_daily_counts`), retornando uma matriz densa data x item no
    mesmo formato de `utils.popularity_matrix.build_popularity_matrix`:
    o valor na linha `d` e a soma ponderada das interacoes de cada item
    ANTES de `d` (exclusive), com peso exp(-lambda * (d - ti)) para uma
    interacao ocorrida em `ti`.

    Calculada via recorrencia (S[0] = c[0]; S[k] = c[k] + S[k-1] *
    exp(-lambda * gap)) em vez de uma formula fechada com exp(lambda *
    data): o fator de decaimento por passo e sempre <= 1, entao nao ha
    risco de overflow mesmo com `lambda` proximo de 1 e um historico de
    milhares de dias.
    """
    item_cols = [c for c in daily_counts.columns if c != date_col]

    dates = daily_counts[date_col]
    if dates.dtype != pl.Date:
        dates = dates.str.to_date()
    day_ordinals = dates.to_numpy().astype("datetime64[D]").astype(np.int64)

    daily_values = daily_counts.drop(date_col).to_numpy().astype(np.float64)
    n_dates = daily_values.shape[0]

    # decayed[k] = soma ponderada ATE d_k, inclusive (usada apenas como
    # acumulador interno); exclusive[k] = mesma soma vista de d_k mas
    # excluindo d_k -- e o decaimento de decayed[k-1] pelo gap ate d_k,
    # exatamente o termo que a recorrencia ja calcula para obter decayed[k].
    decayed = np.empty_like(daily_values)
    exclusive = np.zeros_like(daily_values)
    decayed[0] = daily_values[0]
    for k in range(1, n_dates):
        gap = day_ordinals[k] - day_ordinals[k - 1]
        exclusive[k] = decayed[k - 1] * np.exp(-lambda_ * gap)
        decayed[k] = daily_values[k] + exclusive[k]

    result = pl.DataFrame(exclusive, schema={c: pl.Float64 for c in item_cols}, orient="row")
    return result.insert_column(0, daily_counts[date_col])


def build_decay_popularity_matrix(
    df: pl.DataFrame,
    lambda_: float,
    item_col: str = "app_package",
    date_col: str = "formated_date",
) -> pl.DataFrame:
    """Constroi a matriz de popularidade ponderada por decaimento direto
    das interacoes brutas (`pivot_daily_counts` + `decay_from_daily_counts`
    em sequencia). Usa todas as interacoes do dataset, independente do
    rating -- mesmo criterio de
    `utils.popularity_matrix.build_popularity_matrix`.
    """
    daily_counts = pivot_daily_counts(df, item_col, date_col)
    return decay_from_daily_counts(daily_counts, lambda_, date_col)


def hyperbolic_decay_from_daily_counts(
    daily_counts: pl.DataFrame,
    lambda_: float,
    date_col: str = "formated_date",
) -> pl.DataFrame:
    """Aplica decaimento hiperbolico sobre a matriz de contagens brutas
    (`pivot_daily_counts`), retornando uma matriz densa data x item no
    mesmo formato de `decay_from_daily_counts`: o valor na linha `d` e a
    soma ponderada das interacoes de cada item ANTES de `d` (exclusive),
    com peso 1 / (1 + lambda * (d - ti)) para uma interacao ocorrida em
    `ti`.

    Ao contrario do decaimento exponencial, o peso hiperbolico nao e
    telescopico -- 1/(1 + lambda*(a+b)) != 1/(1+lambda*a) * 1/(1+lambda*b)
    --, entao a recorrencia O(n_datas) de `decay_from_daily_counts` nao se
    aplica aqui. Calculada via produto matricial denso: uma matriz de
    pesos (n_datas x n_datas), zero fora da parte estritamente triangular
    inferior (peso 0 quando ti >= d), multiplicada pela matriz de
    contagens brutas (n_datas x n_itens). Custo O(n_datas^2 * n_itens) em
    vez de O(n_datas * n_itens).
    """
    item_cols = [c for c in daily_counts.columns if c != date_col]

    dates = daily_counts[date_col]
    if dates.dtype != pl.Date:
        dates = dates.str.to_date()
    day_ordinals = dates.to_numpy().astype("datetime64[D]").astype(np.int64)

    daily_values = daily_counts.drop(date_col).to_numpy().astype(np.float64)

    # gap[d, ti] = dia[d] - dia[ti]; so ti estritamente anterior a d conta
    # (mask). `safe_gap` usa 1.0 como placeholder fora da mascara so para
    # evitar 1 + lambda*gap <= 0 (division by zero/negativo) antes do
    # `np.where` zerar essas posicoes -- o valor nunca e usado de fato.
    gap = (day_ordinals[:, None] - day_ordinals[None, :]).astype(np.float64)
    mask = gap > 0
    safe_gap = np.where(mask, gap, 1.0)
    weights = np.where(mask, 1.0 / (1.0 + lambda_ * safe_gap), 0.0)

    exclusive = weights @ daily_values

    result = pl.DataFrame(exclusive, schema={c: pl.Float64 for c in item_cols}, orient="row")
    return result.insert_column(0, daily_counts[date_col])


def build_hyperbolic_decay_popularity_matrix(
    df: pl.DataFrame,
    lambda_: float,
    item_col: str = "app_package",
    date_col: str = "formated_date",
) -> pl.DataFrame:
    """Constroi a matriz de popularidade ponderada por decaimento
    hiperbolico direto das interacoes brutas (`pivot_daily_counts` +
    `hyperbolic_decay_from_daily_counts` em sequencia). Usa todas as
    interacoes do dataset, independente do rating -- mesmo criterio de
    `build_decay_popularity_matrix` / `utils.popularity_matrix.build_popularity_matrix`.
    """
    daily_counts = pivot_daily_counts(df, item_col, date_col)
    return hyperbolic_decay_from_daily_counts(daily_counts, lambda_, date_col)
