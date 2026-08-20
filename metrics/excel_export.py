import pandas as pd
import polars as pl


def save_to_excel(
    per_user_metrics: pl.DataFrame,
    summary: pl.DataFrame,
    output_path: str = "metrics_output.xlsx",
) -> None:
    """
    Salva os dois DataFrames em planilhas distintas do mesmo arquivo Excel.

    Planilha `per_user`: uma linha por uid, colunas na ordem de
    `per_user_metrics` (uid como coluna explicita). Planilha `summary`:
    uma linha por estatistica (mean, median, min, max, q25, q75, p95,
    p99); a coluna `statistic` de `summary` vira o rotulo de cada linha
    (indice), deixando as colunas de dado limitadas a uma por metrica,
    conforme especificado. Usa openpyxl; a conversao polars -> pandas
    ocorre apenas nesta etapa de escrita.
    """
    per_user_pd = per_user_metrics.to_pandas()
    summary_pd = summary.to_pandas().set_index("statistic")
    summary_pd.index.name = None

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        per_user_pd.to_excel(writer, sheet_name="per_user", index=False)
        summary_pd.to_excel(writer, sheet_name="summary", index=True)
