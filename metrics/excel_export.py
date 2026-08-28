import pandas as pd
import polars as pl


def save_to_excel(
    summary: pl.DataFrame,
    output_path: str = "metrics_output.xlsx",
) -> None:
    """
    Salva as estatisticas resumo em uma planilha do arquivo Excel.

    Planilha `summary`: uma linha por estatistica (mean, median, min,
    max, q25, q75, p95, p99); a coluna `statistic` vira o rotulo de
    cada linha (indice), deixando as colunas de dado limitadas a uma
    por metrica. Usa openpyxl; a conversao polars -> pandas ocorre
    apenas nesta etapa de escrita.
    """
    summary_pd = summary.to_pandas().set_index("statistic")
    summary_pd.index.name = None

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_pd.to_excel(writer, sheet_name="summary", index=True)
