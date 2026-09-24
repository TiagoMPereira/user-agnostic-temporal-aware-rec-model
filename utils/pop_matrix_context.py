"""Preparo compartilhado do contexto treino/validacao para os scripts que
usam o pacote ``pop_matrix`` (``predict_pop_matrix.py``,
``optimize_pop_matrix.py``).

Existe para NAO duplicar a logica de "quais apps o usuario ja consumiu" em
mais de um lugar: os dois scripts fazem exatamente a mesma pergunta (dado
um ``uid``, quais ``app_package`` ja apareceram no split de treino?) e
usam o resultado como black list da recomendacao -- uma unica
implementacao, testada uma vez (``verify_blacklist_respected``), reduz o
risco de os dois scripts divergirem silenciosamente.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl

from pop_matrix import InteractionData, build_blacklist_csr, build_interaction_data, build_prefix_sums


@dataclass
class TrainValContext:
    """Tudo que a predicao/otimizacao sobre o split de validacao precisam,
    resolvido uma unica vez a partir de ``interactions_fe.parquet``.

    Attributes
    ----------
    data : InteractionData
        Matriz de interacoes construida SOMENTE com o split de treino.
    prefix : np.ndarray
        Somas de prefixos de ``data.matrix``.
    val_df : pl.DataFrame
        Uma linha por interacao de validacao: ``uid``, ``app_package``
        (o item realmente consumido -- ground truth), ``timestamp``
        (``formated_date`` original, string), ``date`` (``pl.Date``) e
        ``consumed_apps`` (lista dos ``app_package`` que o usuario ja
        consumiu no treino -- ``null`` se o usuario nao teve nenhuma
        interacao de treino).
    ts : np.ndarray
        Indice de dia (``t``) de cada linha de ``val_df``, relativo a
        ``data.start_date`` -- mesma ordem de linhas que ``val_df``.
    bl_indptr, bl_indices : np.ndarray
        Black list de cada interacao de validacao (``consumed_apps``) no
        formato CSR esperado por ``recommend_batch``, mesma ordem de
        ``val_df``.
    """

    data: InteractionData
    prefix: np.ndarray
    val_df: pl.DataFrame
    ts: np.ndarray
    bl_indptr: np.ndarray
    bl_indices: np.ndarray


def prepare_train_val_context(interactions_path: str) -> TrainValContext:
    """
    Le ``interactions_fe.parquet``, remove o split de teste, constroi a
    matriz de interacoes (``M``/``P``) SOMENTE com o split de treino, e
    resolve o necessario para prever/avaliar o split de validacao: para
    cada interacao de val, a black list e exatamente os ``app_package``
    que aquele ``uid`` ja consumiu no treino (nunca teste -- teste e
    cronologicamente posterior ao val no split leave-one-out e nunca deve
    influenciar a recomendacao).

    Parameters
    ----------
    interactions_path : str
        Caminho para ``interactions_fe.parquet`` (colunas ``uid``,
        ``app_package``, ``formated_date`` (string "YYYY-MM-DD") e
        ``split``, geradas por ``feature_engineering.py``).

    Returns
    -------
    TrainValContext
    """
    lf = pl.scan_parquet(interactions_path).filter(pl.col("split") != "test")

    print("Construindo matriz de interacoes (M, P) apenas com o split de treino...")
    train_lf = lf.filter(pl.col("split") == "train").select(
        pl.col("app_package").cast(pl.Utf8),
        pl.col("formated_date").str.to_date().alias("date"),
    )
    data = build_interaction_data(train_lf)
    prefix = build_prefix_sums(data.matrix)

    print("Agregando apps ja consumidos por usuario no treino (black list)...")
    consumed = (
        lf.filter(pl.col("split") == "train")
        .group_by("uid")
        .agg(pl.col("app_package").alias("consumed_apps"))
    )

    print("Selecionando interacoes de validacao...")
    val_df = (
        lf.filter(pl.col("split") == "val")
        .select(
            pl.col("uid"),
            pl.col("app_package"),
            pl.col("formated_date").alias("timestamp"),
            pl.col("formated_date").str.to_date().alias("date"),
        )
        .join(consumed, on="uid", how="left")
        .collect()
    )
    print(f"OK: {val_df.height} interacoes de validacao")

    ts = (val_df["date"] - data.start_date).dt.total_days().cast(pl.Int64).to_numpy()
    # Usuarios "frios" (0 interacoes de treino, N=2 no split leave-one-out)
    # podem ter a unica interacao anterior ao val fora do calendario do
    # treino: `t` relativo a `data.start_date` fica negativo. A janela de
    # popularidade fica vazia de qualquer forma (nao ha treino antes de
    # start_date), entao o dia 0 e equivalente -- clip em vez de deixar
    # `recommend_batch`/`window_bounds` levantar ValueError (t < 0).
    ts = np.maximum(ts, 0)

    black_lists = val_df["consumed_apps"].to_list()
    bl_indptr, bl_indices = build_blacklist_csr(data, black_lists)

    return TrainValContext(data=data, prefix=prefix, val_df=val_df, ts=ts, bl_indptr=bl_indptr, bl_indices=bl_indices)


def rankings_to_predictions(rankings: np.ndarray, apps: np.ndarray, val_df: pl.DataFrame) -> pl.DataFrame:
    """
    Converte ``(Q, n_recs)`` indices de app (``-1`` = sem recomendacao) em
    um ``pl.DataFrame`` (``uid``, ``timestamp``, ``rec000``..``rec{n-1}``),
    mesmo formato usado por ``predict_popularity.py``/``pipeline.py`` --
    ``val_df`` deve estar na MESMA ordem de linhas usada para gerar ``ts``
    (``prepare_train_val_context``).
    """
    rec_cols = [f"rec{j:03d}" for j in range(rankings.shape[1])]
    apps_or_none = np.concatenate([apps, np.array([None], dtype=object)])
    rec_values = apps_or_none[rankings]

    # schema explicito (nao so os nomes): se alguma coluna rec* ficar 100%
    # None (nenhuma consulta do lote teve recomendacao naquela posicao),
    # o polars infere Object em vez de Utf8 a partir do ndarray de objects
    # -- e a comparacao `app_package == recXXX` (metrics/rank.py) quebra
    # com um erro de tipo incompativel. Fixar Utf8 explicitamente evita
    # depender da inferencia de tipos do polars nesse caso.
    predictions = pl.DataFrame(rec_values, schema={c: pl.Utf8 for c in rec_cols})
    predictions = predictions.insert_column(0, val_df["timestamp"])
    predictions = predictions.insert_column(0, val_df["uid"])
    return predictions


def verify_blacklist_respected(predictions: pl.DataFrame, val_df: pl.DataFrame, rec_cols: Sequence[str]) -> None:
    """
    Confere, de forma vetorizada (sem loop Python sobre usuarios), que
    NENHUMA recomendacao em ``predictions`` e um app que o proprio usuario
    ja consumiu no treino (``val_df.consumed_apps``) -- a garantia central
    da black list (secao 4.1/6.3 das instrucoes de ``pop_matrix``: apps ja
    consumidos nunca podem ser recomendados).

    Parameters
    ----------
    predictions : pl.DataFrame
        Saida de ``rankings_to_predictions`` (``uid`` + colunas de ``rec_cols``).
    val_df : pl.DataFrame
        Contexto de validacao (``prepare_train_val_context``), com
        ``uid`` e ``consumed_apps``.
    rec_cols : sequence of str
        Nomes das colunas de recomendacao a verificar.

    Raises
    ------
    AssertionError
        Se qualquer predicao recomendar um app da black list do usuario.
    """
    joined = predictions.join(val_df.select("uid", "consumed_apps"), on="uid", how="left")

    violation_cols = [f"_violates_{c}" for c in rec_cols]
    violations = (
        joined.with_columns(
            [
                pl.col("consumed_apps").list.contains(pl.col(c)).fill_null(False).alias(v)
                for c, v in zip(rec_cols, violation_cols)
            ]
        )
        .select(pl.sum_horizontal(violation_cols).alias("n_violations"))
        .filter(pl.col("n_violations") > 0)
    )

    n_rows_com_violacao = violations.height
    if n_rows_com_violacao:
        n_total = int(violations["n_violations"].sum())
        raise AssertionError(
            f"Black list violada: {n_total} recomendacoes (em {n_rows_com_violacao} "
            "predicoes) sao apps que o proprio usuario ja consumiu no treino."
        )
    print("OK: nenhuma recomendacao viola a black list (apps ja consumidos no treino).")
