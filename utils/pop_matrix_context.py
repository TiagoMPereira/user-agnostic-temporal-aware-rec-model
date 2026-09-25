"""Preparo compartilhado do contexto matriz/avaliacao para os scripts que
usam o pacote ``pop_matrix`` (``predict_pop_matrix.py``,
``optimize_pop_matrix.py``).

Existe para NAO duplicar a logica de "quais apps o usuario ja consumiu" em
mais de um lugar: os scripts fazem exatamente a mesma pergunta (dado um
``uid``, quais ``app_package`` ja apareceram nos splits usados para
construir a matriz?) e usam o resultado como black list da recomendacao --
uma unica implementacao, testada uma vez (``verify_blacklist_respected``),
reduz o risco dos scripts divergirem silenciosamente.

``prepare_pop_matrix_context`` e generico (quais splits formam a matriz,
qual split e avaliado); ``prepare_train_val_context`` e
``prepare_trainval_test_context`` sao os dois casos usados no projeto:

- treino -> validacao: usado durante a otimizacao do Optuna
  (``optimize_pop_matrix.py``) -- nunca toca o split de teste.
- treino+validacao -> teste: usado para a avaliacao final, so depois de
  escolhido o melhor ``window`` via Optuna (``predict_pop_matrix.py
  --stage test``).
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl

from metrics.interaction_metrics import ndcg_at_k
from metrics.rank import compute_rank_expr
from pop_matrix import InteractionData, build_blacklist_csr, build_interaction_data, build_prefix_sums


@dataclass
class PopMatrixContext:
    """Tudo que a predicao/otimizacao sobre um split de avaliacao precisam,
    resolvido uma unica vez a partir de ``interactions_fe.parquet``.

    Attributes
    ----------
    data : InteractionData
        Matriz de interacoes construida SOMENTE com os splits da matriz
        (treino, ou treino+validacao -- ver ``prepare_pop_matrix_context``).
    prefix : np.ndarray
        Somas de prefixos de ``data.matrix``.
    eval_df : pl.DataFrame
        Uma linha por interacao do split de avaliacao: ``uid``,
        ``app_package`` (o item realmente consumido -- ground truth),
        ``timestamp`` (``formated_date`` original, string), ``date``
        (``pl.Date``) e ``consumed_apps`` (lista dos ``app_package`` que o
        usuario ja consumiu nos splits da matriz -- ``null`` se o usuario
        nao teve nenhuma interacao la).
    ts : np.ndarray
        Indice de dia (``t``) de cada linha de ``eval_df``, relativo a
        ``data.start_date`` -- mesma ordem de linhas que ``eval_df``.
    bl_indptr, bl_indices : np.ndarray
        Black list de cada interacao de avaliacao (``consumed_apps``) no
        formato CSR esperado por ``recommend_batch``, mesma ordem de
        ``eval_df``.
    """

    data: InteractionData
    prefix: np.ndarray
    eval_df: pl.DataFrame
    ts: np.ndarray
    bl_indptr: np.ndarray
    bl_indices: np.ndarray


def prepare_pop_matrix_context(
    interactions_path: str,
    matrix_splits: Sequence[str],
    eval_split: str,
) -> PopMatrixContext:
    """
    Le ``interactions_fe.parquet``, restringe aos splits realmente usados
    (``matrix_splits`` + ``eval_split`` -- qualquer outro split, ex. teste
    numa chamada treino->validacao, nunca e tocado), constroi a matriz de
    interacoes (``M``/``P``) com ``matrix_splits`` e resolve o necessario
    para prever/avaliar ``eval_split``: para cada interacao de avaliacao, a
    black list e exatamente os ``app_package`` que aquele ``uid`` ja
    consumiu em ``matrix_splits``.

    Parameters
    ----------
    interactions_path : str
        Caminho para ``interactions_fe.parquet`` (colunas ``uid``,
        ``app_package``, ``formated_date`` (string "YYYY-MM-DD") e
        ``split``, geradas por ``feature_engineering.py``).
    matrix_splits : sequence of str
        Splits usados para construir a matriz de interacoes e a black
        list (ex.: ``["train"]`` ou ``["train", "val"]``).
    eval_split : str
        Split usado para gerar as consultas de recomendacao, uma por
        interacao (ex.: ``"val"`` ou ``"test"``).

    Returns
    -------
    PopMatrixContext
    """
    lf = pl.scan_parquet(interactions_path).filter(pl.col("split").is_in([*matrix_splits, eval_split]))

    print(f"Construindo matriz de interacoes (M, P) com os splits {list(matrix_splits)}...")
    matrix_lf = lf.filter(pl.col("split").is_in(matrix_splits)).select(
        pl.col("app_package").cast(pl.Utf8),
        pl.col("formated_date").str.to_date().alias("date"),
    )
    data = build_interaction_data(matrix_lf)
    prefix = build_prefix_sums(data.matrix)

    print(f"Agregando apps ja consumidos por usuario em {list(matrix_splits)} (black list)...")
    consumed = (
        lf.filter(pl.col("split").is_in(matrix_splits))
        .group_by("uid")
        .agg(pl.col("app_package").cast(pl.Utf8).alias("consumed_apps"))
    )

    print(f"Selecionando interacoes do split '{eval_split}'...")
    eval_df = (
        lf.filter(pl.col("split") == eval_split)
        .select(
            pl.col("uid"),
            pl.col("app_package").cast(pl.Utf8),
            pl.col("formated_date").alias("timestamp"),
            pl.col("formated_date").str.to_date().alias("date"),
        )
        .join(consumed, on="uid", how="left")
        .collect()
    )
    print(f"OK: {eval_df.height} interacoes de '{eval_split}'")

    ts = (eval_df["date"] - data.start_date).dt.total_days().cast(pl.Int64).to_numpy()
    # Usuarios "frios" (0 interacoes na matriz, N=2 no split leave-one-out)
    # podem ter a unica interacao anterior a avaliacao fora do calendario
    # da matriz: `t` relativo a `data.start_date` fica negativo. A janela
    # de popularidade fica vazia de qualquer forma (nao ha interacao antes
    # de start_date), entao o dia 0 e equivalente -- clip em vez de deixar
    # `recommend_batch`/`window_bounds` levantar ValueError (t < 0).
    ts = np.maximum(ts, 0)

    black_lists = eval_df["consumed_apps"].to_list()
    bl_indptr, bl_indices = build_blacklist_csr(data, black_lists)

    return PopMatrixContext(data=data, prefix=prefix, eval_df=eval_df, ts=ts, bl_indptr=bl_indptr, bl_indices=bl_indices)


def prepare_train_val_context(interactions_path: str) -> PopMatrixContext:
    """Matriz construida SOMENTE com o split de treino; avaliacao no split
    de validacao. Configuracao usada durante a otimizacao do Optuna
    (``optimize_pop_matrix.py``) -- nunca toca o split de teste."""
    return prepare_pop_matrix_context(interactions_path, matrix_splits=["train"], eval_split="val")


def prepare_trainval_test_context(interactions_path: str) -> PopMatrixContext:
    """Matriz construida com treino+validacao; avaliacao no split de
    teste. Configuracao da avaliacao final, com o ``window`` ja escolhido
    pelo Optuna -- rodar so depois da otimizacao concluida
    (``predict_pop_matrix.py --stage test``)."""
    return prepare_pop_matrix_context(interactions_path, matrix_splits=["train", "val"], eval_split="test")


def evaluate_ndcg20(predictions: pl.DataFrame, eval_df: pl.DataFrame, n_recs: int) -> float:
    """
    NDCG@20 medio das predicoes contra o app_package realmente consumido
    (``eval_df``), juntando por (``uid``, ``timestamp``).

    Reaproveita ``metrics/rank.py`` e ``metrics/interaction_metrics.py``
    (mesma formula usada por ``evaluate_predictions.py``) em vez de
    recalcular a metrica aqui -- so em memoria, sem passar por arquivo
    (diferente de ``metrics.evaluation.compute_rank``, que le de parquet).

    ``n_recs`` deve ser o numero real de colunas ``rec000``..``rec{n_recs-1}``
    presentes em ``predictions`` (nunca o default global de
    ``metrics.rank``, 250): senao o rank de um miss cairia dentro do top-20
    e seria contado como acerto.

    Parameters
    ----------
    predictions : pl.DataFrame
        Saida de ``rankings_to_predictions`` (``uid``, ``timestamp``,
        ``rec000``..``rec{n_recs-1}``).
    eval_df : pl.DataFrame
        ``eval_df`` de ``PopMatrixContext`` (precisa de ``uid``,
        ``timestamp``, ``app_package``).
    n_recs : int
        Numero de colunas de recomendacao em ``predictions`` (``>= 20``,
        senao NDCG@20 fica mal definido).

    Returns
    -------
    float
        NDCG@20 medio sobre as interacoes de ``eval_df`` que tem par em
        ``predictions``.
    """
    if n_recs < 20:
        raise ValueError(f"n_recs deve ser >= 20 para NDCG@20 ser bem definido, recebido {n_recs}.")

    ground_truth = eval_df.lazy().select("uid", "timestamp", "app_package")
    joined = ground_truth.join(predictions.lazy(), on=["uid", "timestamp"], how="inner")
    with_rank = joined.with_columns(compute_rank_expr(n_recs=n_recs))
    with_metric = with_rank.with_columns(ndcg_at_k(20))
    return with_metric.select(pl.col("ndcg_at_20").mean()).collect().item()


def rankings_to_predictions(rankings: np.ndarray, apps: np.ndarray, eval_df: pl.DataFrame) -> pl.DataFrame:
    """
    Converte ``(Q, n_recs)`` indices de app (``-1`` = sem recomendacao) em
    um ``pl.DataFrame`` (``uid``, ``timestamp``, ``rec000``..``rec{n-1}``),
    mesmo formato usado por ``predict_popularity.py``/``pipeline.py`` --
    ``eval_df`` deve estar na MESMA ordem de linhas usada para gerar ``ts``
    (``prepare_pop_matrix_context``).
    """
    rec_cols = [f"rec{j:03d}" for j in range(rankings.shape[1])]

    # Sentinela de string vazia (nunca um app_package real -- sao sempre
    # nao vazios) em vez de `None`/dtype object: `np.append(apps, "")`
    # preserva o array como unicode nativo (`<U*`), nao promove pra
    # `object`. Um ndarray unicode -> pl.DataFrame e sempre `String`, sem
    # ambiguidade nenhuma -- diferente de um ndarray `object` com `None`
    # misturado, cujo dtype inferido (Object, Int64, ...) ja se mostrou
    # dependente da versao de numpy/pyarrow/polars instalada (foi a causa
    # de um ComputeError em producao: "cannot compare string with numeric
    # type" ao comparar `app_package` com uma coluna `recXXX` que virou
    # i64 numa combinacao de versoes diferente desta). Trocar a sentinela
    # por null e feito depois, ja dentro do polars.
    apps_or_sentinel = np.append(apps, "")
    rec_values = apps_or_sentinel[rankings]

    predictions = pl.DataFrame(rec_values, schema=rec_cols)
    predictions = predictions.with_columns(
        [pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).alias(c) for c in rec_cols]
    )
    predictions = predictions.insert_column(0, eval_df["timestamp"])
    predictions = predictions.insert_column(0, eval_df["uid"])
    return predictions


def verify_blacklist_respected(predictions: pl.DataFrame, eval_df: pl.DataFrame, rec_cols: Sequence[str]) -> None:
    """
    Confere, de forma vetorizada (sem loop Python sobre usuarios), que
    NENHUMA recomendacao em ``predictions`` e um app que o proprio usuario
    ja consumiu na matriz (``eval_df.consumed_apps``) -- a garantia central
    da black list (secao 4.1/6.3 das instrucoes de ``pop_matrix``: apps ja
    consumidos nunca podem ser recomendados).

    Parameters
    ----------
    predictions : pl.DataFrame
        Saida de ``rankings_to_predictions`` (``uid`` + colunas de ``rec_cols``).
    eval_df : pl.DataFrame
        Contexto de avaliacao (``PopMatrixContext.eval_df``), com ``uid``
        e ``consumed_apps``.
    rec_cols : sequence of str
        Nomes das colunas de recomendacao a verificar.

    Raises
    ------
    AssertionError
        Se qualquer predicao recomendar um app da black list do usuario.
    """
    joined = predictions.join(eval_df.select("uid", "consumed_apps"), on="uid", how="left")

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
            "predicoes) sao apps que o proprio usuario ja consumiu na matriz."
        )
    print("OK: nenhuma recomendacao viola a black list (apps ja consumidos na matriz).")
