"""Recomendação em lote com cache por ``t`` e kernel Numba.

Implementa a seção 6.4 de ``pop_matrix/instructions.md``. Cenário: muitas
consultas ``(t_q, black_list_q)``, uma por usuário/instante. Como o
ranking (``ranking.rank_items``) depende só de ``t``, ele é calculado uma
única vez por ``t`` único e depois filtrado por consulta -- a filtragem por
black list (formato CSR) roda em ``_filter_rankings_numba``, o único lugar
do pacote onde Numba é usado (seção 3 das instruções).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from numba import njit, prange

from .data import InteractionData, blacklist_to_indices
from .ranking import rank_items


def build_blacklist_csr(
    data: InteractionData,
    black_lists: Sequence[Iterable[str] | None],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converte uma lista de black lists (uma por consulta) para o formato CSR.

    Returns
    -------
    bl_indptr : np.ndarray
        ``(Q + 1,)`` int64.
    bl_indices : np.ndarray
        int32, ordenado dentro de cada consulta, sem duplicatas.
        Apps desconhecidos são ignorados.

    Notes
    -----
    O loop Python percorre as ``Q`` consultas, não apps nem linhas do
    dataset: cada black list chega como uma lista Python de tamanho
    arbitrário, então não há forma vetorizada de resolvê-las sem tocar uma
    vez em cada consulta (a mesma conversão, por consulta, já é feita por
    ``blacklist_to_indices``).
    """
    per_query = [blacklist_to_indices(data, bl) for bl in black_lists]

    sizes = np.array([idx.size for idx in per_query], dtype=np.int64)
    bl_indptr = np.zeros(len(per_query) + 1, dtype=np.int64)
    bl_indptr[1:] = np.cumsum(sizes)

    bl_indices = np.concatenate(per_query).astype(np.int32) if per_query else np.empty(0, dtype=np.int32)
    return bl_indptr, bl_indices


@njit(parallel=True, cache=True)
def _filter_rankings_numba(
    rankings: np.ndarray,
    query_rank_row: np.ndarray,
    bl_indptr: np.ndarray,
    bl_indices: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Para cada consulta, percorre o ranking do seu ``t`` e coleta os
    primeiros ``n`` apps que não estão na black list.

    Parameters
    ----------
    rankings : np.ndarray
        ``(U, K)`` int32. Linha ``u`` é o ranking do u-ésimo ``t`` único,
        preenchido com ``-1`` após o fim.
    query_rank_row : np.ndarray
        ``(Q,)`` int64. Linha de ``rankings`` usada pela consulta ``q``.
    bl_indptr, bl_indices : np.ndarray
        Black lists em CSR (índices ordenados por consulta).
    n : int
        Número de recomendações por consulta.

    Returns
    -------
    np.ndarray
        ``(Q, n)`` int32 com índices de apps; posições sem recomendação
        ficam com ``-1``.

    Notes
    -----
    Usar ``prange`` sobre as consultas. A verificação de pertinência na
    black list usa ``np.searchsorted`` na fatia ordenada da consulta.
    Parar ao encontrar ``-1`` no ranking ou ao completar ``n`` itens.
    """
    n_queries = query_rank_row.shape[0]
    k_max = rankings.shape[1]
    out = np.full((n_queries, n), -1, dtype=np.int32)

    for q in prange(n_queries):
        u = query_rank_row[q]
        bl = bl_indices[bl_indptr[q] : bl_indptr[q + 1]]
        count = 0
        for j in range(k_max):
            app = rankings[u, j]
            if app == -1:
                break
            pos = np.searchsorted(bl, app)
            if pos >= bl.shape[0] or bl[pos] != app:
                out[q, count] = app
                count += 1
                if count == n:
                    break

    return out


def recommend_batch(
    data: InteractionData,
    prefix: np.ndarray,
    n: int,
    ts: np.ndarray,
    w: int,
    bl_indptr: np.ndarray,
    bl_indices: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Recomendação em lote para ``Q`` consultas.

    Passos
    ------
    1. ``t_unicos, query_rank_row = np.unique(ts, return_inverse=True)``.
    2. ``K = n + max_q(len(black_list_q))``: tamanho suficiente para
       garantir ``n`` itens após remover qualquer black list.
    3. Para cada ``t`` único (loop Python aceitável aqui), chamar
       ``rank_items(M, P, t, w, K, seed, exclude=None)`` e gravar em
       ``rankings[u]`` (preenchido com ``-1``).
    4. Chamar ``_filter_rankings_numba``.

    Parameters
    ----------
    ts : np.ndarray
        ``(Q,)`` índices de dia de cada consulta.
    (demais parâmetros como em ``recommend`` e ``_filter_rankings_numba``)

    Returns
    -------
    np.ndarray
        ``(Q, n)`` int32 com índices de apps, ``-1`` onde não há item.
        Converter para strings com ``data.apps[idx]`` quando necessário.

    Notes
    -----
    Deve produzir exatamente o mesmo resultado que chamar ``recommend``
    consulta a consulta com a mesma ``seed`` (garantido pela chave
    aleatória por permutação de todos os apps, seção 6.3).
    """
    if n < 1:
        raise ValueError(f"n deve ser >= 1, recebido {n}.")

    t_unicos, query_rank_row = np.unique(ts, return_inverse=True)
    query_rank_row = query_rank_row.astype(np.int64)

    bl_sizes = np.diff(bl_indptr)
    max_bl = int(bl_sizes.max()) if bl_sizes.size > 0 else 0
    k = n + max_bl

    rankings = np.full((t_unicos.shape[0], k), -1, dtype=np.int32)
    for u, t in enumerate(t_unicos):
        ranking = rank_items(data.matrix, prefix, int(t), w, k, seed=seed, exclude=None)
        rankings[u, : ranking.size] = ranking

    return _filter_rankings_numba(rankings, query_rank_row, bl_indptr, bl_indices, n)
