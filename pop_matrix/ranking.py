"""Contagem na janela, ranking por popularidade e desempate.

Implementa as seções 6.2 e 6.3 de ``pop_matrix/instructions.md``: os
limites de uma janela ``[t - w, t - 1]`` em índices de linha de ``M``, o
total de interações de cada app nessa janela (via somas de prefixos ``P``)
e a regra de ranking com desempate lexicográfico (``np.lexsort``, sem
recursão), otimizada com ``np.argpartition`` para evitar ordenar todos os
``A`` apps.
"""

from __future__ import annotations

import datetime as dt
from typing import Iterable

import numpy as np

from .data import InteractionData, blacklist_to_indices, date_to_index


def window_bounds(t: int, w: int, n_days: int) -> tuple[int, int]:
    """
    Calcula os limites da janela ``[inicio, fim)`` em índices de linha de ``M``.

    ``inicio = clip(t - w, 0, n_days)`` e ``fim = clip(t, 0, n_days)``.

    Raises
    ------
    ValueError
        Se ``t < 0`` ou ``w < 1``.
    """
    if t < 0:
        raise ValueError(f"t deve ser >= 0, recebido {t}.")
    if w < 1:
        raise ValueError(f"w deve ser >= 1, recebido {w}.")

    inicio = min(max(t - w, 0), n_days)
    fim = min(max(t, 0), n_days)
    return inicio, fim


def window_counts(prefix: np.ndarray, t: int, w: int) -> np.ndarray:
    """
    Total de interações de cada app na janela ``[t - w, t - 1]``.

    Returns
    -------
    np.ndarray
        Vetor ``(A,)`` int64: ``P[fim] - P[inicio]``.
    """
    n_days = prefix.shape[0] - 1
    inicio, fim = window_bounds(t, w, n_days)
    return prefix[fim] - prefix[inicio]


def rank_items(
    matrix: np.ndarray,
    prefix: np.ndarray,
    t: int,
    w: int,
    k: int,
    seed: int | None = None,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """
    Retorna os ``k`` melhores apps no instante ``t`` segundo a regra de
    popularidade com desempate lexicográfico.

    Parameters
    ----------
    matrix : np.ndarray
        Matriz de interações ``M`` ``(T, A)``, usada no desempate diário.
    prefix : np.ndarray
        Somas de prefixos ``P`` ``(T + 1, A)``, usadas nos totais da janela.
    t : int
        Índice do dia da recomendação. O dia ``t`` não é contado.
    w : int
        Tamanho da janela em dias (``w >= 1``).
    k : int
        Número máximo de apps a retornar (``k >= 0``).
    seed : int or None, default None
        Semente do desempate aleatório final. Mesmo ``(seed, t)`` produz
        sempre o mesmo resultado.
    exclude : np.ndarray or None, default None
        Índices de apps a excluir (ex.: black list já convertida).

    Returns
    -------
    np.ndarray
        Índices de apps (int32) em ordem de recomendação, de tamanho
        ``min(k, nº de elegíveis)``. Pode ser vazio.

    Notes
    -----
    Complexidade: O(A) para totais e ``argpartition`` mais
    O(w · c · log c) para o ``lexsort``, onde ``c`` é o número de
    candidatos (normalmente pequeno).
    """
    n_days, n_apps = matrix.shape
    inicio, fim = window_bounds(t, w, n_days)

    if k <= 0 or inicio >= fim:
        return np.empty(0, dtype=np.int32)

    totais = prefix[fim] - prefix[inicio]
    elig_mask = totais > 0
    if exclude is not None and exclude.size > 0:
        elig_mask[exclude] = False

    elig_idx = np.flatnonzero(elig_mask)
    n_elig = elig_idx.size
    if n_elig == 0:
        return np.empty(0, dtype=np.int32)

    elig_totais = totais[elig_idx]
    if n_elig <= k:
        cand = elig_idx
    else:
        # v = k-esimo maior total entre os elegiveis (argpartition, sem
        # ordenar todo mundo); candidatos = todos com total >= v, o que
        # pode incluir mais de k itens se houver empate exatamente em v.
        kth = n_elig - k
        partitioned = np.argpartition(elig_totais, kth)
        v = elig_totais[partitioned[kth]]
        cand = elig_idx[elig_totais >= v]

    seed_seq = [seed, t] if seed is not None else None
    rng = np.random.default_rng(seed_seq)
    prioridade = rng.permutation(n_apps)

    janela = matrix[inicio:fim, cand]  # linha 0 = dia mais antigo (t-w), ultima = t-1
    chaves = np.vstack([prioridade[cand], -janela, -totais[cand]])
    ordem = cand[np.lexsort(chaves)]

    return ordem[:k].astype(np.int32)


def recommend(
    data: InteractionData,
    prefix: np.ndarray,
    n: int,
    t: int | dt.date | str,
    w: int,
    black_list: Iterable[str] | None = None,
    seed: int | None = None,
) -> list[str]:
    """
    Recomenda os ``n`` apps mais populares na janela ``[t - w, t - 1]``.

    Parameters
    ----------
    data : InteractionData
        Matriz de interações original e mapeamentos.
    prefix : np.ndarray
        Somas de prefixos de ``data.matrix`` (``build_prefix_sums``).
    n : int
        Número de apps a recomendar (``n >= 1``).
    t : int, date ou str
        Instante da recomendação. Inteiro é tratado como índice de dia;
        date/str é convertido com ``date_to_index``.
    w : int
        Tamanho da janela em dias.
    black_list : iterable of str, optional
        ``app_package`` que não podem ser recomendados. Apps
        desconhecidos são ignorados.
    seed : int, optional
        Semente do desempate aleatório.

    Returns
    -------
    list[str]
        ``app_package`` recomendados, do melhor para o pior. Pode ter menos
        de ``n`` itens se não houver apps elegíveis suficientes
        (total > 0 e fora da black list), e é vazia se a janela for vazia.

    Examples
    --------
    >>> recommend(data, P, n=10, t="2024-03-15", w=7, black_list=["com.foo"], seed=42)
    ['com.bar', 'com.baz', ...]
    """
    if n < 1:
        raise ValueError(f"n deve ser >= 1, recebido {n}.")

    t_idx = date_to_index(data, t) if isinstance(t, (str, dt.date)) else int(t)
    exclude = blacklist_to_indices(data, black_list)

    ordem = rank_items(data.matrix, prefix, t_idx, w, n, seed=seed, exclude=exclude)
    return data.apps[ordem].tolist()
