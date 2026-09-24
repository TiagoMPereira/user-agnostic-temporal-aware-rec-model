"""Conversões para inspeção manual (seção 6.5 de ``pop_matrix/instructions.md``).

Não faz parte do caminho de recomendação -- apenas materializa recortes de
``M``/``P``/matriz acumulada como ``pl.DataFrame`` largo, legível por
humanos durante depuração.
"""

from __future__ import annotations

import datetime as dt
from typing import Sequence

import numpy as np
import polars as pl

from .data import InteractionData

_MAX_APPS_SEM_RECORTE = 1000


def matrix_to_polars(
    data: InteractionData,
    matrix: np.ndarray | None = None,
    day_range: tuple[int, int] | None = None,
    apps: Sequence[str] | None = None,
) -> pl.DataFrame:
    """
    Converte um recorte da matriz (``M`` ou acumulada) em ``pl.DataFrame``
    indexado por data, com uma coluna por app, apenas para inspeção visual.

    Não usar em produção: com muitos apps a tabela larga é custosa.
    Exigir ``day_range`` ou ``apps`` quando ``A > 1000``.

    Parameters
    ----------
    data : InteractionData
        Mapeamentos de índice (``apps``, ``app_to_idx``, ``start_date``).
    matrix : np.ndarray, optional
        Matriz a converter, forma ``(L, A)``. Padrão ``data.matrix``.
        Também aceita uma matriz acumulada de forma ``(T + 1, A)``
        (``build_accumulated_matrix``) -- a linha ``i`` sempre corresponde
        à data ``start_date + i`` dias, qualquer que seja a origem.
    day_range : tuple[int, int], optional
        ``(inicio, fim)`` de linhas a incluir, mesma convenção de
        ``ranking.window_bounds`` (``fim`` exclusivo). Padrão: todas as
        linhas de ``matrix``.
    apps : sequence of str, optional
        Subconjunto de ``app_package`` a incluir, na ordem dada. Apps
        desconhecidos são ignorados. Padrão: todos os apps de ``data``.

    Returns
    -------
    pl.DataFrame
        Coluna ``date`` (``pl.Date``) mais uma coluna por app selecionado.

    Raises
    ------
    ValueError
        Se ``day_range`` e ``apps`` forem ambos ``None`` e ``data.n_apps``
        for maior que 1000 (recorte largo demais para inspeção).
    """
    if matrix is None:
        matrix = data.matrix

    if day_range is None and apps is None and data.n_apps > _MAX_APPS_SEM_RECORTE:
        raise ValueError(
            f"data.n_apps={data.n_apps} > {_MAX_APPS_SEM_RECORTE}; "
            "especifique 'day_range' ou 'apps' para limitar o recorte "
            "(matrix_to_polars não deve ser usada em produção)."
        )

    row_start, row_end = day_range if day_range is not None else (0, matrix.shape[0])

    if apps is not None:
        col_names = [app for app in apps if app in data.app_to_idx]
        col_idx = [data.app_to_idx[app] for app in col_names]
    else:
        col_names = data.apps.tolist()
        col_idx = list(range(data.n_apps))

    values = matrix[row_start:row_end][:, col_idx]
    dates = [data.start_date + dt.timedelta(days=i) for i in range(row_start, row_end)]

    df = pl.DataFrame(values, schema=col_names)
    return df.insert_column(0, pl.Series("date", dates))
