"""Carregamento, índices e matrizes de interação.

Implementa a seção 6.1 de ``pop_matrix/instructions.md``: leitura lazy do
dataset de interações, construção da matriz densa ``M`` (dias x apps) e da
matriz de somas de prefixos ``P``, além das conversões auxiliares de data e
black list usadas pelo ranking (``pop_matrix/ranking.py``).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import polars as pl

_TIMESTAMP_UNITS = ("s", "ms", "us", None)


@dataclass(frozen=True)
class InteractionData:
    """
    Contêiner com a matriz de interações e os mapeamentos de índice.

    Attributes
    ----------
    matrix : np.ndarray
        Matriz de interações ``M`` de forma ``(T, A)`` e dtype ``int32``.
        ``matrix[d, a]`` é o número de interações do app ``a`` no dia ``d``.
        Dias sem interações existem como linhas de zeros (calendário contínuo).
    apps : np.ndarray
        Array de strings de forma ``(A,)``, ordenado alfabeticamente.
        ``apps[a]`` é o ``app_package`` do índice ``a``.
    start_date : datetime.date
        Data correspondente à linha 0 da matriz.
    app_to_idx : dict[str, int]
        Mapeamento ``app_package -> app_idx``. Construído a partir de ``apps``.

    Notes
    -----
    ``T = matrix.shape[0]`` é o número de dias do calendário contínuo entre
    a primeira e a última data do dataset (inclusive).
    ``A = matrix.shape[1]`` é o número de apps distintos.
    """

    matrix: np.ndarray
    apps: np.ndarray
    start_date: dt.date
    app_to_idx: dict[str, int]

    @property
    def n_days(self) -> int:
        """int: número de dias do calendário contínuo (``T``)."""
        return self.matrix.shape[0]

    @property
    def n_apps(self) -> int:
        """int: número de apps distintos (``A``)."""
        return self.matrix.shape[1]


def load_interactions(
    path: str,
    timestamp_unit: str | None = "s",
    timezone: str = "UTC",
) -> pl.LazyFrame:
    """
    Carrega o dataset de interações de forma lazy e normaliza o tempo em dias.

    Seleciona apenas ``app_package`` e ``timestamp`` (e ``uid``, mantido
    para uso futuro em avaliação). A coluna ``review`` é descartada.

    Parameters
    ----------
    path : str
        Caminho para o arquivo. Extensão ``.parquet`` usa ``pl.scan_parquet``;
        ``.csv`` usa ``pl.scan_csv``. Outras extensões -> ``ValueError``.
    timestamp_unit : {"s", "ms", "us", None}, default "s"
        Unidade do timestamp quando ele é epoch numérico. Use ``None`` quando
        a coluna já é do tipo datetime.
    timezone : str, default "UTC"
        Fuso horário usado antes de truncar para o dia.

    Returns
    -------
    pl.LazyFrame
        Colunas: ``uid``, ``app_package`` (Utf8) e ``date`` (pl.Date).

    Notes
    -----
    Nenhum dado é materializado aqui; o plano é executado apenas no
    ``collect`` de ``build_interaction_data``.
    """
    if timestamp_unit not in _TIMESTAMP_UNITS:
        raise ValueError(f"timestamp_unit deve ser um de {_TIMESTAMP_UNITS}, recebido {timestamp_unit!r}.")

    if path.endswith(".parquet"):
        lf = pl.scan_parquet(path)
    elif path.endswith(".csv"):
        lf = pl.scan_csv(path)
    else:
        raise ValueError(f"Extensão de arquivo não suportada: {path!r}. Use .parquet ou .csv.")

    timestamp_expr = pl.col("timestamp")
    if timestamp_unit is not None:
        timestamp_expr = pl.from_epoch(timestamp_expr, time_unit=timestamp_unit)

    date_expr = timestamp_expr.dt.convert_time_zone(timezone).dt.date().alias("date")

    return lf.select(
        pl.col("uid"),
        pl.col("app_package").cast(pl.Utf8),
        date_expr,
    )


def build_interaction_data(lf: pl.LazyFrame) -> InteractionData:
    """
    Constrói a matriz de interações densa ``M`` (dias x apps) a partir das
    interações em formato longo.

    Passos
    ------
    1. Obter ``start_date = min(date)`` e ``end_date = max(date)``.
    2. Calcular ``day_idx = (date - start_date).dt.total_days()`` (Int32).
    3. Calcular ``app_idx`` como rank denso alfabético de ``app_package``
       menos 1 (Int32), e ``apps`` como os ``app_package`` únicos ordenados.
    4. ``group_by(["day_idx", "app_idx"]).agg(pl.len().alias("count"))``
       e ``collect`` (usar o engine streaming se disponível).
    5. Alocar ``M = np.zeros((T, A), dtype=np.int32)`` com
       ``T = (end_date - start_date).days + 1``.
    6. Atribuir ``M[day_idx, app_idx] = count`` (pares são únicos;
       não usar ``np.add.at``).

    Parameters
    ----------
    lf : pl.LazyFrame
        Saída de ``load_interactions``.

    Returns
    -------
    InteractionData
        Matriz ``M`` e mapeamentos de índice.

    Raises
    ------
    ValueError
        Se o dataset estiver vazio.

    Notes
    -----
    Memória: ``T * A * 4`` bytes. Ex.: 3.000 dias x 50.000 apps ≈ 600 MB.
    Logar ``T``, ``A`` e o tamanho estimado antes de alocar.
    """
    stats = lf.select(
        pl.len().alias("n_rows"),
        pl.col("date").min().alias("start_date"),
        pl.col("date").max().alias("end_date"),
    ).collect()

    if stats["n_rows"][0] == 0:
        raise ValueError("O dataset de interações está vazio.")

    start_date: dt.date = stats["start_date"][0]
    end_date: dt.date = stats["end_date"][0]
    n_days = (end_date - start_date).days + 1

    apps = lf.select(pl.col("app_package").unique().sort()).collect()["app_package"].to_numpy()
    n_apps = len(apps)

    estimated_bytes = n_days * n_apps * 4
    print(
        f"build_interaction_data: T={n_days} dias, A={n_apps} apps, "
        f"matriz M estimada em {estimated_bytes / 2**20:.1f} MiB."
    )

    grouped = (
        lf.select(
            (pl.col("date") - start_date).dt.total_days().cast(pl.Int32).alias("day_idx"),
            (pl.col("app_package").rank(method="dense").cast(pl.Int32) - 1).alias("app_idx"),
        )
        .group_by(["day_idx", "app_idx"])
        .agg(pl.len().alias("count"))
        .collect(engine="streaming")
    )

    matrix = np.zeros((n_days, n_apps), dtype=np.int32)
    matrix[grouped["day_idx"].to_numpy(), grouped["app_idx"].to_numpy()] = grouped["count"].to_numpy()

    app_to_idx = dict(zip(apps.tolist(), range(n_apps)))

    return InteractionData(matrix=matrix, apps=apps, start_date=start_date, app_to_idx=app_to_idx)


def build_prefix_sums(matrix: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de somas de prefixos ``P`` ao longo do tempo.

    ``P[0] = 0`` e ``P[k] = M[0] + ... + M[k-1]``. Com ela, o total de
    interações de qualquer janela ``[i, j)`` é ``P[j] - P[i]``, em O(A),
    para qualquer tamanho de janela.

    Parameters
    ----------
    matrix : np.ndarray
        Matriz ``M`` de forma ``(T, A)``.

    Returns
    -------
    np.ndarray
        Matriz ``P`` de forma ``(T + 1, A)`` e dtype ``int64``
        (int64 evita overflow nas somas acumuladas).

    Notes
    -----
    Implementar com ``np.cumsum(matrix, axis=0, dtype=np.int64)`` escrito
    em ``P[1:]`` de um array pré-alocado com zeros. Não usar loops.
    """
    n_days, n_apps = matrix.shape
    prefix = np.zeros((n_days + 1, n_apps), dtype=np.int64)
    np.cumsum(matrix, axis=0, dtype=np.int64, out=prefix[1:])
    return prefix


def build_accumulated_matrix(prefix: np.ndarray, w: int) -> np.ndarray:
    """
    Materializa a matriz de interações acumuladas para uma janela fixa ``w``.

    ``acc[t, a]`` = interações do app ``a`` nos dias ``[t - w, t - 1]``
    (janela truncada no dia 0). A linha ``t`` NÃO inclui o dia ``t``.

    Parameters
    ----------
    prefix : np.ndarray
        Matriz ``P`` de forma ``(T + 1, A)``.
    w : int
        Tamanho da janela em dias (``w >= 1``).

    Returns
    -------
    np.ndarray
        Matriz de forma ``(T + 1, A)``, dtype ``int64``. A linha ``T``
        corresponde ao dia seguinte ao último dia do dataset.

    Notes
    -----
    Vetorizado: ``acc = P - P[np.maximum(np.arange(T + 1) - w, 0)]``.
    A recomendação NÃO depende desta função; ela usa ``P`` diretamente.
    """
    if w < 1:
        raise ValueError(f"w deve ser >= 1, recebido {w}.")

    n_rows = prefix.shape[0]
    window_start = np.maximum(np.arange(n_rows) - w, 0)
    return prefix - prefix[window_start]


def date_to_index(data: InteractionData, date: dt.date | dt.datetime | str) -> int:
    """
    Converte uma data no índice de dia ``t`` usado pelo recomendador.

    Parameters
    ----------
    data : InteractionData
    date : date, datetime ou str ISO ("YYYY-MM-DD")
        Datetimes são truncados para o dia.

    Returns
    -------
    int
        ``(date - data.start_date).days``. Pode ser >= ``T`` (datas após
        o fim do dataset são válidas). Negativo -> ``ValueError``.
    """
    if isinstance(date, str):
        date = dt.date.fromisoformat(date)
    elif isinstance(date, dt.datetime):
        date = date.date()
    elif not isinstance(date, dt.date):
        raise TypeError(f"date deve ser date, datetime ou str ISO, recebido {type(date)!r}.")

    index = (date - data.start_date).days
    if index < 0:
        raise ValueError(f"date {date} é anterior a start_date ({data.start_date}).")
    return index


def blacklist_to_indices(data: InteractionData, black_list: Iterable[str] | None) -> np.ndarray:
    """
    Converte uma lista de ``app_package`` em índices de app.

    Apps desconhecidos (fora de ``data.apps``) são ignorados silenciosamente.

    Returns
    -------
    np.ndarray
        Índices únicos e ordenados, dtype ``int32``. Array vazio se
        ``black_list`` for ``None`` ou vazia.
    """
    if black_list is None:
        return np.empty(0, dtype=np.int32)

    indices = {data.app_to_idx[app] for app in black_list if app in data.app_to_idx}
    if not indices:
        return np.empty(0, dtype=np.int32)

    return np.array(sorted(indices), dtype=np.int32)
