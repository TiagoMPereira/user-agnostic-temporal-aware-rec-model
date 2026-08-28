"""Pipeline de predicao do modelo de popularidade (Card 8).

Le o dataset de interacoes ja processado pelo feature engineering
(feature_engineering.py), que ja contem `interaction_rank` e `split`,
e a popularity_matrix (Card 4). Para cada interacao de teste, gera um
ranking por popularidade de N_RECS apps dentre os que o usuario ainda
nao havia consumido -- excluindo todo o historico anterior a ela
(treino, validacao e teste com timestamp menor).

A logica de pontuacao (soma cumulativa dentro da janela) e a mesma do
models.pop_model.POPModel (Card 8), mas roda inline em numpy em vez de
chamar POPModel.predict() uma vez por linha de teste. Essa chamada
fazia, por interacao: um `filter().tail(1)` na popularity_matrix
inteira e a conversao de uma linha larga (~10 mil colunas) para dict
Python -- repetido DUAS vezes quando ha janela (cum_until e
cum_before_window).

Aqui, em vez disso:
  - a matriz e convertida para numpy uma unica vez;
  - o indice de linha correspondente a cada timestamp de teste e
    resolvido de uma so vez para TODAS as interacoes de teste via
    busca binaria vetorizada (np.searchsorted), nao uma busca por vez;
  - o app_package e codificado como inteiro (posicao no catalogo/
    colunas da matriz) para que o historico "ja consumido" seja uma
    mascara binaria e a pontuacao vire uma indexacao de array, sem
    lookups em dict por app;
  - o top-N e obtido com np.partition (custo O(catalogo), sem ordenar
    o catalogo inteiro) em vez de sorted() sobre todos os validos.

Desempate: itens com o mesmo score sao desempatados pela popularidade
diaria (nao cumulativa) do dia imediatamente anterior a reference_date
-- independente da janela usada no score principal. Se persistir o
empate, a regra e aplicada regressivamente (dia anterior a esse, e
assim por diante) ate resolver ou esgotar o historico da matrix, caso
em que o empate remanescente e resolvido pela ordem alfabetica do
app_package (equivalente a ordenar pelo codigo inteiro do catalogo,
que ja e alfabetico -- ver Card 4).

Com o split leave-one-out (Card 5) ha apenas 1 interacao de teste por
usuario, entao o resultado cabe inteiro em memoria e e escrito de uma
vez (sem batches).

WINDOW pode ser definido por linha de comando (--window/-w). Sem
argumento, usa o default abaixo (90 dias). Exemplos:
    python predict_pop.py                # WINDOW = 90 (default)
    python predict_pop.py --window 180    # WINDOW = 180
    python predict_pop.py -w 365
    python predict_pop.py --window all    # POP-All (window=None)
"""

import argparse
import os

import numpy as np
import polars as pl

WINDOW = 90  # default: tamanho da janela em dias (None = POP-All, 90/180/365 = POP-3/6/12); sobrescrito por --window

INTERACTIONS_PATH = "data/processed/interactions_fe.parquet"
POPULARITY_MATRIX_PATH = "data/processed/popularity_matrix.parquet"
DATE_COL = "formated_date"
N_RECS = 50

REC_COLS = [f"rec{j:03d}" for j in range(N_RECS)]
SCHEMA = ["uid", "timestamp", *REC_COLS]


def parse_window(value: str) -> int | None:
    if value.lower() in ("all", "none"):
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de predicao do modelo de popularidade (Card 8)."
    )
    parser.add_argument(
        "--window",
        "-w",
        type=parse_window,
        default=WINDOW,
        help=(
            "Tamanho da janela em dias (ex: 90, 180, 365) ou 'all' para "
            f"POP-All. Default: {WINDOW}."
        ),
    )
    return parser.parse_args()


def _break_ties(items: np.ndarray, matrix_values: np.ndarray, idx_until: int) -> np.ndarray:
    """Desempata `items` (codigos de app com o mesmo score), regressivamente,
    pela popularidade diaria dos dias anteriores a reference_date.

    Tier 1 compara a contagem diaria do dia em matrix[idx_until - 1]
    (o dia imediatamente anterior a reference_date); se persistir o
    empate, tier 2 compara o dia anterior a esse, e assim por diante.
    Ao esgotar o historico da matrix, o empate remanescente e resolvido
    pelo codigo do item -- que corresponde a ordem alfabetica do
    catalogo (ja ordenado, ver Card 4).
    """
    groups = [items]
    tier = 1

    while any(g.size > 1 for g in groups):
        lo = idx_until - tier
        new_groups = []

        for g in groups:
            if g.size <= 1:
                new_groups.append(g)
                continue

            if lo < 0:
                # historico esgotado: cada item vira seu proprio grupo, na
                # ordem do codigo (== ordem alfabetica do catalogo, Card 4)
                new_groups.extend(np.split(np.sort(g), np.arange(1, g.size)))
                continue

            daily = matrix_values[lo + 1, g] - matrix_values[lo, g]
            order = np.argsort(-daily, kind="stable")
            g_sorted = g[order]
            daily_sorted = daily[order]
            boundaries = np.flatnonzero(np.diff(daily_sorted) != 0) + 1
            new_groups.extend(np.split(g_sorted, boundaries))

        groups = new_groups
        tier += 1

    return np.concatenate(groups)


def _rank_top_n(
    candidate_idx: np.ndarray,
    scores: np.ndarray,
    matrix_values: np.ndarray,
    idx_until: int,
    n: int,
) -> np.ndarray:
    """Retorna ate `n` codigos de `candidate_idx`, do mais para o menos
    relevante, ordenados por score (descendente) com desempate
    regressivo por popularidade diaria (ver `_break_ties`).
    """
    candidate_scores = scores[candidate_idx]

    if candidate_idx.size > n:
        threshold = np.partition(candidate_scores, -n)[-n]
        keep = candidate_scores >= threshold
        candidate_idx = candidate_idx[keep]
        candidate_scores = candidate_scores[keep]

    order = np.argsort(-candidate_scores, kind="stable")
    sorted_idx = candidate_idx[order]
    sorted_scores = candidate_scores[order]

    boundaries = np.flatnonzero(np.diff(sorted_scores) != 0) + 1
    groups = np.split(sorted_idx, boundaries)

    resolved = [
        _break_ties(g, matrix_values, idx_until) if g.size > 1 else g for g in groups
    ]
    return np.concatenate(resolved)[:n]


def main(window: int | None, df: pl.DataFrame, matrix: pl.DataFrame) -> pl.DataFrame:
    catalog = [c for c in matrix.columns if c != DATE_COL]  # ja ordenado (Card 4), nomes de coluna sao sempre str
    n_items = len(catalog)
    matrix_dates = matrix[DATE_COL].to_numpy()  # datetime64[D], ascendente
    matrix_values = matrix.drop(DATE_COL).to_numpy().astype(np.int64)  # (n_datas, n_items)

    app_dtype = df.schema["app_package"]
    # catalog vem dos nomes de coluna da matrix (sempre str); convertido de volta
    # para o dtype real de app_package para casar por valor no cast pra Enum
    # (cast direto de Int64 pra Enum reinterpreta o int como codigo/posicao, nao
    # como valor) e para os ids de saida terem o mesmo tipo do ground truth.
    catalog_native = pl.Series(catalog, dtype=pl.Utf8).cast(app_dtype).to_list()
    df = df.with_columns(pl.col("app_package").cast(pl.Utf8).cast(pl.Enum(catalog)))
    df = df.with_columns(
        (pl.col("uid") != pl.col("uid").shift(1)).fill_null(True).alias("_new_user")
    )

    codes = df["app_package"].to_physical().to_numpy()
    is_new_user = df["_new_user"].to_numpy()
    is_test = (df["split"] == "test").to_numpy()
    uids = df["uid"].to_list()
    timestamps = df[DATE_COL].to_list()  # strings "YYYY-MM-DD", vao direto pro output

    print("Pre-calculando indices de data (vetorizado)...")
    test_dates = df.filter(pl.col("split") == "test")[DATE_COL].str.to_date().to_numpy()
    idx_until_all = np.searchsorted(matrix_dates, test_dates, side="right") - 1
    if window is not None:
        window_start = test_dates - np.timedelta64(window, "D")
        idx_before_all = np.searchsorted(matrix_dates, window_start, side="right") - 1
    else:
        idx_before_all = None

    zero_row = np.zeros(n_items, dtype=np.int64)

    print("Gerando predicoes...")
    mask = bytearray(n_items)  # 1 = app ja consumido pelo usuario ate aqui
    zero_template = bytes(n_items)
    mask_view = np.frombuffer(mask, dtype=bool)

    rows: list = []
    test_ptr = 0
    n_rows = df.height

    for i in range(n_rows):
        if is_new_user[i]:
            mask[:] = zero_template

        if is_test[i]:
            idx_until = idx_until_all[test_ptr]
            row_until = matrix_values[idx_until] if idx_until >= 0 else zero_row

            if window is not None:
                idx_before = idx_before_all[test_ptr]
                row_before = matrix_values[idx_before] if idx_before >= 0 else zero_row
                scores = row_until - row_before
            else:
                scores = row_until

            candidate_idx = np.flatnonzero(~mask_view)
            top_idx = _rank_top_n(candidate_idx, scores, matrix_values, idx_until, N_RECS)

            preds = [catalog_native[c] for c in top_idx]
            preds += [None] * (N_RECS - len(preds))
            rows.append((uids[i], timestamps[i], *preds))
            test_ptr += 1

        mask[codes[i]] = 1

    print(f"OK: {len(rows)} linhas de teste processadas")

    dtypes = {"uid": pl.Utf8, "timestamp": pl.Utf8, **{col: app_dtype for col in REC_COLS}}
    columns = dict(zip(SCHEMA, zip(*rows)))
    return pl.DataFrame(columns, schema=dtypes)


if __name__ == "__main__":
    args = parse_args()

    print(f"Lendo {POPULARITY_MATRIX_PATH}...")
    matrix = pl.read_parquet(POPULARITY_MATRIX_PATH)
    if matrix.schema[DATE_COL] != pl.Date:
        matrix = matrix.with_columns(pl.col(DATE_COL).str.to_date())
    matrix = matrix.sort(DATE_COL)

    print(f"Lendo {INTERACTIONS_PATH}...")
    df = pl.read_parquet(INTERACTIONS_PATH).sort(["uid", "interaction_rank"])

    predictions = main(args.window, df, matrix)

    output_path = f"data/predictions/pop_{args.window}.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Salvando {output_path}...")
    predictions.write_parquet(output_path)

    print("Concluido!")
