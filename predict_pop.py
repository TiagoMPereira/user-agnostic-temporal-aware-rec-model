"""Pipeline de predicao do modelo de popularidade (Card 8).

Le o dataset de interacoes ja processado pelo feature engineering
(feature_engineering.py), que ja contem `interaction_rank` e `split`,
e a popularity_matrix (Card 4). Para cada interacao de teste, gera um
ranking por popularidade de N_RECS apps dentre os que o usuario ainda
nao havia consumido -- excluindo todo o historico anterior a ela
(treino, validacao e teste com timestamp menor).

A logica de pontuacao (soma cumulativa dentro da janela + tiebreak
aleatorio deterministico) e a mesma do models.pop_model.POPModel
(Card 8), mas roda inline em numpy em vez de chamar POPModel.predict()
uma vez por linha de teste. Essa chamada fazia, por interacao: um
`filter().tail(1)` na popularity_matrix inteira e a conversao de uma
linha larga (~10 mil colunas) para dict Python -- repetido DUAS vezes
quando ha janela (cum_until e cum_before_window). Para ~3.2M linhas de
teste isso e o gargalo real (nao o loop em si, que e o mesmo usado em
predict_random.py e roda rapido la).

Aqui, em vez disso:
  - a matriz e convertida para numpy uma unica vez;
  - o indice de linha correspondente a cada timestamp de teste e
    resolvido de uma so vez para TODAS as interacoes de teste via
    busca binaria vetorizada (np.searchsorted), nao uma busca por vez;
  - o app_package e codificado como inteiro (posicao no catalogo/
    colunas da matriz) para que o historico "ja consumido" seja uma
    mascara binaria e a pontuacao vire uma indexacao de array, sem
    lookups em dict por app;
  - o top-N e obtido com np.argpartition (custo O(catalogo), sem
    ordenar o catalogo inteiro) em vez de sorted() sobre todos os
    validos.

Ainda escreve em lotes de BATCH_SIZE linhas de teste diretamente no
parquet (via pyarrow.ParquetWriter), sem acumular o resultado inteiro
em memoria.

WINDOW pode ser definido por linha de comando (--window/-w). Sem
argumento, usa o default abaixo (90 dias). Exemplos:
    python predict_pop.py                # WINDOW = 90 (default)
    python predict_pop.py --window 180    # WINDOW = 180
    python predict_pop.py -w 365
    python predict_pop.py --window all    # POP-All (window=None)

ATENCAO: este script processa o dataset inteiro (~19M interacoes) e
NAO deve ser executado neste ambiente -- destina-se a rodar em outra
maquina.
"""

import argparse
import os

import numpy as np
import polars as pl
import pyarrow.parquet as pq

WINDOW = 90  # default: tamanho da janela em dias (None = POP-All, 90/180/365 = POP-3/6/12); sobrescrito por --window

INTERACTIONS_PATH = "data/processed/interactions_fe.parquet"
POPULARITY_MATRIX_PATH = "data/processed/popularity_matrix.parquet"
DATE_COL = "formated_date"
N_RECS = 250
BATCH_SIZE = 200_000  # linhas de teste por lote escrito no parquet, controla o pico de memoria
TIEBREAK_SEED = 42  # mesma semente do POPModel (Card 8, regra 4)
TIEBREAK_SCALE = 1e-6  # perturbacao do tiebreak: bem menor que 1 para nao afetar a ordem por score

REC_COLS = [f"rec{j:03d}" for j in range(N_RECS)]
SCHEMA = ["uid", "timestamp", *REC_COLS]
DTYPES = {col: pl.Utf8 for col in SCHEMA}


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


def main(window: int | None):
    output_path = f"data/predictions/pop_{window}.parquet"

    print(f"Lendo {POPULARITY_MATRIX_PATH}...")
    matrix = pl.read_parquet(POPULARITY_MATRIX_PATH)
    if matrix.schema[DATE_COL] != pl.Date:
        matrix = matrix.with_columns(pl.col(DATE_COL).str.to_date())
    matrix = matrix.sort(DATE_COL)

    catalog = [c for c in matrix.columns if c != DATE_COL]  # ja ordenado (Card 4)
    n_items = len(catalog)
    matrix_dates = matrix[DATE_COL].to_numpy()  # datetime64[D], ascendente
    matrix_values = matrix.drop(DATE_COL).to_numpy().astype(np.int64)  # (n_datas, n_items)

    print(f"Lendo {INTERACTIONS_PATH}...")
    df = pl.read_parquet(INTERACTIONS_PATH).sort(["uid", "interaction_rank"])
    df = df.with_columns(pl.col("app_package").cast(pl.Enum(catalog)))
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
    tiebreak = np.random.default_rng(TIEBREAK_SEED).random(n_items)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Gerando predicoes e salvando em lotes...")
    mask = bytearray(n_items)  # 1 = app ja consumido pelo usuario ate aqui
    zero_template = bytes(n_items)
    mask_view = np.frombuffer(mask, dtype=bool)

    batch: list = []
    writer = None
    total_written = 0
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
            composite = scores[candidate_idx] - tiebreak[candidate_idx] * TIEBREAK_SCALE

            k = min(N_RECS, candidate_idx.size)
            if k < candidate_idx.size:
                top_part = np.argpartition(-composite, k - 1)[:k]
            else:
                top_part = np.arange(candidate_idx.size)
            order = top_part[np.argsort(-composite[top_part])]
            top_idx = candidate_idx[order]

            preds = [catalog[c] for c in top_idx]
            preds += [None] * (N_RECS - len(preds))
            batch.append((uids[i], timestamps[i], *preds))
            test_ptr += 1

            if len(batch) >= BATCH_SIZE:
                writer = _flush(batch, writer, output_path)
                total_written += len(batch)
                print(f"  {total_written} linhas de teste processadas...")
                batch = []

        mask[codes[i]] = 1

    if batch:
        writer = _flush(batch, writer, output_path)
        total_written += len(batch)

    if writer is not None:
        writer.close()

    print("Validando...")
    expected = df.filter(pl.col("split") == "test").height
    assert total_written == expected, (
        f"esperado {expected} linhas de teste, obtido {total_written}"
    )
    print(f"OK: {total_written} linhas == {expected} interacoes de teste")

    print(f"Salvo em {output_path}")
    print("Concluido!")


def _flush(batch: list, writer, output_path: str) -> pq.ParquetWriter:
    """Converte um lote de linhas para colunar e escreve no parquet, sem
    acumular nada alem desse lote em memoria."""
    columns = dict(zip(SCHEMA, zip(*batch)))
    table = pl.DataFrame(columns, schema=DTYPES).to_arrow()

    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema)
    writer.write_table(table)
    return writer


if __name__ == "__main__":
    args = parse_args()
    main(args.window)
