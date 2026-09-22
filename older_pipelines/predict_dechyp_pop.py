"""Pipeline de predicao do modelo de popularidade com decaimento
hiperbolico (dechypPop).

Mesma logica de predicao do POPModel (predict_pop.py) e do decayPop
(predict_decay_pop.py): para cada interacao de teste, ranqueia N_RECS
apps dentre os que o usuario ainda nao havia consumido. A diferenca em
relacao ao decayPop esta apenas na funcao de decaimento usada para
pontuar os itens -- em vez do peso exponencial exp(-lambda * (t - ti)),
cada interacao passada contribui com um peso hiperbolico
1 / (1 + lambda * (t - ti)), onde `t` e a data de referencia (o dia
atual) e `ti` o dia em que a interacao ocorreu. O decaimento hiperbolico
cai mais devagar no longo prazo do que o exponencial (cauda mais pesada),
para o mesmo `lambda`.

Como a matriz depende de `lambda`, ela precisa ser construida em tempo
de execucao (utils.decay_popularity_matrix.build_hyperbolic_decay_popularity_matrix)
a partir das interacoes brutas (data/processed/interactions.parquet --
todas as interacoes, independente do rating, mesma fonte usada por
preprocess_interactions.py para o popularity_matrix.parquet do POPModel
sem decaimento), em vez de ler um parquet ja pre-calculado.

O restante do pipeline (leitura de interactions_fe.parquet, geracao das
predicoes de teste, desempate) reaproveita a mesma logica vetorizada e a
mesma funcao de ranking/desempate (`_rank_top_n`, models/pop_utils.py)
de predict_pop.py e predict_decay_pop.py. O catalogo de itens usado como
candidato e sempre o catalogo inteiro (sem corte), como em
optimize_dechyp_pop.py.

LAMBDA pode ser definido por linha de comando (--lambda/-l). Sem
argumento, usa o default abaixo. Exemplos:
    python predict_dechyp_pop.py                # LAMBDA = 0.01 (default)
    python predict_dechyp_pop.py --lambda 0.05
    python predict_dechyp_pop.py -l 0.2
"""

import argparse
import os

import numpy as np
import polars as pl

from models.pop_utils import _rank_top_n
from utils.decay_popularity_matrix import build_hyperbolic_decay_popularity_matrix

LAMBDA = 0.01  # default: fator de decaimento hiperbolico; sobrescrito por --lambda

INTERACTIONS_PATH = "data/processed/interactions_fe.parquet"
RAW_INTERACTIONS_PATH = "data/processed/interactions.parquet"
DATE_COL = "formated_date"
N_RECS = 50

REC_COLS = [f"rec{j:03d}" for j in range(N_RECS)]
SCHEMA = ["uid", "timestamp", *REC_COLS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de predicao do modelo de popularidade com decaimento hiperbolico (dechypPop)."
    )
    parser.add_argument(
        "--lambda",
        "-l",
        dest="lambda_",
        type=float,
        default=LAMBDA,
        help=f"Fator de decaimento hiperbolico (0.001 a 1). Default: {LAMBDA}.",
    )
    return parser.parse_args()


def main(df: pl.DataFrame, matrix: pl.DataFrame) -> pl.DataFrame:
    catalog = [c for c in matrix.columns if c != DATE_COL]  # ja ordenado, nomes de coluna sao sempre str
    n_items = len(catalog)
    matrix_dates = matrix[DATE_COL].to_numpy()  # datetime64[D], ascendente
    matrix_values = matrix.drop(DATE_COL).to_numpy().astype(np.float64)  # (n_datas, n_items)

    app_dtype = df.schema["app_package"]
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

    zero_row = np.zeros(n_items, dtype=np.float64)

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
            scores = matrix_values[idx_until] if idx_until >= 0 else zero_row

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

    print(f"Lendo {RAW_INTERACTIONS_PATH}...")
    raw_df = pl.read_parquet(RAW_INTERACTIONS_PATH)

    print(f"Construindo matriz de popularidade com decaimento hiperbolico (lambda={args.lambda_})...")
    matrix = build_hyperbolic_decay_popularity_matrix(raw_df, args.lambda_).sort(DATE_COL)
    if matrix.schema[DATE_COL] != pl.Date:
        matrix = matrix.with_columns(pl.col(DATE_COL).str.to_date())
    del raw_df

    print(f"Lendo {INTERACTIONS_PATH}...")
    df = pl.read_parquet(INTERACTIONS_PATH).sort(["uid", "interaction_rank"])

    predictions = main(df, matrix)

    output_path = f"data/predictions/dechyp_pop_{args.lambda_}.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Salvando {output_path}...")
    predictions.write_parquet(output_path)

    print("Concluido!")
