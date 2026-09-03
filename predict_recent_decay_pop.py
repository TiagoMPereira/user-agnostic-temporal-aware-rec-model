"""Pipeline de predicao do modelo de popularidade recente com decaimento
exponencial (recentDecayPop).

Combina os dois mecanismos ja implementados: a janela de recencia do
POPModel (predict_pop.py) e o decaimento exponencial do decayPop
(predict_decay_pop.py). Em vez da soma cumulativa bruta dentro da janela
(POPModel) ou do decaimento sobre TODO o historico (decayPop), o score
de cada item e a soma dos pesos exp(-lambda * (d_until - ti)) das
interacoes ocorridas dentro da janela [d_before, d_until), onde
`d_until` e a data valida mais recente <= reference_date na matriz (a
mesma data que qualquer predicao ja "snapa" para le-la -- mesmo criterio
do decayPop) e `d_before` e a data valida mais recente <= (reference_date
- window).

Como a matriz de decaimento completo
(`utils.decay_popularity_matrix.build_decay_popularity_matrix`) ja
acumula o peso exponencial de TODO o historico anterior a cada data
(D(d)), o score windowed e obtido subtraindo a parcela anterior a
janela, reescalada para `d_until`:

    score = D(d_until) - exp(-lambda * gap) * D(d_before)

onde gap = d_until - d_before (em dias, o gap REAL entre as duas linhas
da matriz -- nao `window` nominal, ja que a matriz so tem linha para
datas com >=1 interacao e pode haver buracos no calendario). Ancorar o
gap em `d_until` (em vez da reference_date literal) e essencial para a
subtracao ser algebricamente exata; a derivacao completa esta em
optimize_recent_decay_pop.py, onde a formula foi validada por forca
bruta antes de ser usada na busca dos hiperparametros. `window=None`
(ou 'all') desliga o corte de janela -- reduz ao decayPop puro.

O restante do pipeline (leitura de interactions_fe.parquet, geracao das
predicoes de teste, desempate) reaproveita a mesma logica vetorizada e a
mesma funcao de ranking/desempate (`_rank_top_n`, models/pop_utils.py)
de predict_pop.py e predict_decay_pop.py. O catalogo de itens candidato
e sempre o catalogo inteiro (sem corte), como em
optimize_recent_decay_pop.py.

WINDOW e LAMBDA podem ser definidos por linha de comando (--window/-w e
--lambda/-l). Sem argumento, usam os defaults abaixo. Exemplos:
    python predict_recent_decay_pop.py                         # WINDOW=90, LAMBDA=0.01 (defaults)
    python predict_recent_decay_pop.py --window 180 --lambda 0.05
    python predict_recent_decay_pop.py -w 365 -l 0.2
    python predict_recent_decay_pop.py --window all             # sem corte de janela (== decayPop)
"""

import argparse
import os

import numpy as np
import polars as pl

from models.pop_utils import _rank_top_n
from utils.decay_popularity_matrix import build_decay_popularity_matrix

WINDOW = 90  # default: tamanho da janela em dias (None = sem corte); sobrescrito por --window
LAMBDA = 0.01  # default: fator de decaimento exponencial; sobrescrito por --lambda

INTERACTIONS_PATH = "data/processed/interactions_fe.parquet"
RAW_INTERACTIONS_PATH = "data/processed/interactions.parquet"
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
        description="Pipeline de predicao do modelo de popularidade recente com decaimento exponencial (recentDecayPop)."
    )
    parser.add_argument(
        "--window",
        "-w",
        type=parse_window,
        default=WINDOW,
        help=(
            "Tamanho da janela em dias (ex: 90, 180, 365) ou 'all' para nao "
            f"aplicar corte de janela (== decayPop). Default: {WINDOW}."
        ),
    )
    parser.add_argument(
        "--lambda",
        "-l",
        dest="lambda_",
        type=float,
        default=LAMBDA,
        help=f"Fator de decaimento exponencial (0.001 a 1). Default: {LAMBDA}.",
    )
    return parser.parse_args()


def main(window: int | None, lambda_: float, df: pl.DataFrame, matrix: pl.DataFrame) -> pl.DataFrame:
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

    if window is not None:
        window_start = test_dates - np.timedelta64(window, "D")
        idx_before_all = np.searchsorted(matrix_dates, window_start, side="right") - 1
        decay_factor_all = np.zeros(len(test_dates), dtype=np.float64)
        valid_before = idx_before_all >= 0
        # gap ancorado em matrix_dates[idx_until] -- a data da linha que
        # row_until de fato usa (ver derivacao no docstring do modulo) --,
        # NAO em test_dates: idx_before <= idx_until sempre (window >= 0),
        # entao onde valid_before e True, idx_until_all tambem e >= 0.
        d_until_valid = matrix_dates[idx_until_all[valid_before]]
        gap_days = (
            (d_until_valid - matrix_dates[idx_before_all[valid_before]])
            .astype("timedelta64[D]")
            .astype(np.int64)
        )
        decay_factor_all[valid_before] = np.exp(-lambda_ * gap_days)
    else:
        idx_before_all = None
        decay_factor_all = None

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
            row_until = matrix_values[idx_until] if idx_until >= 0 else zero_row

            if window is not None:
                idx_before = idx_before_all[test_ptr]
                if idx_before >= 0:
                    row_before = matrix_values[idx_before] * decay_factor_all[test_ptr]
                else:
                    row_before = zero_row
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

    print(f"Lendo {RAW_INTERACTIONS_PATH}...")
    raw_df = pl.read_parquet(RAW_INTERACTIONS_PATH)

    print(f"Construindo matriz de popularidade com decaimento (lambda={args.lambda_})...")
    matrix = build_decay_popularity_matrix(raw_df, args.lambda_).sort(DATE_COL)
    if matrix.schema[DATE_COL] != pl.Date:
        matrix = matrix.with_columns(pl.col(DATE_COL).str.to_date())
    del raw_df

    print(f"Lendo {INTERACTIONS_PATH}...")
    df = pl.read_parquet(INTERACTIONS_PATH).sort(["uid", "interaction_rank"])

    predictions = main(args.window, args.lambda_, df, matrix)

    window_label = args.window if args.window is not None else "all"
    output_path = f"data/predictions/recent_decay_pop_{window_label}_{args.lambda_}.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Salvando {output_path}...")
    predictions.write_parquet(output_path)

    print("Concluido!")
