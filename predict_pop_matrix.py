"""Predicao do split de validacao com o recomendador por popularidade em
janela deslizante (pacote ``pop_matrix``).

Le o dataset ja processado pelo feature engineering
(``feature_engineering.py``), que contem a coluna ``split``
(train/val/test, leave-one-out por usuario -- ``utils.add_data_split``).
O split de teste e removido logo no inicio e nunca e tocado. A matriz de
interacoes (``M``/``P``) e construida SOMENTE com o split de treino; a
predicao roda sobre o split de validacao, uma recomendacao por interacao
de val, excluindo (black list) os apps que o proprio usuario ja consumiu
no treino.

``interactions_fe.parquet`` ja vem no formato dia-a-dia
(``formated_date`` string "YYYY-MM-DD", mais a coluna ``split``) --
schema diferente do dataset bruto que ``pop_matrix.load_interactions``
espera (secao 2 das instrucoes de pop_matrix: uid/app_package/review/
timestamp epoch). Por isso a matriz e construida direto com
``build_interaction_data`` a partir de um ``LazyFrame`` so com
``app_package``/``date``, sem passar por ``load_interactions``.

Exemplos:
    python predict_pop_matrix.py --window 30
    python predict_pop_matrix.py --window 90 --n-recs 20 --seed 42
"""

import argparse
import os

import numpy as np
import polars as pl

from pop_matrix import build_blacklist_csr, build_interaction_data, build_prefix_sums, recommend_batch

INTERACTIONS_PATH = "data/processed/interactions_fe.parquet"
OUTPUT_DIR = "data/predictions/popularity_matrix"
N_RECS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--window",
        type=int,
        required=True,
        help="Tamanho da janela deslizante (dias) usada na contagem de popularidade.",
    )
    parser.add_argument("--n-recs", type=int, default=N_RECS, help=f"Numero de recomendacoes por usuario. Default: {N_RECS}.")
    parser.add_argument("--seed", type=int, default=None, help="Semente do desempate aleatorio (default: nao reprodutivel).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Lendo {INTERACTIONS_PATH}...")
    lf = pl.scan_parquet(INTERACTIONS_PATH)

    print("Removendo split de teste (nunca usado nesta predicao)...")
    lf = lf.filter(pl.col("split") != "test")

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
            pl.col("formated_date").str.to_date().alias("date"),
        )
        .join(consumed, on="uid", how="left")
        .collect()
    )
    print(f"OK: {val_df.height} interacoes de validacao")

    print("Calculando indices de dia (t) de cada interacao de validacao...")
    ts = (val_df["date"] - data.start_date).dt.total_days().cast(pl.Int64).to_numpy()
    # Usuarios "frios" (0 interacoes de treino, N=2 no split leave-one-out)
    # podem ter a unica interacao anterior ao val cair fora do calendario do
    # treino: `t` relativo a `data.start_date` fica negativo. A janela de
    # popularidade fica vazia de qualquer forma (nao ha treino antes de
    # start_date), entao o dia 0 e equivalente -- clip em vez de deixar
    # `recommend_batch`/`window_bounds` levantar ValueError (t < 0).
    ts = np.maximum(ts, 0)

    black_lists = val_df["consumed_apps"].to_list()
    bl_indptr, bl_indices = build_blacklist_csr(data, black_lists)

    print(f"Gerando recomendacoes (window={args.window}, n_recs={args.n_recs}, seed={args.seed})...")
    rankings = recommend_batch(
        data,
        prefix,
        n=args.n_recs,
        ts=ts,
        w=args.window,
        bl_indptr=bl_indptr,
        bl_indices=bl_indices,
        seed=args.seed,
    )

    print("Convertendo indices de app para app_package...")
    # rankings usa -1 para "sem recomendacao"; np.concatenate([apps, [None]])
    # + indexacao negativa faz -1 apontar pro None anexado no final, sem
    # precisar de um loop Python sobre as Q consultas.
    apps_or_none = np.concatenate([data.apps, np.array([None], dtype=object)])
    rec_values = apps_or_none[rankings]

    rec_cols = [f"rec{j:03d}" for j in range(args.n_recs)]
    predictions = pl.DataFrame(rec_values, schema=rec_cols)
    predictions = predictions.insert_column(0, val_df["date"].cast(pl.Utf8).alias("timestamp"))
    predictions = predictions.insert_column(0, val_df["uid"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f"{OUTPUT_DIR}/val_w{args.window}_n{args.n_recs}.parquet"
    print(f"Salvando {output_path}...")
    predictions.write_parquet(output_path)

    print(f"Concluido! {predictions.height} linhas de predicao salvas.")


if __name__ == "__main__":
    main()
    # predict_pop_matrix.py --window 30 --n-recs 10 --seed 42