"""Pipeline de predicao do modelo random (Card 9).

Le o dataset de interacoes ja processado pelo feature engineering
(feature_engineering.py), que ja contem `split`. Para cada interacao
de teste, gera um ranking aleatorio de N_RECS apps dentre os que o
usuario ainda nao havia consumido -- excluindo todo o historico
anterior a ela (treino e validacao).

A amostragem usa models.random_model.RandomModel (Card 7).

Com o split leave-one-out (Card 5), a linha de teste de cada usuario e
sempre a ultima cronologicamente -- entao o conjunto "ja consumido"
naquela linha e simplesmente todo o resto do historico do usuario
(train + val). Por isso esse conjunto e calculado de uma vez via
`group_by("uid")` do polars (vetorizado), e o loop em Python roda
apenas sobre as linhas de teste (1 por usuario) -- nao mais sobre o
dataset inteiro. Como ha apenas 1 interacao de teste por usuario, o
resultado tambem cabe inteiro em memoria e e escrito de uma vez (sem
batches).

O seed do RandomModel pode ser definido por linha de comando
(--seed/-s). Sem argumento, usa o default abaixo (42). Exemplos:
    python predict_random.py            # seed = 42 (default)
    python predict_random.py --seed 7
    python predict_random.py -s 123
"""

import argparse
import os

import polars as pl

from models import RandomModel

INPUT_PATH = "data/processed/interactions_fe.parquet"
SEED = 42  # default: semente do RandomModel; sobrescrito por --seed
N_RECS = 50

REC_COLS = [f"rec{j:03d}" for j in range(N_RECS)]
SCHEMA = ["uid", "timestamp", *REC_COLS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de predicao do modelo random (Card 9)."
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=SEED,
        help=f"Semente do RandomModel. Default: {SEED}.",
    )
    return parser.parse_args()


def main(seed: int, df: pl.DataFrame) -> pl.DataFrame:
    app_dtype = df.schema["app_package"]
    dtypes = {"uid": pl.Utf8, "timestamp": pl.Utf8, **{col: app_dtype for col in REC_COLS}}

    catalog = sorted(df["app_package"].unique().to_list())
    model = RandomModel(random_state=seed)

    print("Agregando apps consumidos por usuario (train+val)...")
    consumed = (
        df.filter(pl.col("split") != "test")
        .group_by("uid")
        .agg(pl.col("app_package").alias("consumed_apps"))
    )
    test_df = df.filter(pl.col("split") == "test").join(consumed, on="uid", how="left")

    uids = test_df["uid"].to_list()
    timestamps = test_df["formated_date"].to_list()
    consumed_lists = test_df["consumed_apps"].to_list()  # None ou lista de apps por usuario

    print("Gerando predicoes...")
    rows: list = []

    for uid, timestamp, consumed_apps in zip(uids, timestamps, consumed_lists):
        consumed_set = set(consumed_apps) if consumed_apps else set()
        valid_apps = [a for a in catalog if a not in consumed_set]
        preds = model.predict(valid_apps, N_RECS)
        preds += [None] * (N_RECS - len(preds))
        rows.append((uid, timestamp, *preds))

    print(f"OK: {len(rows)} linhas de teste processadas")

    columns = dict(zip(SCHEMA, zip(*rows)))
    return pl.DataFrame(columns, schema=dtypes)


if __name__ == "__main__":
    args = parse_args()

    print(f"Lendo {INPUT_PATH}...")
    df = pl.read_parquet(INPUT_PATH)

    predictions = main(args.seed, df)

    output_path = f"data/predictions/random_{args.seed}.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Salvando {output_path}...")
    predictions.write_parquet(output_path)

    print("Concluido!")
