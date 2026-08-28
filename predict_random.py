"""Pipeline de predicao do modelo random (Card 9).

Le o dataset de interacoes ja processado pelo feature engineering
(feature_engineering.py), que ja contem `interaction_rank` e `split`.
Para cada interacao de teste, gera um ranking aleatorio de N_RECS apps
dentre os que o usuario ainda nao havia consumido -- excluindo todo o
historico anterior a ela (treino, validacao e teste com timestamp
menor).

A amostragem usa models.random_model.RandomModel (Card 7).

Com o split leave-one-out (Card 5) ha apenas 1 interacao de teste por
usuario, entao o resultado cabe inteiro em memoria e e escrito de uma
vez (sem batches).

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

    uids = df["uid"].to_list()
    apps = df["app_package"].to_list()
    timestamps = df["formated_date"].to_list()
    splits = df["split"].to_list()

    print("Gerando predicoes...")
    consumed: set = set()
    current_uid = None
    rows: list = []

    for uid, app, timestamp, split in zip(uids, apps, timestamps, splits):
        if uid != current_uid:
            consumed = set()
            current_uid = uid

        if split == "test":
            valid_apps = [a for a in catalog if a not in consumed]
            preds = model.predict(valid_apps, N_RECS)
            preds += [None] * (N_RECS - len(preds))
            rows.append((uid, timestamp, *preds))

        consumed.add(app)

    print(f"OK: {len(rows)} linhas de teste processadas")

    columns = dict(zip(SCHEMA, zip(*rows)))
    return pl.DataFrame(columns, schema=dtypes)


if __name__ == "__main__":
    args = parse_args()

    print(f"Lendo {INPUT_PATH}...")
    df = pl.read_parquet(INPUT_PATH).sort(["uid", "interaction_rank"])

    predictions = main(args.seed, df)

    output_path = f"data/predictions/random_{args.seed}.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Salvando {output_path}...")
    predictions.write_parquet(output_path)

    print("Concluido!")
