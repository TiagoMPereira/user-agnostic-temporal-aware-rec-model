"""Pipeline de predicao do modelo random (Card 9).

Le o dataset de interacoes ja processado pelo feature engineering
(feature_engineering.py), que ja contem `interaction_rank` e `split`.
Para cada interacao de teste, gera um ranking aleatorio de N_RECS apps
dentre os que o usuario ainda nao havia consumido -- excluindo todo o
historico anterior a ela (treino, validacao e teste com timestamp
menor).

A amostragem usa models.random_model.RandomModel (Card 7).

ATENCAO: este script processa o dataset inteiro (~19M interacoes) e
NAO deve ser executado neste ambiente -- destina-se a rodar em outra
maquina.
"""

import polars as pl

from models import RandomModel

INPUT_PATH = "data/processed/interactions_fe.parquet"
OUTPUT_PATH = "data/predictions/random.parquet"
N_RECS = 250


def main():
    print(f"Lendo {INPUT_PATH}...")
    df = pl.read_parquet(INPUT_PATH).sort(["uid", "interaction_rank"])

    catalog = sorted(df["app_package"].unique().to_list())
    model = RandomModel()

    uids = df["uid"].to_list()
    apps = df["app_package"].to_list()
    timestamps = df["formated_date"].to_list()
    splits = df["split"].to_list()

    rec_cols = [f"rec{j:03d}" for j in range(N_RECS)]
    rows = []

    print("Gerando predicoes...")
    consumed: set = set()
    current_uid = None

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

    print("Montando dataframe de saida...")
    out_df = pl.DataFrame(rows, schema=["uid", "timestamp", *rec_cols], orient="row")

    print("Validando...")
    expected = df.filter(pl.col("split") == "test").height
    assert out_df.height == expected, (
        f"esperado {expected} linhas de teste, obtido {out_df.height}"
    )
    print(f"OK: {out_df.height} linhas == {expected} interacoes de teste")

    print(f"Salvando {OUTPUT_PATH}...")
    out_df.write_parquet(OUTPUT_PATH)

    print("Concluido!")


if __name__ == "__main__":
    main()
