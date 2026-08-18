"""Pipeline de predicao do modelo de popularidade (Card 8).

Le o dataset de interacoes ja processado pelo feature engineering
(feature_engineering.py), que ja contem `interaction_rank` e `split`,
e a popularity_matrix (Card 4). Para cada interacao de teste, gera um
ranking por popularidade de N_RECS apps dentre os que o usuario ainda
nao havia consumido -- excluindo todo o historico anterior a ela
(treino, validacao e teste com timestamp menor).

A pontuacao usa models.pop_model.POPModel (Card 8), com a data da
propria interacao como reference_date.

ATENCAO: este script processa o dataset inteiro (~19M interacoes) e
NAO deve ser executado neste ambiente -- destina-se a rodar em outra
maquina.
"""

import polars as pl

from models import POPModel

WINDOW = 90  # tamanho da janela em dias (None = POP-All, 90/180/365 = POP-3/6/12)

INTERACTIONS_PATH = "data/processed/interactions_fe.parquet"
POPULARITY_MATRIX_PATH = "data/processed/popularity_matrix.parquet"
OUTPUT_PATH = f"data/predictions/pop_{WINDOW}.parquet"
N_RECS = 250


def main():
    print(f"Lendo {INTERACTIONS_PATH}...")
    df = pl.read_parquet(INTERACTIONS_PATH).sort(["uid", "interaction_rank"])

    print(f"Lendo {POPULARITY_MATRIX_PATH}...")
    popularity_matrix = pl.read_parquet(POPULARITY_MATRIX_PATH)

    catalog = sorted(df["app_package"].unique().to_list())
    model = POPModel(window=WINDOW)

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
            preds = model.predict(valid_apps, N_RECS, popularity_matrix, timestamp)
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
