"""Extrai o ground truth do split de teste (Card 9).

Le o dataset de interacoes ja processado pelo feature engineering
(feature_engineering.py), que ja contem `split`, filtra apenas as
linhas de teste e mantem somente uid, app_package e timestamp
(formated_date, renomeada para `timestamp`) -- as mesmas chaves
(uid, timestamp) usadas nos arquivos de predicao (predict_random.py,
predict_pop.py) e esperadas pelo modulo de metricas (metrics/, Card 10),
para comparar o app_package realmente consumido contra os rankings
preditos.
"""

import polars as pl

INPUT_PATH = "data/processed/interactions_fe.parquet"
OUTPUT_PATH = "data/predictions/test_ground_truth.parquet"


def main():
    print(f"Lendo {INPUT_PATH}...")
    df = pl.read_parquet(INPUT_PATH)

    print("Filtrando split de teste...")
    test_df = df.filter(pl.col("split") == "test").select(
        pl.col("uid"),
        pl.col("app_package"),
        pl.col("formated_date").alias("timestamp"),
    )

    print(f"Salvando {OUTPUT_PATH}...")
    test_df.write_parquet(OUTPUT_PATH)

    print(f"Concluido! {test_df.height} linhas de teste salvas.")


if __name__ == "__main__":
    main()
