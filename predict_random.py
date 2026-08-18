"""Pipeline de predicao do modelo random (Card 9).

Le o dataset de interacoes ja processado pelo feature engineering
(feature_engineering.py), que ja contem `interaction_rank` e `split`.
Para cada interacao de teste, gera um ranking aleatorio de N_RECS apps
dentre os que o usuario ainda nao havia consumido -- excluindo todo o
historico anterior a ela (treino, validacao e teste com timestamp
menor).

A amostragem usa models.random_model.RandomModel (Card 7).

O resultado (~3.2M linhas x 250 colunas) e grande demais para caber
inteiro em memoria de uma vez, entao e escrito em lotes de BATCH_SIZE
linhas de teste diretamente no parquet (via pyarrow.ParquetWriter), sem
nunca acumular o dataframe completo em RAM.

ATENCAO: este script processa o dataset inteiro (~19M interacoes) e
NAO deve ser executado neste ambiente -- destina-se a rodar em outra
maquina.
"""

import os

import polars as pl
import pyarrow.parquet as pq

from models import RandomModel

INPUT_PATH = "data/processed/interactions_fe.parquet"
OUTPUT_PATH = "data/predictions/random.parquet"
N_RECS = 250
BATCH_SIZE = 200_000  # linhas de teste por lote escrito no parquet, controla o pico de memoria

REC_COLS = [f"rec{j:03d}" for j in range(N_RECS)]
SCHEMA = ["uid", "timestamp", *REC_COLS]
DTYPES = {col: pl.Utf8 for col in SCHEMA}


def main():
    print(f"Lendo {INPUT_PATH}...")
    df = pl.read_parquet(INPUT_PATH).sort(["uid", "interaction_rank"])

    catalog = sorted(df["app_package"].unique().to_list())
    model = RandomModel()

    uids = df["uid"].to_list()
    apps = df["app_package"].to_list()
    timestamps = df["formated_date"].to_list()
    splits = df["split"].to_list()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Gerando predicoes e salvando em lotes...")
    consumed: set = set()
    current_uid = None
    batch: list = []
    writer = None
    total_written = 0

    for uid, app, timestamp, split in zip(uids, apps, timestamps, splits):
        if uid != current_uid:
            consumed = set()
            current_uid = uid

        if split == "test":
            valid_apps = [a for a in catalog if a not in consumed]
            preds = model.predict(valid_apps, N_RECS)
            preds += [None] * (N_RECS - len(preds))
            batch.append((uid, timestamp, *preds))

            if len(batch) >= BATCH_SIZE:
                writer = _flush(batch, writer)
                total_written += len(batch)
                print(f"  {total_written} linhas de teste processadas...")
                batch = []

        consumed.add(app)

    if batch:
        writer = _flush(batch, writer)
        total_written += len(batch)

    if writer is not None:
        writer.close()

    print("Validando...")
    expected = df.filter(pl.col("split") == "test").height
    assert total_written == expected, (
        f"esperado {expected} linhas de teste, obtido {total_written}"
    )
    print(f"OK: {total_written} linhas == {expected} interacoes de teste")

    print("Concluido!")


def _flush(batch: list, writer) -> pq.ParquetWriter:
    """Converte um lote de linhas para colunar e escreve no parquet, sem
    acumular nada alem desse lote em memoria."""
    columns = dict(zip(SCHEMA, zip(*batch)))
    table = pl.DataFrame(columns, schema=DTYPES).to_arrow()

    if writer is None:
        writer = pq.ParquetWriter(OUTPUT_PATH, table.schema)
    writer.write_table(table)
    return writer


if __name__ == "__main__":
    main()
