"""Pipeline de predicao do modelo de popularidade (Card 8).

Le o dataset de interacoes ja processado pelo feature engineering
(feature_engineering.py), que ja contem `interaction_rank` e `split`,
e a popularity_matrix (Card 4). Para cada interacao de teste, gera um
ranking por popularidade de N_RECS apps dentre os que o usuario ainda
nao havia consumido -- excluindo todo o historico anterior a ela
(treino, validacao e teste com timestamp menor).

A pontuacao usa models.pop_model.POPModel (Card 8), com a data da
propria interacao como reference_date.

O resultado (~3.2M linhas x 250 colunas) e grande demais para caber
inteiro em memoria de uma vez, entao e escrito em lotes de BATCH_SIZE
linhas de teste diretamente no parquet (via pyarrow.ParquetWriter), sem
nunca acumular o dataframe completo em RAM.

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

import polars as pl
import pyarrow.parquet as pq

from models import POPModel

WINDOW = 90  # default: tamanho da janela em dias (None = POP-All, 90/180/365 = POP-3/6/12); sobrescrito por --window

INTERACTIONS_PATH = "data/processed/interactions_fe.parquet"
POPULARITY_MATRIX_PATH = "data/processed/popularity_matrix.parquet"
N_RECS = 250
BATCH_SIZE = 200_000  # linhas de teste por lote escrito no parquet, controla o pico de memoria

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

    print(f"Lendo {INTERACTIONS_PATH}...")
    df = pl.read_parquet(INTERACTIONS_PATH).sort(["uid", "interaction_rank"])

    print(f"Lendo {POPULARITY_MATRIX_PATH}...")
    popularity_matrix = pl.read_parquet(POPULARITY_MATRIX_PATH)

    catalog = sorted(df["app_package"].unique().to_list())
    model = POPModel(window=window)

    uids = df["uid"].to_list()
    apps = df["app_package"].to_list()
    timestamps = df["formated_date"].to_list()
    splits = df["split"].to_list()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
            preds = model.predict(valid_apps, N_RECS, popularity_matrix, timestamp)
            preds += [None] * (N_RECS - len(preds))
            batch.append((uid, timestamp, *preds))

            if len(batch) >= BATCH_SIZE:
                writer = _flush(batch, writer, output_path)
                total_written += len(batch)
                print(f"  {total_written} linhas de teste processadas...")
                batch = []

        consumed.add(app)

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
