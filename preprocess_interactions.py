import polars as pl

from utils import build_popularity_matrix

INPUT_PATH = "data/processed/interactions.parquet"
OUTPUT_PATH = "data/processed/popularity_matrix.parquet"


def main():
    print(f"Lendo {INPUT_PATH}...")
    df = pl.read_parquet(INPUT_PATH)

    print("Construindo matriz de popularidade cumulativa...")
    matrix = build_popularity_matrix(df)

    print(f"Salvando {OUTPUT_PATH}...")
    matrix.write_parquet(OUTPUT_PATH)

    print("Concluido!")


if __name__ == "__main__":
    main()
