import polars as pl

from utils import add_running_mean, add_centered_rating, add_data_split

INPUT_PATH = "data/processed/interactions.parquet"
OUTPUT_PATH = "data/processed/interactions_fe.parquet"


def main():
    print(f"Lendo {INPUT_PATH}...")
    df = pl.read_parquet(INPUT_PATH)

    print("Removendo linhas com rating nulo...")
    df = df.filter(pl.col("rating").is_not_null())

    print("Card 2: calculando running_mean...")
    df = add_running_mean(df)

    print("Card 3: calculando centered_rating...")
    df = add_centered_rating(df)

    print("Card 5: calculando split treino/val/teste...")
    df = add_data_split(df)

    print(f"Salvando {OUTPUT_PATH}...")
    df.write_parquet(OUTPUT_PATH)

    print("Concluido!")


if __name__ == "__main__":
    main()
