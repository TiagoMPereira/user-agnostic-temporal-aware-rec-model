import pandas as pd

from utils import generate_description_embeddings

INPUT_PATH = "data/processed/metadata.parquet"
OUTPUT_PATH = "data/processed/descriptions.parquet"


def main():
    print(f"Lendo {INPUT_PATH}...")
    metadata = pd.read_parquet(INPUT_PATH, columns=["app_package", "description"])

    print("Gerando embeddings das descricoes...")
    embeddings = generate_description_embeddings(metadata)

    print(f"Salvando {OUTPUT_PATH}...")
    embeddings.to_parquet(OUTPUT_PATH)

    print("Concluido!")


if __name__ == "__main__":
    main()
