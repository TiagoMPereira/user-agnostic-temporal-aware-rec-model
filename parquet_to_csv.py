import pandas as pd

INPUT_PATH = "data/processed/popularity_matrix.parquet"
OUTPUT_PATH = "data/processed/popularity_matrix.csv"

if __name__ == "__main__":

    obj = pd.read_parquet(INPUT_PATH)

    print(obj.size)
    print(obj.head())
    print(obj.tail())

    obj.to_csv(OUTPUT_PATH)
