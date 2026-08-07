# import polars as pl


# pl.scan_csv('.datadataset1.csv').sink_parquet('dataset1.parquet')
# pl.scan_csv('dataset2.csv').sink_parquet('dataset2.parquet')

# print("Conversão finalizada com sucesso!")

import pandas as pd
import re
import polars as pl

def remove_illegal_char(column: pd.Series):
    # ASCII characters
    column = column.str.encode('ascii', 'ignore').str.decode("utf-8")

    # Remove illegal XML characters that cause issues with to_excel
    # Define a regex for illegal XML 1.0 characters (excluding valid control characters like tab, newline, carriage return)
    ILLEGAL_CHARACTERS_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
    column = column.apply(lambda x: ILLEGAL_CHARACTERS_RE.sub('', x) if isinstance(x, str) else x)
    return column

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).replace("\n", " ").lower().strip()

print("Preparando metadados dos apps...")

metadata_cols = [
    'app_package',
    'app_category',
    'description'
]

metadata = pd.read_csv('data/input/metadata.csv', usecols=metadata_cols)

metadata["description"] = remove_illegal_char(metadata["description"])
metadata["description"] = metadata["description"].apply(clean_text)

print("Salvando metadados dos apps em csv...")
metadata.to_csv('data/processed/metadata.csv', index=False)

print("Salvando metadados dos apps em parquet...")
pl.scan_csv('data/processed/metadata.csv').sink_parquet('data/processed/metadata.parquet')

##############################

print("Preparando interações...")

interactions_cols = [
    'uid',
    'app_package',
    'rating',
    'formated_date'
]

interactions = pd.read_csv('data/input/interactions.csv', usecols=interactions_cols)
interactions['formated_date'] = pd.to_datetime(interactions['formated_date'])

print("Salvando interações em csv...")
interactions.to_csv('data/processed/interactions.csv', index=False)

print("Salvando interações em parquet...")
pl.scan_csv('data/processed/interactions.csv').sink_parquet('data/processed/interactions.parquet')

print("Concluído!")