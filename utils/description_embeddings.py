import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def generate_description_embeddings(
    df: pd.DataFrame,
    model_name: str = MODEL_NAME,
    id_col: str = "app_package",
    text_col: str = "description",
) -> pd.DataFrame:
    """Gera embeddings textuais das descricoes dos apps.

    Recebe um DataFrame com as colunas `id_col` e `text_col`, codifica
    `text_col` com um SentenceTransformer e retorna um DataFrame indexado
    por `id_col`, com uma coluna `emb_i` para cada dimensao do embedding.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df[text_col].tolist(), show_progress_bar=True)

    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    embeddings_df = pd.DataFrame(embeddings, columns=emb_cols, index=df[id_col])

    return embeddings_df
