import pandas as pd

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

    Import de `sentence_transformers` feito aqui dentro (nao no topo do
    modulo): e uma dependencia pesada e opcional -- so quem de fato chama
    esta funcao precisa dela instalada. No topo do modulo, ela e importada
    a toda vez que `utils` e importado (mesmo por quem so precisa de
    `pivot_daily_counts`/`build_popularity_matrix`, por exemplo).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(df[text_col].tolist(), show_progress_bar=True)

    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    embeddings_df = pd.DataFrame(embeddings, columns=emb_cols, index=df[id_col])

    return embeddings_df
