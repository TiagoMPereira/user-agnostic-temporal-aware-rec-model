"""Predicao do split de validacao com o recomendador por popularidade em
janela deslizante (pacote ``pop_matrix``).

Le o dataset ja processado pelo feature engineering
(``feature_engineering.py``), que contem a coluna ``split``
(train/val/test, leave-one-out por usuario -- ``utils.add_data_split``).
O split de teste e removido logo no inicio e nunca e tocado. A matriz de
interacoes (``M``/``P``) e construida SOMENTE com o split de treino; a
predicao roda sobre o split de validacao, uma recomendacao por interacao
de val, excluindo (black list) os apps que o proprio usuario ja consumiu
no treino (``utils.prepare_train_val_context`` -- mesma logica usada por
``optimize_pop_matrix.py``, para as duas nunca divergirem). Antes de
salvar, confere de forma vetorizada que a black list foi mesmo respeitada
(``utils.verify_blacklist_respected``).

``interactions_fe.parquet`` ja vem no formato dia-a-dia
(``formated_date`` string "YYYY-MM-DD", mais a coluna ``split``) --
schema diferente do dataset bruto que ``pop_matrix.load_interactions``
espera (secao 2 das instrucoes de pop_matrix: uid/app_package/review/
timestamp epoch). Por isso a matriz e construida direto com
``build_interaction_data`` a partir de um ``LazyFrame`` so com
``app_package``/``date``, sem passar por ``load_interactions``.

Exemplos:
    python predict_pop_matrix.py --window 30
    python predict_pop_matrix.py --window 90 --n-recs 20 --seed 42
"""

import argparse
import os

from pop_matrix import recommend_batch
from utils import prepare_train_val_context, rankings_to_predictions, verify_blacklist_respected

INTERACTIONS_PATH = "data/processed/interactions_fe.parquet"
OUTPUT_DIR = "data/predictions/popularity_matrix"
N_RECS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--window",
        type=int,
        required=True,
        help="Tamanho da janela deslizante (dias) usada na contagem de popularidade.",
    )
    parser.add_argument("--n-recs", type=int, default=N_RECS, help=f"Numero de recomendacoes por usuario. Default: {N_RECS}.")
    parser.add_argument("--seed", type=int, default=None, help="Semente do desempate aleatorio (default: nao reprodutivel).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Lendo {INTERACTIONS_PATH}...")
    ctx = prepare_train_val_context(INTERACTIONS_PATH)

    print(f"Gerando recomendacoes (window={args.window}, n_recs={args.n_recs}, seed={args.seed})...")
    rankings = recommend_batch(
        ctx.data,
        ctx.prefix,
        n=args.n_recs,
        ts=ctx.ts,
        w=args.window,
        bl_indptr=ctx.bl_indptr,
        bl_indices=ctx.bl_indices,
        seed=args.seed,
    )

    print("Convertendo indices de app para app_package...")
    predictions = rankings_to_predictions(rankings, ctx.data.apps, ctx.val_df)

    print("Verificando que a black list foi respeitada (apps ja consumidos no treino)...")
    rec_cols = [f"rec{j:03d}" for j in range(args.n_recs)]
    verify_blacklist_respected(predictions, ctx.val_df, rec_cols)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f"{OUTPUT_DIR}/val_w{args.window}_n{args.n_recs}.parquet"
    print(f"Salvando {output_path}...")
    predictions.write_parquet(output_path)

    print(f"Concluido! {predictions.height} linhas de predicao salvas.")


if __name__ == "__main__":
    main()
