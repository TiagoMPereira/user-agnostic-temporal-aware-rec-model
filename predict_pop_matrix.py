"""Predicao com o recomendador por popularidade em janela deslizante
(pacote ``pop_matrix``).

Suporta duas configuracoes (``--stage``), ambas via
``utils.PopMatrixContext``:

- ``val`` (default): matriz de interacoes construida SOMENTE com o split
  de treino; predicao roda sobre o split de validacao, excluindo (black
  list) os apps que o proprio usuario ja consumiu no treino. Mesma
  configuracao usada por ``optimize_pop_matrix.py`` -- para rodar o
  modelo do jeito exato em que o ``window`` foi otimizado.
- ``test``: matriz construida com treino+validacao; predicao roda sobre o
  split de teste, excluindo os apps ja consumidos em treino+validacao.
  Usar so depois de escolhido o melhor ``window`` via Optuna
  (``optimize_pop_matrix.py``), para a avaliacao final do modelo.

Em ambos os casos, antes de salvar confere de forma vetorizada que a
black list foi respeitada (``utils.verify_blacklist_respected``) e
calcula o NDCG@20 da recomendacao (``utils.evaluate_ndcg20``, mesma
formula usada pelo Optuna em ``optimize_pop_matrix.py``).

``interactions_fe.parquet`` ja vem no formato dia-a-dia
(``formated_date`` string "YYYY-MM-DD", mais a coluna ``split``) --
schema diferente do dataset bruto que ``pop_matrix.load_interactions``
espera (secao 2 das instrucoes de pop_matrix: uid/app_package/review/
timestamp epoch). Por isso a matriz e construida direto com
``build_interaction_data`` a partir de um ``LazyFrame`` so com
``app_package``/``date``, sem passar por ``load_interactions``.

Salva, em ``data/predictions/popularity_matrix/<stage>/``, o dataset de
recomendacoes (``w<window>_n<n_recs>.parquet``) e o NDCG@20
(``w<window>_n<n_recs>_ndcg20.json``) -- o ``window`` fica no nome dos
dois arquivos.

Exemplos:
    python predict_pop_matrix.py --window 30
    python predict_pop_matrix.py --window 90 --n-recs 20 --seed 42
    python predict_pop_matrix.py --window 30 --stage test
"""

import argparse
import json
import os

from pop_matrix import recommend_batch
from utils import (
    evaluate_ndcg20,
    prepare_train_val_context,
    prepare_trainval_test_context,
    rankings_to_predictions,
    verify_blacklist_respected,
)

INTERACTIONS_PATH = "data/processed/interactions_fe.parquet"
OUTPUT_DIR = "data/predictions/popularity_matrix"
N_RECS = 50

# stage -> como montar o PopMatrixContext (utils.pop_matrix_context).
STAGE_CONTEXT = {
    "val": prepare_train_val_context,
    "test": prepare_trainval_test_context,
}


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
    parser.add_argument(
        "--stage",
        choices=sorted(STAGE_CONTEXT),
        default="val",
        help=(
            "'val' (default): matriz de treino, predicao/avaliacao no split de "
            "validacao -- mesma configuracao usada pelo Optuna (optimize_pop_matrix.py). "
            "'test': matriz de treino+validacao, predicao/avaliacao no split de teste -- "
            "rodar so com o window ja escolhido pelo Optuna, para a avaliacao final."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Lendo {INTERACTIONS_PATH} (stage={args.stage})...")
    ctx = STAGE_CONTEXT[args.stage](INTERACTIONS_PATH)

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
    predictions = rankings_to_predictions(rankings, ctx.data.apps, ctx.eval_df)

    print("Verificando que a black list foi respeitada (apps ja consumidos na matriz)...")
    rec_cols = [f"rec{j:03d}" for j in range(args.n_recs)]
    verify_blacklist_respected(predictions, ctx.eval_df, rec_cols)

    print("Calculando NDCG@20...")
    ndcg20 = evaluate_ndcg20(predictions, ctx.eval_df, n_recs=args.n_recs)
    print(f"NDCG@20: {ndcg20:.6f}")

    output_dir = f"{OUTPUT_DIR}/{args.stage}"
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"w{args.window}_n{args.n_recs}"

    predictions_path = f"{output_dir}/{base_name}.parquet"
    print(f"Salvando {predictions_path}...")
    predictions.write_parquet(predictions_path)

    score_path = f"{output_dir}/{base_name}_ndcg20.json"
    print(f"Salvando {score_path}...")
    with open(score_path, "w") as f:
        json.dump(
            {
                "stage": args.stage,
                "window": args.window,
                "n_recs": args.n_recs,
                "seed": args.seed,
                "ndcg20": ndcg20,
            },
            f,
            indent=2,
        )

    print(f"Concluido! {predictions.height} linhas de predicao salvas. NDCG@20={ndcg20:.6f}")


if __name__ == "__main__":
    main()
