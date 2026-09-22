"""Predicao final unificada dos modelos de popularidade (recentPop /
decayPop / recentDecayPop / dechypPop-sem-janela).

Um unico script para o que hoje esta espalhado em predict_pop.py,
predict_decay_pop.py, predict_recent_decay_pop.py e predict_dechyp_pop.py
-- esses 4 scripts continuam intactos no repositorio (uso manual, fora
deste pipeline), mas este arquivo NAO os importa nem reaproveita seu
codigo: e uma implementacao nova, sobre pipeline/ (ver
plano_pipeline_popularidade.md).

Roda no split de TESTE, com os hiperparametros ja escolhidos (por
optimize_popularity.py ou manualmente). Usa exatamente a mesma
`PopularityPipeline.score()` usada pela otimizacao (via `.predict()`,
split="test") -- garante que a formula usada para prever e a mesma usada
para validar, eliminando o risco de otimizacao e predicao divergirem
silenciosamente que existia nos scripts antigos (cada par
optimize_*/predict_*.py reimplementava a formula duas vezes).

Depois de gerar as predicoes, o fluxo de avaliacao final e o mesmo ja
existente no projeto: extract_test_ground_truth.py + evaluate_predictions.py
(ou, para qualquer uma das metricas novas do registry,
pipeline.metric_registry.evaluate diretamente).

Exemplos:
    python predict_popularity.py --decay none --window 90
    python predict_popularity.py --decay exponential --window none --lambda 0.03
    python predict_popularity.py --decay exponential --window 90 --lambda 0.03
    python predict_popularity.py --decay hyperbolic --window none --lambda 0.05
"""

import argparse
import os

import polars as pl

from pipeline import DECAY_REGISTRY, PopularityConfig, PopularityPipeline

N_RECS = 50

RAW_INTERACTIONS_PATH = "data/processed/interactions.parquet"
INTERACTIONS_PATH = "data/processed/interactions_fe.parquet"
OUTPUT_DIR = "data/predictions/popularity"


def parse_window_arg(value: str) -> int | None:
    if value.lower() in ("none", "all"):
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--decay", choices=list(DECAY_REGISTRY), required=True)
    parser.add_argument(
        "--window",
        type=parse_window_arg,
        default=None,
        help="Janela em dias, ou 'none' para nao aplicar corte de recencia. Default: none.",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=None,
        help="Fator de decaimento (obrigatorio para --decay exponential|hyperbolic).",
    )
    parser.add_argument("--n-recs", type=int, default=N_RECS)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PopularityConfig:
    strategy = DECAY_REGISTRY[args.decay]

    decay_params = {}
    if "lambda_" in strategy.param_space:
        if args.lambda_ is None:
            raise SystemExit(f"--lambda e obrigatorio para --decay {args.decay}")
        decay_params["lambda_"] = args.lambda_
    elif args.lambda_ is not None:
        raise SystemExit(f"--decay {args.decay} nao tem hiperparametro 'lambda_' -- remova --lambda")

    if args.window is not None and not strategy.supports_window:
        raise SystemExit(
            f"--decay {args.decay} ainda nao suporta corte de janela (recencia). "
            "Use --window none (ou omita --window)."
        )

    return PopularityConfig(decay=args.decay, decay_params=decay_params, window=args.window, n_recs=args.n_recs)


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)

    print(f"Lendo {RAW_INTERACTIONS_PATH}...")
    raw_df = pl.read_parquet(RAW_INTERACTIONS_PATH)

    print(f"Lendo {INTERACTIONS_PATH}...")
    df = pl.read_parquet(INTERACTIONS_PATH)

    pipeline = PopularityPipeline(n_recs=args.n_recs)
    print(
        f"Gerando predicoes de teste (decay={args.decay}, window={config.window}, "
        f"params={config.decay_params})..."
    )
    predictions = pipeline.predict(config, df, raw_df)
    print(f"OK: {predictions.height} linhas de teste processadas")

    window_label = config.window if config.window is not None else "none"
    lambda_label = f"_{args.lambda_}" if args.lambda_ is not None else ""
    output_path = f"{OUTPUT_DIR}/{args.decay}_{window_label}{lambda_label}.parquet"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Salvando {output_path}...")
    predictions.write_parquet(output_path)

    print("Concluido!")
