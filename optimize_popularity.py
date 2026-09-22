"""Otimizacao unificada dos modelos de popularidade (recentPop / decayPop /
recentDecayPop / dechypPop-sem-janela) via Optuna.

Um unico script para as 4 combinacoes hoje espalhadas em optimize_pop.py,
optimize_decay_pop.py, optimize_recent_decay_pop.py e
optimize_dechyp_pop.py -- esses 4 scripts continuam intactos no
repositorio (uso manual, fora deste pipeline), mas este arquivo NAO os
importa nem reaproveita seu codigo: e uma implementacao nova, sobre
pipeline/ (ver plano_pipeline_popularidade.md).

Cada modelo e uma composicao de 2 decisoes independentes, resolvidas por
`pipeline.PopularityPipeline` (mesma logica usada depois por
predict_popularity.py, para que otimizacao e predicao final nunca
divirjam):

    decay=none                        -> recentPop
    decay=exponential, window=none    -> decayPop
    decay=exponential, window=W       -> recentDecayPop
    decay=hyperbolic,  window=none    -> dechypPop (janela ainda nao
                                          suportada para hyperbolic -- ver
                                          pipeline/decay_strategies.py)

"Rodar tudo" ou "rodar por partes" e so uma questao de quais dimensoes
ficam livres (--decay auto, --window auto) ou fixas
(--decay none, --window 90) -- mesmo objective do Optuna nos dois casos.

Metrica default: ndcg@20 (mesmo default usado nos 4 scripts antigos).
Outras opcoes: hr@1/5/10/15/20, ndcg@5/10/15, mrr (ver
pipeline/metric_registry.py) -- escolhida via --metric, sem editar codigo.

Split de teste NUNCA e usado aqui -- so validacao (val), mesma garantia
dos scripts antigos.

Exemplos:
    python optimize_popularity.py --decay none                       # == optimize_pop.py
    python optimize_popularity.py --decay exponential --window none  # == optimize_decay_pop.py
    python optimize_popularity.py --decay exponential                # == optimize_recent_decay_pop.py
    python optimize_popularity.py --decay hyperbolic --window none   # == optimize_dechyp_pop.py
    python optimize_popularity.py --decay auto                       # busca conjunta (decay + window)
    python optimize_popularity.py --decay auto --metric hr@10        # otimiza para outra nota
"""

import argparse
import os
import pickle
import time

import optuna
import polars as pl

from pipeline import DECAY_REGISTRY, PopularityConfig, PopularityPipeline
from pipeline.metric_registry import DEFAULT_METRIC, METRIC_NAMES, evaluate

SEED = 42
N_TRIALS = 100
N_RECS = 50
WINDOW_MAX_DAYS = 365

# Seeds de trials iniciais (via study.enqueue_trial), so para as dimensoes
# que estiverem livres -- cobrem a faixa de busca antes de partir para o
# TPE, mesmo espirito dos scripts antigos (INITIAL_WINDOWS/INITIAL_LAMBDAS
# em optimize_pop.py e companhia), sem tentar reproduzi-los ponto a ponto.
INITIAL_WINDOWS = [0, 1, 30, 90, 180, 365]
INITIAL_LAMBDAS = [0.001, 0.01, 0.1, 1.0]

RAW_INTERACTIONS_PATH = "data/processed/interactions.parquet"
INTERACTIONS_PATH = "data/processed/interactions_fe.parquet"
RESULTS_DIR = "data/optuna/popularity"


def parse_window_arg(value: str) -> int | None | str:
    if value.lower() == "auto":
        return "auto"
    if value.lower() in ("none", "all"):
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--decay",
        choices=[*DECAY_REGISTRY, "auto"],
        default="auto",
        help="Familia de decaimento fixa, ou 'auto' para o Optuna escolher. Default: auto.",
    )
    parser.add_argument(
        "--window",
        type=parse_window_arg,
        default="auto",
        help=(
            "Janela em dias fixa (ex.: 90), 'none' (sem corte de recencia) ou "
            "'auto' (Optuna escolhe, 0..--window-max-days). Default: auto."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_NAMES,
        default=DEFAULT_METRIC,
        help=f"Metrica otimizada. Default: {DEFAULT_METRIC}.",
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--window-max-days", type=int, default=WINDOW_MAX_DAYS)
    parser.add_argument("--n-recs", type=int, default=N_RECS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def _initial_trials(args: argparse.Namespace) -> list[dict]:
    """Enfileira alguns pontos conhecidos do espaco de busca antes do TPE
    assumir -- so nas dimensoes que estao livres (--decay/--window auto).
    Se ambos estiverem fixos, nao ha nada para enfileirar (so os
    hiperparametros continuos, se houver, ficam com o TPE puro)."""
    decays = list(DECAY_REGISTRY) if args.decay == "auto" else [args.decay]
    trials = []

    for decay_name in decays:
        strategy = DECAY_REGISTRY[decay_name]
        window_free = args.window == "auto" and strategy.supports_window

        lambda_values = INITIAL_LAMBDAS if "lambda_" in strategy.param_space else [None]
        # clipado a --window-max-days: um seed fora do range declarado em
        # trial.suggest_int (objective, abaixo) faria o Optuna rejeitar o
        # enqueue_trial quando --window-max-days for customizado (< 365)
        window_values = [w for w in INITIAL_WINDOWS if w <= args.window_max_days] if window_free else [None]

        for lambda_ in lambda_values:
            for window_days in window_values:
                params: dict = {}
                if args.decay == "auto":
                    params["decay"] = decay_name
                if lambda_ is not None:
                    params[f"{decay_name}__lambda_"] = lambda_
                if window_free:
                    params["window_days"] = window_days
                if params:
                    trials.append(params)

    return trials


def build_objective(pipeline: PopularityPipeline, ctx, ground_truth: pl.DataFrame, args: argparse.Namespace):
    decay_free = args.decay == "auto"
    window_free = args.window == "auto"

    def objective(trial: optuna.Trial) -> float:
        decay_name = trial.suggest_categorical("decay", list(DECAY_REGISTRY)) if decay_free else args.decay
        strategy = DECAY_REGISTRY[decay_name]

        decay_params = {}
        for param_name, (kind, low, high) in strategy.param_space.items():
            # nome do parametro prefixado por decay_name: evita que o TPE
            # trate hiperparametros de familias diferentes (ex.: lambda_ de
            # exponential vs. de hyperbolic) como a mesma dimensao quando
            # --decay auto esta ativo (padrao recomendado do Optuna para
            # espacos de busca condicionais).
            key = f"{decay_name}__{param_name}"
            if kind == "float":
                decay_params[param_name] = trial.suggest_float(key, low, high)
            elif kind == "int":
                decay_params[param_name] = trial.suggest_int(key, int(low), int(high))
            else:
                raise ValueError(f"Tipo de hiperparametro '{kind}' nao suportado")

        if window_free:
            if strategy.supports_window:
                window_days = trial.suggest_int("window_days", 0, args.window_max_days)
                window = None if not window_days else window_days
            else:
                window = None
        else:
            window = args.window

        if window is not None and not strategy.supports_window:
            # combinacao invalida sorteada pelo TPE em modo --decay auto
            # (estrategia sem suporte a janela, mas window!=None fixo por
            # --window): poda o trial em vez de deixar score() estourar.
            raise optuna.TrialPruned(
                f"'{decay_name}' ainda nao suporta corte de janela (window={window})"
            )

        config = PopularityConfig(
            decay=decay_name, decay_params=decay_params, window=window, n_recs=args.n_recs
        )

        start = time.perf_counter()
        predictions = pipeline.score(config, ctx)
        value = evaluate(predictions, ground_truth, metric=args.metric, n_recs=args.n_recs)
        elapsed = time.perf_counter() - start

        trial.set_user_attr("decay", decay_name)
        trial.set_user_attr("window", window)

        print(
            f"[trial {trial.number:03d}] decay={decay_name} window={window} "
            f"params={decay_params} {args.metric}={value:.6f} tempo={elapsed:.1f}s"
        )
        return value

    return objective


if __name__ == "__main__":
    args = parse_args()

    print(f"Lendo {RAW_INTERACTIONS_PATH}...")
    raw_df = pl.read_parquet(RAW_INTERACTIONS_PATH)

    print(f"Lendo {INTERACTIONS_PATH}...")
    df = pl.read_parquet(INTERACTIONS_PATH)

    pipeline = PopularityPipeline(n_recs=args.n_recs)
    ctx = pipeline.prepare(df, raw_df, split="val")
    del df, raw_df  # libera as ~19M linhas; so os arrays de val em `ctx` sao necessarios daqui pra frente
    ground_truth = ctx.ground_truth

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))

    initial_trials = _initial_trials(args)
    for params in initial_trials:
        study.enqueue_trial(params)

    objective = build_objective(pipeline, ctx, ground_truth, args)
    print(
        f"Rodando {args.n_trials} trials ({len(initial_trials)} fixos + TPE) -- "
        f"decay={args.decay} window={args.window} metrica={args.metric}..."
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    print(f"Melhor decay: {best.user_attrs.get('decay')}")
    print(f"Melhor window: {best.user_attrs.get('window')}")
    print(f"Melhores hiperparametros: {best.params}")
    print(f"Melhor {args.metric}: {study.best_value:.6f}")

    tag = args.decay  # "auto" ou o nome da familia fixada
    results_dir = f"{RESULTS_DIR}/{tag}/{args.seed}"
    os.makedirs(results_dir, exist_ok=True)

    results_path = f"{results_dir}/optuna_results.csv"
    study.trials_dataframe().to_csv(results_path, index=False)
    print(f"Historico de trials salvo em {results_path}")

    study_path = f"{results_dir}/optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"Study salvo em {study_path}")
