"""Otimizacao do tamanho da janela (``window``) do recomendador por
popularidade em janela deslizante (pacote ``pop_matrix``), via Optuna.

Mesma base e split de ``predict_pop_matrix.py --stage val`` (default --
``utils.prepare_train_val_context``: matriz de interacoes construida
SOMENTE com o split de treino, split de teste removido logo no inicio e
nunca usado, black list de cada consulta de validacao = apps que o
proprio usuario ja consumiu no treino). A busca roda sobre o split de
VALIDACAO, maximizando NDCG@20 -- reaproveita ``utils.evaluate_ndcg20``
(que por sua vez reaproveita ``metrics/``, a mesma formula ja usada por
``evaluate_predictions.py``) em vez de recalcular a metrica aqui.

Depois de escolhido o melhor ``window`` aqui, a avaliacao final roda em
``predict_pop_matrix.py --stage test --window <melhor>`` (matriz
treino+validacao, split de teste) -- este script nunca toca o teste.

Espaco de busca: ``window_days`` inteiro em ``[0, --window-max-days]``
(365 por padrao). ``window_days=0`` e o sentinela para "sem corte de
janela -- considera todas as datas de treino disponiveis": nesse caso
``w`` e fixado em ``data.n_days`` (o numero de dias do calendario de
treino), grande o bastante para que a janela de qualquer ``t`` nunca
seja truncada antes do dia 0 (mesma convencao de window=0/None em
``optimize_popularity.py``/``predict_popularity.py``).

``data``/``P``/black lists/``ts`` nao dependem de ``window`` -- sao
construidos uma unica vez (fora do laco do Optuna) e reaproveitados em
todos os trials; so ``recommend_batch`` (que depende de ``w``) e a
metrica sao recalculados a cada trial. A black list e conferida
(``utils.verify_blacklist_respected``) uma vez, sobre o melhor trial --
a mesma ``bl_indptr``/``bl_indices`` e usada, inalterada, em todos os
trials, entao valer para um vale para todos.

Salva o historico COMPLETO de trials (CSV) e o ``study`` (pickle), assim
como ``optimize_popularity.py``.

Exemplos:
    python optimize_pop_matrix.py
    python optimize_pop_matrix.py --n-trials 50 --seed 7
    python optimize_pop_matrix.py --window-max-days 180
"""

import argparse
import os
import pickle
import random
import time

import optuna

from pop_matrix import InteractionData, recommend_batch
from utils import PopMatrixContext, evaluate_ndcg20, prepare_train_val_context, rankings_to_predictions, verify_blacklist_respected

SEED = 42
N_TRIALS = 100
N_RECS = 20  # NDCG@20: as posicoes alem de 20 nao influenciam a metrica
WINDOW_MAX_DAYS = 365
METRIC = "ndcg@20"

# Alguns pontos conhecidos do espaco de busca enfileirados antes do TPE
# assumir (mesmo espirito de INITIAL_WINDOWS em optimize_popularity.py);
# 0 e o sentinela "todas as datas disponiveis".
INITIAL_WINDOWS = [0, 1, 7, 30, 90, 180, 365]

INTERACTIONS_PATH = "data/processed/interactions_fe.parquet"
RESULTS_DIR = "data/optuna/popularity_matrix"
MAX_RESAMPLE_ATTEMPTS = 100


class UniqueIntTPESampler(optuna.samplers.TPESampler):
    """Evita repetir valores de ``window_days`` nos trials amostrados pelo TPE."""

    def sample_independent(self, study, trial, param_name, param_distribution):
        value = super().sample_independent(study, trial, param_name, param_distribution)
        if param_name != "window_days":
            return value

        tried = {
            t.params["window_days"]
            for t in study.get_trials(deepcopy=False)
            if t.number != trial.number and "window_days" in t.params
        }

        for _ in range(MAX_RESAMPLE_ATTEMPTS):
            if value not in tried:
                return value
            value = super().sample_independent(study, trial, param_name, param_distribution)

        low, high = int(param_distribution.low), int(param_distribution.high)
        untried = [candidate for candidate in range(low, high + 1) if candidate not in tried]
        return random.choice(untried) if untried else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--window-max-days", type=int, default=WINDOW_MAX_DAYS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def build_objective(ctx: PopMatrixContext, seed: int, window_max_days: int):
    def objective(trial: optuna.Trial) -> float:
        window_days = trial.suggest_int("window_days", 0, window_max_days)
        window = ctx.data.n_days if window_days == 0 else window_days

        start = time.perf_counter()
        rankings = recommend_batch(
            ctx.data,
            ctx.prefix,
            n=N_RECS,
            ts=ctx.ts,
            w=window,
            bl_indptr=ctx.bl_indptr,
            bl_indices=ctx.bl_indices,
            seed=seed,
        )
        predictions = rankings_to_predictions(rankings, ctx.data.apps, ctx.eval_df)
        value = evaluate_ndcg20(predictions, ctx.eval_df, n_recs=N_RECS)
        elapsed = time.perf_counter() - start

        trial.set_user_attr("window", window)
        print(
            f"[trial {trial.number:03d}] window_days={window_days} (window={window}) "
            f"{METRIC}={value:.6f} tempo={elapsed:.1f}s"
        )
        return value

    return objective


def main() -> None:
    args = parse_args()

    print(f"Lendo {INTERACTIONS_PATH}...")
    ctx = prepare_train_val_context(INTERACTIONS_PATH)

    study = optuna.create_study(direction="maximize", sampler=UniqueIntTPESampler(seed=args.seed))

    for window_days in INITIAL_WINDOWS:
        if window_days <= args.window_max_days:
            study.enqueue_trial({"window_days": window_days})

    objective = build_objective(ctx, args.seed, args.window_max_days)
    print(
        f"Rodando {args.n_trials} trials (window_days em [0, {args.window_max_days}], "
        f"0 = todas as datas de treino disponiveis) -- metrica {METRIC}..."
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    best_window = best.user_attrs["window"]
    print(f"Melhor window_days: {best.params['window_days']} (window={best_window})")
    print(f"Melhor {METRIC}: {study.best_value:.6f}")

    print("Reconferindo a black list no melhor trial (garantia final antes de salvar)...")
    rankings = recommend_batch(
        ctx.data,
        ctx.prefix,
        n=N_RECS,
        ts=ctx.ts,
        w=best_window,
        bl_indptr=ctx.bl_indptr,
        bl_indices=ctx.bl_indices,
        seed=args.seed,
    )
    predictions = rankings_to_predictions(rankings, ctx.data.apps, ctx.eval_df)
    verify_blacklist_respected(predictions, ctx.eval_df, [f"rec{j:03d}" for j in range(N_RECS)])

    results_dir = f"{RESULTS_DIR}/{args.seed}"
    os.makedirs(results_dir, exist_ok=True)

    results_path = f"{results_dir}/optuna_results.csv"
    study.trials_dataframe().to_csv(results_path, index=False)
    print(f"Historico de trials salvo em {results_path}")

    study_path = f"{results_dir}/optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"Study salvo em {study_path}")


if __name__ == "__main__":
    main()
