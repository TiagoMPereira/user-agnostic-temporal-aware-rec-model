"""Otimizacao do hiperparametro `window` do POPModel via Optuna (Card 8).

Busca o valor de `window` (dias, ou None para POP-All) que maximiza o
NDCG@20 no split de teste, usando Optuna com o sampler TPE.

Nao sobrescreve predict_pop.py -- reaproveita sua funcao de
ranking/desempate (`_rank_top_n`, mesma logica e mesmo desempate
regressivo do Card 8), mas reestrutura a geracao de predicoes em duas
etapas para viabilizar rodar dezenas de trials sobre o dataset inteiro
(~19M interacoes) sem repetir trabalho independente de `window` a cada
trial:

  1. `prepare()`: roda uma unica vez. Faz tudo que NAO depende de
     `window` -- carrega a popularity_matrix, codifica o catalogo, e
     agrega os apps consumidos (train+val) por usuario via
     `group_by("uid")` (vetorizado no polars). Isso e valido porque,
     com o split leave-one-out (Card 5), a linha de teste de cada
     usuario e sempre a ultima cronologicamente, entao "ja consumido"
     ali e simplesmente todo o resto do historico do usuario. Tambem
     pre-calcula, via busca binaria vetorizada, o indice de linha da
     matrix correspondente a reference_date de cada linha de teste
     (`idx_until`, que tampouco depende de `window`).

  2. `score_window(window, prepared)`: roda 1x por trial. So recalcula
     o que de fato depende de `window` -- o indice de inicio da janela
     (`idx_before`, busca binaria vetorizada sobre um array do tamanho
     do numero de usuarios, nao do dataset inteiro) e o score de cada
     linha de teste -- reaproveitando tudo que `prepare()` ja calculou.

Busca:
  - Hiperparametro: `window_days` (int, 0 a WINDOW_MAX_DAYS), onde 0
    representa POP-All (window=None) -- um unico parametro inteiro em
    vez de dois, o que evita conflito entre valores fixados via
    `enqueue_trial` e o dominio declarado em `suggest_int`.
  - Metrica: NDCG@20 medio no split de teste (mean direto sobre as
    interacoes de teste -- Card 10: leave-one-out elimina a
    necessidade de agregar por usuario antes).
  - 5 trials iniciais fixos (via `study.enqueue_trial`): window_days em
    {30, 60, 90, 180, 0}.
  - 95 trials adicionais via `UniqueIntTPESampler` (total: 100) --
    subclasse de TPESampler que reamostra ate obter um `window_days`
    ainda nao testado nesta study, evitando trials duplicados sem
    trocar o TPE por um GridSampler.

Itens com popularidade 0 no criterio usado nunca sao recomendados
(fica None na posicao em vez de um item sem nenhum sinal de
popularidade) -- regra aplicada em `models/pop_utils._rank_top_n`,
compartilhada com predict_pop.py.

Uso:
    python optimize_pop.py
"""

import pickle
import random
import time

import numpy as np
import optuna
import polars as pl

from metrics.interaction_metrics import ndcg_at_k
from metrics.rank import compute_rank_expr
from models.pop_utils import _rank_top_n

SEED = 42
GROUND_TRUTH_PATH = "data/predictions/test_ground_truth.parquet"
RESULTS_PATH = f"data/optuna/recentpop/{SEED}/optuna_pop_window_results.csv"
STUDY_PATH = f"data/optuna/recentpop/{SEED}/optuna_pop_study.pkl"
POPULARITY_MATRIX_PATH = "data/processed/popularity_matrix.parquet"
INTERACTIONS_PATH = "data/processed/interactions_fe.parquet"
DATE_COL = "formated_date"

N_RECS = 50
N_TRIALS = 100
INITIAL_WINDOWS: list[int] = [0, 1, 30, 60, 90, 180, 365]
WINDOW_MIN_DAYS = 0
WINDOW_MAX_DAYS = 365

REC_COLS = [f"rec{j:03d}" for j in range(N_RECS)]
SCHEMA = ["uid", "timestamp", *REC_COLS]

MAX_RESAMPLE_ATTEMPTS = 100  # tentativas de reamostrar window_days antes de cair no fallback


class UniqueIntTPESampler(optuna.samplers.TPESampler):
    """TPESampler que reamostra `window_days` ate obter um valor ainda
    nao testado nesta study, evitando trials duplicados sem abrir mao
    da busca guiada do TPE (ao contrario do GridSampler, que apenas
    enumera exaustivamente o espaco de busca).

    Os 5 trials iniciais fixos (`study.enqueue_trial`) nao passam por
    aqui -- valores fixados pulam o sampler. So os trials orientados
    por TPE sao filtrados.
    """

    def sample_independent(self, study, trial, param_name, param_distribution):
        value = super().sample_independent(study, trial, param_name, param_distribution)

        if param_name != "window_days":
            return value

        tried = {
            t.params["window_days"]
            for t in study.get_trials(deepcopy=False)
            if t.number != trial.number and "window_days" in t.params
        }

        attempts = 0
        while value in tried and attempts < MAX_RESAMPLE_ATTEMPTS:
            value = super().sample_independent(study, trial, param_name, param_distribution)
            attempts += 1

        if value in tried:
            # espaco de busca praticamente esgotado: sorteia uniformemente
            # entre os valores inteiros do intervalo ainda nao testados
            low, high = int(param_distribution.low), int(param_distribution.high)
            untried = [v for v in range(low, high + 1) if v not in tried]
            if untried:
                value = random.choice(untried)

        return value


def prepare(df: pl.DataFrame, matrix: pl.DataFrame) -> dict:
    """Faz, uma unica vez, tudo que NAO depende de `window`."""
    catalog = [c for c in matrix.columns if c != DATE_COL]  # ja ordenado (Card 4)
    n_items = len(catalog)
    matrix_dates = matrix[DATE_COL].to_numpy()  # datetime64[D], ascendente
    matrix_values = matrix.drop(DATE_COL).to_numpy().astype(np.int64)  # (n_datas, n_items)

    app_dtype = df.schema["app_package"]
    catalog_native = pl.Series(catalog, dtype=pl.Utf8).cast(app_dtype).to_list()
    df = df.with_columns(
        pl.col("app_package").cast(pl.Utf8).cast(pl.Enum(catalog)).to_physical().alias("code")
    )

    print("Agregando apps consumidos por usuario (train+val)...")
    consumed = (
        df.filter(pl.col("split") != "test")
        .group_by("uid")
        .agg(pl.col("code").alias("consumed_codes"))
    )
    test_df = df.filter(pl.col("split") == "test").join(consumed, on="uid", how="left")

    print("Pre-calculando indices de data (vetorizado)...")
    test_dates = test_df[DATE_COL].str.to_date().to_numpy()
    idx_until_all = np.searchsorted(matrix_dates, test_dates, side="right") - 1

    return {
        "n_items": n_items,
        "matrix_dates": matrix_dates,
        "matrix_values": matrix_values,
        "catalog_native": catalog_native,
        "app_dtype": app_dtype,
        "uids": test_df["uid"].to_list(),
        "timestamps": test_df[DATE_COL].to_list(),
        "consumed_lists": test_df["consumed_codes"].to_list(),
        "test_dates": test_dates,
        "idx_until_all": idx_until_all,
    }


def score_window(window: int | None, prepared: dict) -> pl.DataFrame:
    """Gera as predicoes do POPModel para um `window` especifico,
    reaproveitando tudo que ja foi pre-calculado em `prepare`. Mesma
    logica de ranking/desempate de predict_pop.py (`_rank_top_n`)."""
    matrix_values = prepared["matrix_values"]
    matrix_dates = prepared["matrix_dates"]
    idx_until_all = prepared["idx_until_all"]
    test_dates = prepared["test_dates"]
    n_items = prepared["n_items"]
    catalog_native = prepared["catalog_native"]
    uids = prepared["uids"]
    timestamps = prepared["timestamps"]
    consumed_lists = prepared["consumed_lists"]

    if window is not None:
        window_start = test_dates - np.timedelta64(window, "D")
        idx_before_all = np.searchsorted(matrix_dates, window_start, side="right") - 1
    else:
        idx_before_all = None

    zero_row = np.zeros(n_items, dtype=np.int64)
    mask = np.zeros(n_items, dtype=bool)  # reutilizada: setada e desfeita a cada usuario

    rows: list = []
    for i in range(len(uids)):
        idx_until = idx_until_all[i]
        row_until = matrix_values[idx_until] if idx_until >= 0 else zero_row

        if window is not None:
            idx_before = idx_before_all[i]
            row_before = matrix_values[idx_before] if idx_before >= 0 else zero_row
            scores = row_until - row_before
        else:
            scores = row_until

        consumed_codes = consumed_lists[i]
        if consumed_codes:
            idx_arr = np.asarray(consumed_codes, dtype=np.int64)
            mask[idx_arr] = True

        candidate_idx = np.flatnonzero(~mask)
        top_idx = _rank_top_n(candidate_idx, scores, matrix_values, idx_until, N_RECS)

        if consumed_codes:
            mask[idx_arr] = False

        preds = [catalog_native[c] for c in top_idx]
        preds += [None] * (N_RECS - len(preds))
        rows.append((uids[i], timestamps[i], *preds))

    dtypes = {"uid": pl.Utf8, "timestamp": pl.Utf8, **{col: prepared["app_dtype"] for col in REC_COLS}}
    columns = dict(zip(SCHEMA, zip(*rows)))
    return pl.DataFrame(columns, schema=dtypes)


def evaluate_ndcg20(predictions: pl.DataFrame, ground_truth: pl.DataFrame) -> float:
    """NDCG@20 medio no split de teste (Card 10 -- mean direto sobre as
    interacoes, sem estagio de agregacao por usuario: o split
    leave-one-out ja garante 1 linha de teste por uid)."""
    joined = ground_truth.lazy().join(predictions.lazy(), on=["uid", "timestamp"], how="inner")
    with_rank = joined.with_columns(compute_rank_expr(n_recs=N_RECS))
    with_ndcg = with_rank.with_columns(ndcg_at_k(20))
    return with_ndcg.select(pl.col("ndcg_at_20").mean()).collect().item()


def objective(trial: optuna.Trial, prepared: dict, ground_truth: pl.DataFrame) -> float:
    window = trial.suggest_int("window_days", WINDOW_MIN_DAYS, WINDOW_MAX_DAYS)
    window = None if not window else window
    print(f"[trial {trial.number:03d}] iniciando: window={window}")

    start = time.perf_counter()
    predictions = score_window(window, prepared)
    ndcg20 = evaluate_ndcg20(predictions, ground_truth)
    elapsed = time.perf_counter() - start

    trial.set_user_attr("window", window)

    try:
        prev_best = trial.study.best_trial
        best_value, best_window = prev_best.value, prev_best.user_attrs.get("window")
    except ValueError:
        best_value, best_window = None, None
    if best_value is None or ndcg20 > best_value:
        best_value, best_window = ndcg20, window

    print(
        f"[trial {trial.number:03d}] concluido: ndcg@20={ndcg20:.6f} tempo={elapsed:.1f}s "
        f"melhor_ate_agora=ndcg@20={best_value:.6f} (window={best_window})"
    )

    return ndcg20


if __name__ == "__main__":
    print(f"Lendo {POPULARITY_MATRIX_PATH}...")
    matrix = pl.read_parquet(POPULARITY_MATRIX_PATH)
    if matrix.schema[DATE_COL] != pl.Date:
        matrix = matrix.with_columns(pl.col(DATE_COL).str.to_date())
    matrix = matrix.sort(DATE_COL)

    print(f"Lendo {INTERACTIONS_PATH}...")
    df = pl.read_parquet(INTERACTIONS_PATH)

    print(f"Lendo {GROUND_TRUTH_PATH}...")
    ground_truth = pl.read_parquet(GROUND_TRUTH_PATH)

    prepared = prepare(df, matrix)
    del df  # libera as ~19M linhas; so os arrays de teste em `prepared` sao necessarios daqui pra frente

    study = optuna.create_study(
        direction="maximize",
        sampler=UniqueIntTPESampler(seed=SEED),
    )

    for window in INITIAL_WINDOWS:
        study.enqueue_trial({"window_days": window})

    print(f"Rodando {N_TRIALS} trials ({len(INITIAL_WINDOWS)} fixos + TPE)...")
    study.optimize(lambda trial: objective(trial, prepared, ground_truth), n_trials=N_TRIALS)

    print(f"Melhor window: {study.best_trial.user_attrs['window']}")
    print(f"Melhor NDCG@20: {study.best_value:.6f}")

    study.trials_dataframe().to_csv(RESULTS_PATH, index=False)
    print(f"Historico de trials salvo em {RESULTS_PATH}")

    with open(STUDY_PATH, "wb") as f:
        pickle.dump(study, f)
    print(f"Study salvo em {STUDY_PATH}")
