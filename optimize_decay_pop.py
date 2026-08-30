"""Otimizacao do hiperparametro `lambda` do decayPop via Optuna.

Busca o valor de `lambda` (fator de decaimento exponencial, 0.001 a 1)
que maximiza o NDCG@20 no split de VALIDACAO (val), usando Optuna com o
sampler TPE. O split de teste NAO e usado em nenhum momento aqui -- nem
para gerar predicoes, nem para calcular ground truth. Isso evita que a
escolha do hiperparametro seja contaminada pelos dados que serao usados
para reportar a metrica final.

No decayPop, a matriz de popularidade nao e mais a soma cumulativa
bruta desde o inicio do historico: cada interacao passada contribui com
um peso exp(-lambda * (t - ti)), onde `t` e a data de referencia (o dia
atual) e `ti` o dia em que a interacao ocorreu (ver
utils.decay_popularity_matrix). O catalogo de itens candidatos e sempre
o catalogo inteiro (sem corte), como em optimize_pop.py.

Fluxo completo (dois scripts separados):
  1. `optimize_decay_pop.py` (este arquivo): treino -> prediz validacao,
     escolhe o melhor `lambda` por NDCG@20 no val.
  2. `predict_decay_pop.py --lambda <melhor>` + `evaluate_predictions.py`:
     roda a predicao de fato sobre o split de teste (unica vez, com o
     lambda ja escolhido) para obter a metrica real reportada.

Nao sobrescreve predict_decay_pop.py -- reaproveita sua funcao de
ranking/desempate (`_rank_top_n`, mesma logica e mesmo desempate
regressivo do POPModel), mas reestrutura a geracao de predicoes em duas
etapas para viabilizar rodar dezenas de trials sobre o dataset inteiro
(~19M interacoes) sem repetir trabalho independente de `lambda` a cada
trial:

  1. `prepare()`: roda uma unica vez. Faz tudo que NAO depende de
     `lambda` -- pivota as contagens diarias brutas por item/data
     (`pivot_daily_counts`, base da matriz de decaimento), codifica o
     catalogo, e agrega os apps consumidos (somente TREINO) por usuario
     via `group_by("uid")` (vetorizado no polars). Isso e valido porque,
     com o split leave-one-out (Card 5), a linha de val de cada usuario
     e sempre a penultima cronologicamente (a ultima e o teste, que fica
     de fora), entao "ja consumido" ali e todo o historico de treino do
     usuario. Tambem pre-calcula, via busca binaria vetorizada, o indice
     de linha correspondente a reference_date de cada linha de val
     (`idx_until_all`, que tampouco depende de `lambda` -- as datas da
     matriz sao as mesmas para qualquer `lambda`), e extrai o ground
     truth de val (uid, app_package, timestamp) direto do dataframe ja
     carregado.

  2. `score_lambda(lambda_, prepared)`: roda 1x por trial. Aplica o
     decaimento exponencial sobre as contagens diarias ja pivotadas
     (`decay_from_daily_counts`, unica parte que de fato depende de
     `lambda`) e pontua cada linha de val, reaproveitando tudo que
     `prepare()` ja calculou.

Busca:
  - Hiperparametro: `lambda_` (float, LAMBDA_MIN a LAMBDA_MAX).
  - Metrica: NDCG@20 medio no split de validacao (mean direto sobre as
    interacoes de val -- leave-one-out elimina a necessidade de agregar
    por usuario antes).
  - Alguns trials iniciais fixos (via `study.enqueue_trial`), cobrindo a
    faixa de busca (INITIAL_LAMBDAS), seguidos de trials guiados por TPE
    (total: N_TRIALS). Ao contrario de optimize_pop.py (`window_days`,
    inteiro), aqui nao ha necessidade de um sampler que evite valores
    repetidos: `lambda_` e continuo, a chance de o TPE reamostrar
    exatamente o mesmo float e desprezivel.

Itens com popularidade 0 no criterio usado nunca sao recomendados (fica
None na posicao em vez de um item sem nenhum sinal de popularidade) --
regra aplicada em `models/pop_utils._rank_top_n`, compartilhada com
predict_decay_pop.py.

Uso:
    python optimize_decay_pop.py
    python predict_decay_pop.py --lambda <melhor_lambda_encontrado>
    python evaluate_predictions.py
"""

import pickle
import time

import numpy as np
import optuna
import polars as pl

from metrics.interaction_metrics import ndcg_at_k
from metrics.rank import compute_rank_expr
from models.pop_utils import _rank_top_n
from utils.decay_popularity_matrix import decay_from_daily_counts, pivot_daily_counts

SEED = 42
RESULTS_PATH = f"data/optuna/decaypop/{SEED}/optuna_decay_pop_lambda_results.csv"
STUDY_PATH = f"data/optuna/decaypop/{SEED}/optuna_decay_pop_study.pkl"
RAW_INTERACTIONS_PATH = "data/processed/interactions.parquet"
INTERACTIONS_PATH = "data/processed/interactions_fe.parquet"
DATE_COL = "formated_date"

N_RECS = 50
N_TRIALS = 100
INITIAL_LAMBDAS: list[float] = [0.001, 0.01, 0.1, 1.0]
LAMBDA_MIN = 0.001
LAMBDA_MAX = 1.0

REC_COLS = [f"rec{j:03d}" for j in range(N_RECS)]
SCHEMA = ["uid", "timestamp", *REC_COLS]


def prepare(df: pl.DataFrame, raw_df: pl.DataFrame) -> dict:
    """Faz, uma unica vez, tudo que NAO depende de `lambda`."""
    print("Pivotando contagens diarias por item/data (independente de lambda)...")
    daily_counts = pivot_daily_counts(raw_df).sort(DATE_COL)
    if daily_counts.schema[DATE_COL] != pl.Date:
        daily_counts = daily_counts.with_columns(pl.col(DATE_COL).str.to_date())
    catalog = [c for c in daily_counts.columns if c != DATE_COL]  # ja ordenado
    matrix_dates = daily_counts[DATE_COL].to_numpy()  # datetime64[D], ascendente

    app_dtype = df.schema["app_package"]
    catalog_native = pl.Series(catalog, dtype=pl.Utf8).cast(app_dtype).to_list()
    df = df.with_columns(
        pl.col("app_package").cast(pl.Utf8).cast(pl.Enum(catalog)).to_physical().alias("code")
    )

    print("Agregando apps consumidos por usuario (somente treino)...")
    consumed = (
        df.filter(pl.col("split") == "train")
        .group_by("uid")
        .agg(pl.col("code").alias("consumed_codes"))
    )
    val_df = df.filter(pl.col("split") == "val").join(consumed, on="uid", how="left")

    print("Extraindo ground truth de validacao...")
    ground_truth = val_df.select(
        pl.col("uid"),
        pl.col("app_package"),
        pl.col(DATE_COL).alias("timestamp"),
    )

    print("Pre-calculando indices de data (vetorizado)...")
    val_dates = val_df[DATE_COL].str.to_date().to_numpy()
    idx_until_all = np.searchsorted(matrix_dates, val_dates, side="right") - 1

    return {
        "daily_counts": daily_counts,
        "n_items": len(catalog),
        "catalog_native": catalog_native,
        "app_dtype": app_dtype,
        "uids": val_df["uid"].to_list(),
        "timestamps": val_df[DATE_COL].to_list(),
        "consumed_lists": val_df["consumed_codes"].to_list(),
        "idx_until_all": idx_until_all,
        "ground_truth": ground_truth,
    }


def score_lambda(lambda_: float, prepared: dict) -> pl.DataFrame:
    """Gera as predicoes do decayPop para um `lambda` especifico,
    reaproveitando tudo que ja foi pre-calculado em `prepare`. Mesma
    logica de ranking/desempate de predict_decay_pop.py (`_rank_top_n`)."""
    daily_counts = prepared["daily_counts"]
    idx_until_all = prepared["idx_until_all"]
    n_items = prepared["n_items"]
    catalog_native = prepared["catalog_native"]
    uids = prepared["uids"]
    timestamps = prepared["timestamps"]
    consumed_lists = prepared["consumed_lists"]

    matrix = decay_from_daily_counts(daily_counts, lambda_, DATE_COL)
    matrix_values = matrix.drop(DATE_COL).to_numpy().astype(np.float64)

    zero_row = np.zeros(n_items, dtype=np.float64)
    mask = np.zeros(n_items, dtype=bool)  # reutilizada: setada e desfeita a cada usuario

    rows: list = []
    for i in range(len(uids)):
        idx_until = idx_until_all[i]
        scores = matrix_values[idx_until] if idx_until >= 0 else zero_row

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
    """NDCG@20 medio no split de validacao (mean direto sobre as
    interacoes, sem estagio de agregacao por usuario: o split
    leave-one-out ja garante 1 linha de val por uid)."""
    joined = ground_truth.lazy().join(predictions.lazy(), on=["uid", "timestamp"], how="inner")
    with_rank = joined.with_columns(compute_rank_expr(n_recs=N_RECS))
    with_ndcg = with_rank.with_columns(ndcg_at_k(20))
    return with_ndcg.select(pl.col("ndcg_at_20").mean()).collect().item()


def objective(trial: optuna.Trial, prepared: dict, ground_truth: pl.DataFrame) -> float:
    lambda_ = trial.suggest_float("lambda_", LAMBDA_MIN, LAMBDA_MAX)
    print(f"[trial {trial.number:03d}] iniciando: lambda={lambda_:.6f}")

    start = time.perf_counter()
    predictions = score_lambda(lambda_, prepared)
    ndcg20 = evaluate_ndcg20(predictions, ground_truth)
    elapsed = time.perf_counter() - start

    try:
        best_value = trial.study.best_trial.value
    except ValueError:
        best_value = None
    if best_value is None or ndcg20 > best_value:
        best_value = ndcg20

    print(
        f"[trial {trial.number:03d}] concluido: ndcg@20={ndcg20:.6f} tempo={elapsed:.1f}s "
        f"melhor_ate_agora=ndcg@20={best_value:.6f}"
    )

    return ndcg20


if __name__ == "__main__":
    print(f"Lendo {RAW_INTERACTIONS_PATH}...")
    raw_df = pl.read_parquet(RAW_INTERACTIONS_PATH)

    print(f"Lendo {INTERACTIONS_PATH}...")
    df = pl.read_parquet(INTERACTIONS_PATH)

    prepared = prepare(df, raw_df)
    del df, raw_df  # libera as ~19M linhas; so os arrays de val em `prepared` sao necessarios daqui pra frente
    ground_truth = prepared.pop("ground_truth")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    for lambda_ in INITIAL_LAMBDAS:
        study.enqueue_trial({"lambda_": lambda_})

    print(f"Rodando {N_TRIALS} trials ({len(INITIAL_LAMBDAS)} fixos + TPE)...")
    study.optimize(lambda trial: objective(trial, prepared, ground_truth), n_trials=N_TRIALS)

    print(f"Melhor lambda: {study.best_params['lambda_']:.6f}")
    print(f"Melhor NDCG@20: {study.best_value:.6f}")

    study.trials_dataframe().to_csv(RESULTS_PATH, index=False)
    print(f"Historico de trials salvo em {RESULTS_PATH}")

    with open(STUDY_PATH, "wb") as f:
        pickle.dump(study, f)
    print(f"Study salvo em {STUDY_PATH}")
