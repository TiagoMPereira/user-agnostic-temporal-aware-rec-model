"""Otimizacao conjunta dos hiperparametros `window` e `lambda` do
recentDecayPop via Optuna -- combinacao do POPModel (janela de recencia,
optimize_pop.py) com o decayPop (decaimento exponencial,
optimize_decay_pop.py).

Busca o par (`window`, `lambda`) que maximiza o NDCG@20 no split de
VALIDACAO (val), usando Optuna com o sampler TPE. O split de teste NAO e
usado em nenhum momento aqui -- nem para gerar predicoes, nem para
calcular ground truth. Isso evita que a escolha dos hiperparametros seja
contaminada pelos dados que serao usados para reportar a metrica final.

Formula do score (recentDecayPop):
  No decayPop, a matriz de decaimento D(d) acumula o peso exponencial de
  TODO o historico anterior a `d`: D(d) = soma, para toda interacao em
  `ti < d`, de exp(-lambda * (d - ti)) (ver utils.decay_popularity_matrix
  .decay_from_daily_counts). Assim como em optimize_pop.py, a matriz so
  tem linha para datas com >=1 interacao: por isso `d_until` (a data
  usada de fato para pontuar) e a data valida mais recente <=
  reference_date -- e todo o resto abaixo trabalha em cima de `d_until`,
  nao da reference_date literal.

  O recentDecayPop restringe a soma de D a uma janela [d_before,
  d_until), onde `d_before` e a data valida mais recente <= (reference_date
  - window) -- mesmo criterio de snap que o POPModel ja usa para o
  proprio window (predict_pop.py: idx_before via searchsorted). Como
  D(d_until) ja acumula tudo antes de `d_until`, a parcela ANTERIOR a
  janela pode ser subtraida algebricamente: com gap = d_until - d_before
  (em dias, o gap REAL entre as duas linhas da matriz),

      score = D(d_until) - exp(-lambda * gap) * D(d_before)

  Prova: exp(-lambda*gap) * D(d_before)
       = exp(-lambda*gap) * soma_{ti < d_before} exp(-lambda*(d_before - ti))
       = soma_{ti < d_before} exp(-lambda*(d_before - ti + gap))
       = soma_{ti < d_before} exp(-lambda*(d_until - ti))
  ou seja, e exatamente a soma ponderada das interacoes anteriores a
  `d_before`, ja reescalada para `d_until`. Subtraindo de D(d_until)
  sobra soma_{d_before <= ti < d_until} exp(-lambda*(d_until - ti)) -- a
  janela desejada, com AMBOS os limites ancorados nas datas de linha
  realmente usadas (nao nos valores nominais `reference_date` e
  `reference_date - window`). Ancorar o gap em `d_until` (em vez da
  reference_date literal) e essencial: sem isso a subtracao deixa de ser
  exata -- foi validado numericamente contra um calculo por forca bruta
  antes deste script ser finalizado. window=0 (ou None) equivale a nao
  aplicar corte de janela nenhum -- reduz ao decayPop puro.

Fluxo completo (dois scripts separados):
  1. `optimize_recent_decay_pop.py` (este arquivo): treino -> prediz
     validacao, escolhe o melhor par (window, lambda) por NDCG@20 no val.
  2. `predict_recent_decay_pop.py --window <melhor> --lambda <melhor>` +
     `evaluate_predictions.py`: roda a predicao de fato sobre o split de
     teste (unica vez, com os hiperparametros ja escolhidos) para obter
     a metrica real reportada.

Nao sobrescreve predict_recent_decay_pop.py -- reaproveita sua funcao de
ranking/desempate (`_rank_top_n`, mesma logica e mesmo desempate
regressivo do POPModel/decayPop), mas reestrutura a geracao de predicoes
em duas etapas para viabilizar rodar dezenas de trials sobre o dataset
inteiro (~19M interacoes) sem repetir trabalho independente de (window,
lambda) a cada trial:

  1. `prepare()`: roda uma unica vez. Faz tudo que NAO depende de
     `window` nem de `lambda` -- pivota as contagens diarias brutas por
     item/data (`pivot_daily_counts`, base da matriz de decaimento),
     codifica o catalogo, e agrega os apps consumidos (somente TREINO)
     por usuario via `group_by("uid")` (vetorizado no polars). Isso e
     valido porque, com o split leave-one-out (Card 5), a linha de val
     de cada usuario e sempre a penultima cronologicamente (a ultima e o
     teste, que fica de fora), entao "ja consumido" ali e todo o
     historico de treino do usuario. Tambem pre-calcula, via busca
     binaria vetorizada, o indice de linha correspondente a
     reference_date de cada linha de val (`idx_until_all`, que nao
     depende de window nem de lambda -- as datas da matriz sao as mesmas
     para qualquer valor de ambos), e extrai o ground truth de val (uid,
     app_package, timestamp) direto do dataframe ja carregado.

  2. `score_recent_decay(window, lambda_, prepared)`: roda 1x por trial.
     Aplica o decaimento exponencial sobre as contagens diarias ja
     pivotadas (`decay_from_daily_counts`, a parte mais cara, que
     depende de `lambda`) e, em seguida, o corte de janela via subtracao
     algebrica derivada acima (que depende de `window`, mas e barata --
     apenas indexacao vetorizada), pontuando cada linha de val e
     reaproveitando tudo que `prepare()` ja calculou.

Busca:
  - Hiperparametros: `window_days` (int, WINDOW_MIN_DAYS a
    WINDOW_MAX_DAYS, onde 0 representa "sem corte de janela" -- reduz ao
    decayPop puro) e `lambda_` (float, LAMBDA_MIN a LAMBDA_MAX).
  - Metrica: NDCG@20 medio no split de validacao (mean direto sobre as
    interacoes de val -- leave-one-out elimina a necessidade de agregar
    por usuario antes).
  - Trials iniciais fixos (via `study.enqueue_trial`): produto cartesiano
    de INITIAL_WINDOWS x INITIAL_LAMBDAS, cobrindo os cantos e pontos
    intermediarios do espaco de busca 2D antes de partir para o TPE.
  - Diferente de optimize_pop.py, aqui nao ha necessidade de um sampler
    que evite trials duplicados: com `lambda_` continuo no par, a chance
    de o TPE reamostrar exatamente o mesmo par (window_days, lambda_) e
    desprezivel (mesmo raciocinio de optimize_decay_pop.py).

Itens com popularidade 0 no criterio usado nunca sao recomendados (fica
None na posicao em vez de um item sem nenhum sinal de popularidade) --
regra aplicada em `models/pop_utils._rank_top_n`, compartilhada com
predict_recent_decay_pop.py.

Uso:
    python optimize_recent_decay_pop.py
    python predict_recent_decay_pop.py --window <melhor_window> --lambda <melhor_lambda>
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
RESULTS_PATH = f"data/optuna/recentdecaypop/optuna_recent_decay_pop_results.csv"
STUDY_PATH = f"data/optuna/recentdecaypop/optuna_recent_decay_pop_study.pkl"
RAW_INTERACTIONS_PATH = "data/processed/interactions.parquet"
INTERACTIONS_PATH = "data/processed/interactions_fe.parquet"
DATE_COL = "formated_date"

N_RECS = 50
N_TRIALS = 150
INITIAL_WINDOWS: list[int] = [0, 1, 365]
INITIAL_LAMBDAS: list[float] = [0.001, 1.0]
WINDOW_MIN_DAYS = 0
WINDOW_MAX_DAYS = 365
LAMBDA_MIN = 0.001
LAMBDA_MAX = 1.0

REC_COLS = [f"rec{j:03d}" for j in range(N_RECS)]
SCHEMA = ["uid", "timestamp", *REC_COLS]


def prepare(df: pl.DataFrame, raw_df: pl.DataFrame) -> dict:
    """Faz, uma unica vez, tudo que NAO depende de `window` nem de `lambda`."""
    print("Pivotando contagens diarias por item/data (independente de window/lambda)...")
    daily_counts = pivot_daily_counts(raw_df).sort(DATE_COL)
    if daily_counts.schema[DATE_COL] != pl.Date:
        daily_counts = daily_counts.with_columns(pl.col(DATE_COL).str.to_date())
    assert daily_counts[DATE_COL].is_sorted(), (
        "daily_counts precisa estar ordenado por data ascendente -- tanto o "
        "np.searchsorted abaixo quanto a recorrencia em decay_from_daily_counts "
        "assumem isso"
    )

    catalog = [c for c in daily_counts.columns if c != DATE_COL]  # ja ordenado
    assert list(daily_counts.drop(DATE_COL).columns) == catalog, (
        "ordem das colunas de daily_counts diverge de catalog -- os codigos de "
        "item gerados via pl.Enum(catalog) deixariam de corresponder as colunas "
        "de matrix_values indexadas por esses mesmos codigos em score_recent_decay"
    )
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
        "matrix_dates": matrix_dates,
        "n_items": len(catalog),
        "catalog_native": catalog_native,
        "app_dtype": app_dtype,
        "uids": val_df["uid"].to_list(),
        "timestamps": val_df[DATE_COL].to_list(),
        "consumed_lists": val_df["consumed_codes"].to_list(),
        "val_dates": val_dates,
        "idx_until_all": idx_until_all,
        "ground_truth": ground_truth,
    }


def score_recent_decay(window: int | None, lambda_: float, prepared: dict) -> pl.DataFrame:
    """Gera as predicoes do recentDecayPop para um par (window, lambda)
    especifico, reaproveitando tudo que ja foi pre-calculado em
    `prepare`. Mesma logica de ranking/desempate de
    predict_recent_decay_pop.py (`_rank_top_n`)."""
    daily_counts = prepared["daily_counts"]
    matrix_dates = prepared["matrix_dates"]
    idx_until_all = prepared["idx_until_all"]
    val_dates = prepared["val_dates"]
    n_items = prepared["n_items"]
    catalog_native = prepared["catalog_native"]
    uids = prepared["uids"]
    timestamps = prepared["timestamps"]
    consumed_lists = prepared["consumed_lists"]

    matrix = decay_from_daily_counts(daily_counts, lambda_, DATE_COL)
    matrix_values = matrix.drop(DATE_COL).to_numpy().astype(np.float64)

    if window is not None:
        window_start = val_dates - np.timedelta64(window, "D")
        idx_before_all = np.searchsorted(matrix_dates, window_start, side="right") - 1
        decay_factor_all = np.zeros(len(val_dates), dtype=np.float64)
        valid_before = idx_before_all >= 0
        # gap ancorado em matrix_dates[idx_until] -- a data da linha que
        # row_until de fato usa (ver derivacao no docstring do modulo) --,
        # NAO em val_dates: idx_before <= idx_until sempre (window >= 0),
        # entao onde valid_before e True, idx_until_all tambem e >= 0.
        d_until_valid = matrix_dates[idx_until_all[valid_before]]
        gap_days = (
            (d_until_valid - matrix_dates[idx_before_all[valid_before]])
            .astype("timedelta64[D]")
            .astype(np.int64)
        )
        decay_factor_all[valid_before] = np.exp(-lambda_ * gap_days)
    else:
        idx_before_all = None
        decay_factor_all = None

    zero_row = np.zeros(n_items, dtype=np.float64)
    mask = np.zeros(n_items, dtype=bool)  # reutilizada: setada e desfeita a cada usuario

    rows: list = []
    for i in range(len(uids)):
        idx_until = idx_until_all[i]
        row_until = matrix_values[idx_until] if idx_until >= 0 else zero_row

        if window is not None:
            idx_before = idx_before_all[i]
            if idx_before >= 0:
                row_before = matrix_values[idx_before] * decay_factor_all[i]
            else:
                row_before = zero_row
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
    """NDCG@20 medio no split de validacao (mean direto sobre as
    interacoes, sem estagio de agregacao por usuario: o split
    leave-one-out ja garante 1 linha de val por uid)."""
    joined = ground_truth.lazy().join(predictions.lazy(), on=["uid", "timestamp"], how="inner")
    with_rank = joined.with_columns(compute_rank_expr(n_recs=N_RECS))
    with_ndcg = with_rank.with_columns(ndcg_at_k(20))
    return with_ndcg.select(pl.col("ndcg_at_20").mean()).collect().item()


def objective(trial: optuna.Trial, prepared: dict, ground_truth: pl.DataFrame) -> float:
    window_days = trial.suggest_int("window_days", WINDOW_MIN_DAYS, WINDOW_MAX_DAYS)
    window = None if not window_days else window_days
    lambda_ = trial.suggest_float("lambda_", LAMBDA_MIN, LAMBDA_MAX)
    print(f"[trial {trial.number:03d}] iniciando: window={window} lambda={lambda_:.6f}")

    start = time.perf_counter()
    predictions = score_recent_decay(window, lambda_, prepared)
    ndcg20 = evaluate_ndcg20(predictions, ground_truth)
    elapsed = time.perf_counter() - start

    trial.set_user_attr("window", window)

    try:
        prev_best = trial.study.best_trial
        best_value = prev_best.value
        best_window = prev_best.user_attrs.get("window")
        best_lambda = prev_best.params.get("lambda_")
    except ValueError:
        best_value, best_window, best_lambda = None, None, None
    if best_value is None or ndcg20 > best_value:
        best_value, best_window, best_lambda = ndcg20, window, lambda_

    print(
        f"[trial {trial.number:03d}] concluido: ndcg@20={ndcg20:.6f} tempo={elapsed:.1f}s "
        f"melhor_ate_agora=ndcg@20={best_value:.6f} (window={best_window}, lambda={best_lambda:.6f})"
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

    for window in INITIAL_WINDOWS:
        for lambda_ in INITIAL_LAMBDAS:
            study.enqueue_trial({"window_days": window, "lambda_": lambda_})

    n_initial = len(INITIAL_WINDOWS) * len(INITIAL_LAMBDAS)
    print(f"Rodando {N_TRIALS} trials ({n_initial} fixos + TPE)...")
    study.optimize(lambda trial: objective(trial, prepared, ground_truth), n_trials=N_TRIALS)

    print(f"Melhor window: {study.best_trial.user_attrs['window']}")
    print(f"Melhor lambda: {study.best_params['lambda_']:.6f}")
    print(f"Melhor NDCG@20: {study.best_value:.6f}")

    study.trials_dataframe().to_csv(RESULTS_PATH, index=False)
    print(f"Historico de trials salvo em {RESULTS_PATH}")

    with open(STUDY_PATH, "wb") as f:
        pickle.dump(study, f)
    print(f"Study salvo em {STUDY_PATH}")
