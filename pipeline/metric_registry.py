"""Registry de metricas ("nota") usadas como objetivo do Optuna ou para
reportar o resultado final.

Nao recalcula metrica nenhuma -- e so uma camada de selecao por nome sobre
`metrics/interaction_metrics.py`, que ja implementa HR@K, NDCG@K e MRR.
Isso resolve a limitacao dos scripts antigos (optimize_pop.py e os demais),
que tinham NDCG@20 fixo no codigo (`evaluate_ndcg20`, duplicada 4x): aqui a
metrica e so mais um parametro (`--metric` em optimize_popularity.py).

Diferenca deliberada em relacao a `metrics.evaluation.compute_rank`: aqui
`n_recs` e sempre o numero real de colunas rec000..rec{n_recs-1} presentes
nas predicoes (default 50, como todo o pipeline de popularidade), nunca o
default global de `metrics.rank` (250) -- inclusive para o `miss_rank` do
MRR, que senao classificaria erroneamente um miss como hit sempre que
n_recs != 250 (mrr_expr usa MISS_RANK=251 por padrao, mas compute_rank_expr
com n_recs=50 atribui rank=51 a um miss -- 51 < 251, seria contado como
hit). Aqui `mrr_expr` sempre recebe `miss_rank=n_recs + 1`, explicito.
"""

import polars as pl

from metrics.interaction_metrics import HR_K_VALUES, NDCG_K_VALUES, hr_at_k, mrr_expr, ndcg_at_k
from metrics.rank import compute_rank_expr

_HR_NAMES: dict[str, int] = {f"hr@{k}": k for k in HR_K_VALUES}
_NDCG_NAMES: dict[str, int] = {f"ndcg@{k}": k for k in NDCG_K_VALUES}

METRIC_NAMES: list[str] = [*_HR_NAMES, *_NDCG_NAMES, "mrr"]
DEFAULT_METRIC = "ndcg@20"


def _metric_expr(metric: str, n_recs: int) -> pl.Expr:
    if metric in _HR_NAMES:
        return hr_at_k(_HR_NAMES[metric]).alias("_metric_value")
    if metric in _NDCG_NAMES:
        return ndcg_at_k(_NDCG_NAMES[metric]).alias("_metric_value")
    if metric == "mrr":
        return mrr_expr(miss_rank=n_recs + 1).alias("_metric_value")
    raise ValueError(f"Metrica '{metric}' desconhecida. Opcoes: {METRIC_NAMES}")


def evaluate(
    predictions: pl.DataFrame,
    ground_truth: pl.DataFrame,
    metric: str = DEFAULT_METRIC,
    n_recs: int = 50,
) -> float:
    """Junta ground_truth x predictions por (uid, timestamp), calcula o
    rank e a metrica escolhida, retorna a media.

    Com o split leave-one-out (Card 5), cada usuario contribui com
    exatamente 1 interacao de avaliacao, entao a media direta sobre as
    interacoes ja e o resultado -- sem estagio de agregacao por usuario
    (mesmo raciocinio de metrics/evaluation.py::compute_summary).
    """
    joined = ground_truth.lazy().join(predictions.lazy(), on=["uid", "timestamp"], how="inner")
    with_rank = joined.with_columns(compute_rank_expr(n_recs=n_recs))
    with_metric = with_rank.with_columns(_metric_expr(metric, n_recs))
    return with_metric.select(pl.col("_metric_value").mean()).collect().item()
