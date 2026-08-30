import numpy as np

def _break_ties(items: np.ndarray, matrix_values: np.ndarray, idx_until: int) -> np.ndarray:
    """Desempata `items` (codigos de app com o mesmo score), regressivamente,
    pela popularidade diaria dos dias anteriores a reference_date.

    Tier 1 compara a contagem diaria do dia em matrix[idx_until - 1]
    (o dia imediatamente anterior a reference_date); se persistir o
    empate, tier 2 compara o dia anterior a esse, e assim por diante.
    Ao esgotar o historico da matrix, o empate remanescente e resolvido
    pelo codigo do item -- que corresponde a ordem alfabetica do
    catalogo (ja ordenado, ver Card 4).
    """
    groups = [items]
    tier = 1

    while any(g.size > 1 for g in groups):
        lo = idx_until - tier
        new_groups = []

        for g in groups:
            if g.size <= 1:
                new_groups.append(g)
                continue

            if lo < 0:
                # historico esgotado: cada item vira seu proprio grupo, na
                # ordem do codigo (== ordem alfabetica do catalogo, Card 4)
                new_groups.extend(np.split(np.sort(g), np.arange(1, g.size)))
                continue

            daily = matrix_values[lo + 1, g] - matrix_values[lo, g]
            order = np.argsort(-daily, kind="stable")
            g_sorted = g[order]
            daily_sorted = daily[order]
            boundaries = np.flatnonzero(np.diff(daily_sorted) != 0) + 1
            new_groups.extend(np.split(g_sorted, boundaries))

        groups = new_groups
        tier += 1

    return np.concatenate(groups)


def _rank_top_n(
    candidate_idx: np.ndarray,
    scores: np.ndarray,
    matrix_values: np.ndarray,
    idx_until: int,
    n: int,
) -> np.ndarray:
    """Retorna ate `n` codigos de `candidate_idx`, do mais para o menos
    relevante, ordenados por score (descendente) com desempate
    regressivo por popularidade diaria (ver `_break_ties`).

    Itens com score 0 (nenhuma popularidade no criterio usado) nunca
    sao recomendados -- ficam de fora do ranking em vez de preencher
    posicoes com um item sem nenhum sinal de popularidade. Se sobrarem
    menos de `n` itens com score > 0, as posicoes restantes ficam
    vazias (o chamador preenche com None).
    """
    candidate_scores = scores[candidate_idx]

    positive = candidate_scores > 0
    candidate_idx = candidate_idx[positive]
    candidate_scores = candidate_scores[positive]

    if candidate_idx.size > n:
        threshold = np.partition(candidate_scores, -n)[-n]
        keep = candidate_scores >= threshold
        candidate_idx = candidate_idx[keep]
        candidate_scores = candidate_scores[keep]

    order = np.argsort(-candidate_scores, kind="stable")
    sorted_idx = candidate_idx[order]
    sorted_scores = candidate_scores[order]

    boundaries = np.flatnonzero(np.diff(sorted_scores) != 0) + 1
    groups = np.split(sorted_idx, boundaries)

    resolved = [
        _break_ties(g, matrix_values, idx_until) if g.size > 1 else g for g in groups
    ]
    return np.concatenate(resolved)[:n]