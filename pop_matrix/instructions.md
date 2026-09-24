# Especificação: Recomendador por Popularidade em Janela Deslizante

> Documento de contexto para o assistente de código (Copilot).
> Leia este documento inteiro antes de gerar qualquer código. As regras aqui são requisitos, não sugestões.

---

## 1. Objetivo

Implementar um recomendador baseado em popularidade. Dado um instante `t`, o modelo recomenda os `N` apps mais consumidos no intervalo de tempo `[t - w, t - 1]` (inclusive nas duas pontas), onde `w` é o tamanho da janela. O dia `t` **nunca** entra na contagem (evita vazamento de dados).

O ranking é **global**: para um mesmo `t`, é igual para todos os usuários. A única personalização é a `black_list` (apps que não podem ser recomendados, tipicamente os que o usuário já consumiu).

---

## 2. Dados de entrada

Tabela de interações com cerca de **19 milhões de linhas** e as colunas:

| Coluna        | Tipo esperado                  | Uso                                  |
|---------------|--------------------------------|--------------------------------------|
| `uid`         | string ou inteiro              | Não usado no ranking (só em avaliação) |
| `app_package` | string                         | Identificador do item                |
| `review`      | numérico                       | **Ignorar completamente**            |
| `timestamp`   | epoch (s ou ms) ou datetime    | Tempo da interação                   |

Cada linha conta como **1 interação**, independentemente da nota.

---

## 3. Stack e restrições técnicas

- Python ≥ 3.10, com type hints em todas as funções.
- **Polars** (≥ 1.0) para leitura e pré-processamento, preferencialmente em modo lazy (`scan_parquet` / `scan_csv`).
- **NumPy** para as matrizes e o ranking.
- **Numba** apenas no loop de avaliação em lote (seção 7). Não usar Numba onde NumPy vetorizado resolve.
- **Não usar pandas.**
- **Não usar `pivot` do Polars** para construir a matriz. Uma tabela com uma coluna por app é lenta e pesada.
- **Não usar `np.add.at`** (lento). Após o `group_by`, os pares (dia, app) são únicos, então atribuição direta basta.
- **Nenhum loop Python sobre apps ou sobre linhas do dataset.** Loops Python são aceitáveis apenas sobre valores únicos de `t` na avaliação em lote.
- **Não usar recursão** no desempate. O desempate é uma ordenação lexicográfica (seção 6.3).
- Docstrings no estilo NumPy, em português.

---

## 4. Conceitos e convenções

### 4.1 Granularidade do tempo

O tempo é discretizado em **dias** (parâmetro configurável, padrão `"1d"`). Todo "timestamp" na lógica de recomendação é um índice inteiro de dia.

Antes de truncar para o dia, o timestamp deve ser convertido para um fuso horário explícito (parâmetro `timezone`, padrão `"UTC"`).

### 4.2 Calendário contínuo (regra crítica)

O índice de linha de um dia é:

```
dia_idx = (data - data_minima).em_dias
```

Ou seja, a linha 0 é o primeiro dia do dataset e a linha `T - 1` é o último, **incluindo dias sem nenhuma interação** (linhas de zeros). Assim "t - w" significa sempre "w dias atrás", e nunca "w linhas com dados atrás". **Não** usar o índice da k-ésima data distinta.

### 4.3 Índice de apps

`app_idx` é o rank denso de `app_package` em ordem alfabética, começando em 0. O array `apps` (ordenado) faz o mapeamento inverso: `apps[app_idx] == app_package`.

### 4.4 Matrizes

| Nome | Forma      | dtype   | Significado |
|------|------------|---------|-------------|
| `M`  | `(T, A)`   | `int32` | `M[d, a]` = nº de interações do app `a` no dia `d` |
| `P`  | `(T+1, A)` | `int64` | Soma de prefixos: `P[0] = 0`, `P[k] = M[0] + ... + M[k-1]` |

Contagem na janela de `t` com tamanho `w`:

```
inicio = clip(t - w, 0, T)
fim    = clip(t,     0, T)
totais = P[fim] - P[inicio]          # vetor (A,)
```

`P` independe de `w`: a mesma matriz serve para qualquer tamanho de janela.

### 4.5 Bordas da janela

- Se `t - w < 0`, a janela é **truncada** no dia 0 (usa apenas os dias disponíveis).
- Se `t > T`, os dias além do dataset são tratados como zeros (a janela é truncada em `T`).
- Se `inicio >= fim` (ex.: `t = 0`), a janela é vazia e a recomendação retorna lista vazia.
- `t` negativo ou `w < 1` → `ValueError`.

---

## 5. Estrutura do projeto

```
popularity_recsys/
├── __init__.py
├── data.py        # carregamento, índices, matriz M, prefixos P
├── ranking.py     # contagem na janela, desempate, recomendação individual
├── batch.py       # recomendação em lote com cache por t + kernel Numba
└── debug.py       # conversões para inspeção (opcional)
tests/
├── test_data.py
├── test_ranking.py
└── test_batch.py
```

---

## 6. Funções

### 6.1 `data.py`

#### Estrutura `InteractionData`

```python
from dataclasses import dataclass
import datetime as dt
import numpy as np


@dataclass(frozen=True)
class InteractionData:
    """
    Contêiner com a matriz de interações e os mapeamentos de índice.

    Attributes
    ----------
    matrix : np.ndarray
        Matriz de interações ``M`` de forma ``(T, A)`` e dtype ``int32``.
        ``matrix[d, a]`` é o número de interações do app ``a`` no dia ``d``.
        Dias sem interações existem como linhas de zeros (calendário contínuo).
    apps : np.ndarray
        Array de strings de forma ``(A,)``, ordenado alfabeticamente.
        ``apps[a]`` é o ``app_package`` do índice ``a``.
    start_date : datetime.date
        Data correspondente à linha 0 da matriz.
    app_to_idx : dict[str, int]
        Mapeamento ``app_package -> app_idx``. Construído a partir de ``apps``.

    Notes
    -----
    ``T = matrix.shape[0]`` é o número de dias do calendário contínuo entre
    a primeira e a última data do dataset (inclusive).
    ``A = matrix.shape[1]`` é o número de apps distintos.
    """
    matrix: np.ndarray
    apps: np.ndarray
    start_date: dt.date
    app_to_idx: dict[str, int]

    @property
    def n_days(self) -> int: ...

    @property
    def n_apps(self) -> int: ...
```

#### `load_interactions`

```python
def load_interactions(
    path: str,
    timestamp_unit: str | None = "s",
    timezone: str = "UTC",
) -> pl.LazyFrame:
    """
    Carrega o dataset de interações de forma lazy e normaliza o tempo em dias.

    Seleciona apenas ``app_package`` e ``timestamp`` (e ``uid``, mantido
    para uso futuro em avaliação). A coluna ``review`` é descartada.

    Parameters
    ----------
    path : str
        Caminho para o arquivo. Extensão ``.parquet`` usa ``pl.scan_parquet``;
        ``.csv`` usa ``pl.scan_csv``. Outras extensões -> ``ValueError``.
    timestamp_unit : {"s", "ms", "us", None}, default "s"
        Unidade do timestamp quando ele é epoch numérico. Use ``None`` quando
        a coluna já é do tipo datetime.
    timezone : str, default "UTC"
        Fuso horário usado antes de truncar para o dia.

    Returns
    -------
    pl.LazyFrame
        Colunas: ``uid``, ``app_package`` (Utf8) e ``date`` (pl.Date).

    Notes
    -----
    Nenhum dado é materializado aqui; o plano é executado apenas no
    ``collect`` de ``build_interaction_data``.
    """
```

Implementação esperada: `pl.from_epoch(col, time_unit=...)` quando numérico, `.dt.convert_time_zone(timezone)`, `.dt.date()`.

#### `build_interaction_data`

```python
def build_interaction_data(lf: pl.LazyFrame) -> InteractionData:
    """
    Constrói a matriz de interações densa ``M`` (dias x apps) a partir das
    interações em formato longo.

    Passos
    ------
    1. Obter ``start_date = min(date)`` e ``end_date = max(date)``.
    2. Calcular ``day_idx = (date - start_date).dt.total_days()`` (Int32).
    3. Calcular ``app_idx`` como rank denso alfabético de ``app_package``
       menos 1 (Int32), e ``apps`` como os ``app_package`` únicos ordenados.
    4. ``group_by(["day_idx", "app_idx"]).agg(pl.len().alias("count"))``
       e ``collect`` (usar o engine streaming se disponível).
    5. Alocar ``M = np.zeros((T, A), dtype=np.int32)`` com
       ``T = (end_date - start_date).days + 1``.
    6. Atribuir ``M[day_idx, app_idx] = count`` (pares são únicos;
       não usar ``np.add.at``).

    Parameters
    ----------
    lf : pl.LazyFrame
        Saída de ``load_interactions``.

    Returns
    -------
    InteractionData
        Matriz ``M`` e mapeamentos de índice.

    Raises
    ------
    ValueError
        Se o dataset estiver vazio.

    Notes
    -----
    Memória: ``T * A * 4`` bytes. Ex.: 3.000 dias x 50.000 apps ≈ 600 MB.
    Logar ``T``, ``A`` e o tamanho estimado antes de alocar.
    """
```

#### `build_prefix_sums`

```python
def build_prefix_sums(matrix: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de somas de prefixos ``P`` ao longo do tempo.

    ``P[0] = 0`` e ``P[k] = M[0] + ... + M[k-1]``. Com ela, o total de
    interações de qualquer janela ``[i, j)`` é ``P[j] - P[i]``, em O(A),
    para qualquer tamanho de janela.

    Parameters
    ----------
    matrix : np.ndarray
        Matriz ``M`` de forma ``(T, A)``.

    Returns
    -------
    np.ndarray
        Matriz ``P`` de forma ``(T + 1, A)`` e dtype ``int64``
        (int64 evita overflow nas somas acumuladas).

    Notes
    -----
    Implementar com ``np.cumsum(matrix, axis=0, dtype=np.int64)`` escrito
    em ``P[1:]`` de um array pré-alocado com zeros. Não usar loops.
    """
```

#### `build_accumulated_matrix` (opcional, apenas para inspeção)

```python
def build_accumulated_matrix(prefix: np.ndarray, w: int) -> np.ndarray:
    """
    Materializa a matriz de interações acumuladas para uma janela fixa ``w``.

    ``acc[t, a]`` = interações do app ``a`` nos dias ``[t - w, t - 1]``
    (janela truncada no dia 0). A linha ``t`` NÃO inclui o dia ``t``.

    Parameters
    ----------
    prefix : np.ndarray
        Matriz ``P`` de forma ``(T + 1, A)``.
    w : int
        Tamanho da janela em dias (``w >= 1``).

    Returns
    -------
    np.ndarray
        Matriz de forma ``(T + 1, A)``, dtype ``int64``. A linha ``T``
        corresponde ao dia seguinte ao último dia do dataset.

    Notes
    -----
    Vetorizado: ``acc = P - P[np.maximum(np.arange(T + 1) - w, 0)]``.
    A recomendação NÃO depende desta função; ela usa ``P`` diretamente.
    """
```

#### Conversões de tempo e black list

```python
def date_to_index(data: InteractionData, date: dt.date | dt.datetime | str) -> int:
    """
    Converte uma data no índice de dia ``t`` usado pelo recomendador.

    Parameters
    ----------
    data : InteractionData
    date : date, datetime ou str ISO ("YYYY-MM-DD")
        Datetimes são truncados para o dia.

    Returns
    -------
    int
        ``(date - data.start_date).days``. Pode ser >= ``T`` (datas após
        o fim do dataset são válidas). Negativo -> ``ValueError``.
    """


def blacklist_to_indices(data: InteractionData, black_list: Iterable[str] | None) -> np.ndarray:
    """
    Converte uma lista de ``app_package`` em índices de app.

    Apps desconhecidos (fora de ``data.apps``) são ignorados silenciosamente.

    Returns
    -------
    np.ndarray
        Índices únicos e ordenados, dtype ``int32``. Array vazio se
        ``black_list`` for ``None`` ou vazia.
    """
```

---

### 6.2 `ranking.py`: contagem na janela

```python
def window_bounds(t: int, w: int, n_days: int) -> tuple[int, int]:
    """
    Calcula os limites da janela ``[inicio, fim)`` em índices de linha de ``M``.

    ``inicio = clip(t - w, 0, n_days)`` e ``fim = clip(t, 0, n_days)``.

    Raises
    ------
    ValueError
        Se ``t < 0`` ou ``w < 1``.
    """


def window_counts(prefix: np.ndarray, t: int, w: int) -> np.ndarray:
    """
    Total de interações de cada app na janela ``[t - w, t - 1]``.

    Returns
    -------
    np.ndarray
        Vetor ``(A,)`` int64: ``P[fim] - P[inicio]``.
    """
```

---

### 6.3 `ranking.py`: regra de ranking e desempate

#### Regra (requisito)

Para um instante `t` e janela `w`, os apps **elegíveis** são aqueles com:

- total na janela **> 0**, e
- **não** presentes na black list.

Apps com total 0 **nunca** são recomendados e **não participam** do desempate.

A ordenação dos elegíveis é lexicográfica, do critério mais importante para o menos importante:

1. total na janela `[t-w, t-1]`, **decrescente**;
2. `M[t-1]` (contagem de ontem), decrescente;
3. `M[t-2]`, decrescente;
4. … e assim por diante até `M[t-w]` (ou até o dia 0, se a janela foi truncada);
5. chave aleatória reprodutível (desempate final).

Isso equivale ao desempate recursivo "se empatou, olhe o dia anterior", mas **deve ser implementado com `np.lexsort`**, sem recursão.

Dentro dos critérios 2 a 4, uma contagem diária igual a 0 é um valor normal de comparação (só o critério de elegibilidade usa o total).

#### Aleatoriedade (requisito de reprodutibilidade)

- O gerador é `np.random.default_rng([seed, t])` quando `seed` não é `None`, e `np.random.default_rng()` caso contrário.
- A chave aleatória é **uma permutação de todos os `A` apps**: `prioridade = rng.permutation(A)`, indexada depois pelos candidatos.
- Motivo: a ordem aleatória entre dois apps empatados depende apenas de `(seed, t)`, e não de quais outros apps estão na black list ou entre os candidatos. Isso garante que a recomendação individual e a em lote produzam exatamente o mesmo resultado.

#### Otimização com `argpartition` (requisito)

Não ordenar todos os `A` apps. Procedimento:

1. Calcular `totais` e a máscara de elegíveis.
2. Se o número de elegíveis `≤ k`, os candidatos são todos os elegíveis.
3. Senão, usar `np.argpartition` para encontrar `v`, o k-ésimo maior total entre os elegíveis. Os candidatos são **todos os elegíveis com `total >= v`** (inclui todo o grupo empatado em `v`, podendo ter mais de `k` itens).
4. Aplicar o `lexsort` somente nos candidatos e cortar os `k` primeiros.

Atenção: itens com total estritamente maior que `v` também podem estar empatados entre si, por isso **todos** os candidatos passam pelo `lexsort`, não só o grupo em `v`.

#### Montagem das chaves do `lexsort`

`np.lexsort` usa a **última** chave como primária. Com `cand` = índices dos candidatos, `inicio, fim` = limites da janela:

```
janela = M[inicio:fim, cand]          # forma (fim - inicio, k_cand); linha 0 = dia mais antigo
chaves = [
    prioridade[cand],                 # menos significativa: aleatório
    -janela[0], -janela[1], ..., -janela[-1],   # t-w ... t-1 (t-1 é mais significativo que t-2)
    -totais[cand],                    # mais significativa: total
]
ordem = cand[np.lexsort(chaves)]
```

Usar `np.vstack` para montar as chaves em um array 2D. Negar valores exige dtype com sinal (int32/int64): converter antes de negar.

#### `rank_items`

```python
def rank_items(
    matrix: np.ndarray,
    prefix: np.ndarray,
    t: int,
    w: int,
    k: int,
    seed: int | None = None,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """
    Retorna os ``k`` melhores apps no instante ``t`` segundo a regra de
    popularidade com desempate lexicográfico.

    Parameters
    ----------
    matrix : np.ndarray
        Matriz de interações ``M`` ``(T, A)``, usada no desempate diário.
    prefix : np.ndarray
        Somas de prefixos ``P`` ``(T + 1, A)``, usadas nos totais da janela.
    t : int
        Índice do dia da recomendação. O dia ``t`` não é contado.
    w : int
        Tamanho da janela em dias (``w >= 1``).
    k : int
        Número máximo de apps a retornar (``k >= 0``).
    seed : int or None, default None
        Semente do desempate aleatório final. Mesmo ``(seed, t)`` produz
        sempre o mesmo resultado.
    exclude : np.ndarray or None, default None
        Índices de apps a excluir (ex.: black list já convertida).

    Returns
    -------
    np.ndarray
        Índices de apps (int32) em ordem de recomendação, de tamanho
        ``min(k, nº de elegíveis)``. Pode ser vazio.

    Notes
    -----
    Complexidade: O(A) para totais e ``argpartition`` mais
    O(w · c · log c) para o ``lexsort``, onde ``c`` é o número de
    candidatos (normalmente pequeno).
    """
```

#### `recommend` (interface principal)

```python
def recommend(
    data: InteractionData,
    prefix: np.ndarray,
    n: int,
    t: int | dt.date | str,
    w: int,
    black_list: Iterable[str] | None = None,
    seed: int | None = None,
) -> list[str]:
    """
    Recomenda os ``n`` apps mais populares na janela ``[t - w, t - 1]``.

    Parameters
    ----------
    data : InteractionData
        Matriz de interações original e mapeamentos.
    prefix : np.ndarray
        Somas de prefixos de ``data.matrix`` (``build_prefix_sums``).
    n : int
        Número de apps a recomendar (``n >= 1``).
    t : int, date ou str
        Instante da recomendação. Inteiro é tratado como índice de dia;
        date/str é convertido com ``date_to_index``.
    w : int
        Tamanho da janela em dias.
    black_list : iterable of str, optional
        ``app_package`` que não podem ser recomendados. Apps
        desconhecidos são ignorados.
    seed : int, optional
        Semente do desempate aleatório.

    Returns
    -------
    list[str]
        ``app_package`` recomendados, do melhor para o pior. Pode ter menos
        de ``n`` itens se não houver apps elegíveis suficientes
        (total > 0 e fora da black list), e é vazia se a janela for vazia.

    Examples
    --------
    >>> recommend(data, P, n=10, t="2024-03-15", w=7, black_list=["com.foo"], seed=42)
    ['com.bar', 'com.baz', ...]
    """
```

---

### 6.4 `batch.py`: avaliação em lote

Cenário: muitas consultas `(t_q, black_list_q)`, uma por usuário e instante. Como o ranking depende só de `t`, ele é calculado **uma vez por `t` único** e depois filtrado por consulta.

#### Formato das black lists em lote (CSR)

- `bl_indptr`: `(Q + 1,)` int64, onde `bl_indptr[0] = 0`.
- `bl_indices`: índices de apps; os da consulta `q` estão em `bl_indices[bl_indptr[q]:bl_indptr[q+1]]`, **ordenados** dentro de cada consulta.

#### `build_blacklist_csr`

```python
def build_blacklist_csr(
    data: InteractionData,
    black_lists: Sequence[Iterable[str] | None],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converte uma lista de black lists (uma por consulta) para o formato CSR.

    Returns
    -------
    bl_indptr : np.ndarray
        ``(Q + 1,)`` int64.
    bl_indices : np.ndarray
        int32, ordenado dentro de cada consulta, sem duplicatas.
        Apps desconhecidos são ignorados.
    """
```

#### `_filter_rankings_numba` (kernel Numba)

```python
@njit(parallel=True, cache=True)
def _filter_rankings_numba(
    rankings: np.ndarray,
    query_rank_row: np.ndarray,
    bl_indptr: np.ndarray,
    bl_indices: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Para cada consulta, percorre o ranking do seu ``t`` e coleta os
    primeiros ``n`` apps que não estão na black list.

    Parameters
    ----------
    rankings : np.ndarray
        ``(U, K)`` int32. Linha ``u`` é o ranking do u-ésimo ``t`` único,
        preenchido com ``-1`` após o fim.
    query_rank_row : np.ndarray
        ``(Q,)`` int64. Linha de ``rankings`` usada pela consulta ``q``.
    bl_indptr, bl_indices : np.ndarray
        Black lists em CSR (índices ordenados por consulta).
    n : int
        Número de recomendações por consulta.

    Returns
    -------
    np.ndarray
        ``(Q, n)`` int32 com índices de apps; posições sem recomendação
        ficam com ``-1``.

    Notes
    -----
    Usar ``prange`` sobre as consultas. A verificação de pertinência na
    black list usa ``np.searchsorted`` na fatia ordenada da consulta.
    Parar ao encontrar ``-1`` no ranking ou ao completar ``n`` itens.
    """
```

#### `recommend_batch`

```python
def recommend_batch(
    data: InteractionData,
    prefix: np.ndarray,
    n: int,
    ts: np.ndarray,
    w: int,
    bl_indptr: np.ndarray,
    bl_indices: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Recomendação em lote para ``Q`` consultas.

    Passos
    ------
    1. ``t_unicos, query_rank_row = np.unique(ts, return_inverse=True)``.
    2. ``K = n + max_q(len(black_list_q))``: tamanho suficiente para
       garantir ``n`` itens após remover qualquer black list.
    3. Para cada ``t`` único (loop Python aceitável aqui), chamar
       ``rank_items(M, P, t, w, K, seed, exclude=None)`` e gravar em
       ``rankings[u]`` (preenchido com ``-1``).
    4. Chamar ``_filter_rankings_numba``.

    Parameters
    ----------
    ts : np.ndarray
        ``(Q,)`` índices de dia de cada consulta.
    (demais parâmetros como em ``recommend`` e ``_filter_rankings_numba``)

    Returns
    -------
    np.ndarray
        ``(Q, n)`` int32 com índices de apps, ``-1`` onde não há item.
        Converter para strings com ``data.apps[idx]`` quando necessário.

    Notes
    -----
    Deve produzir exatamente o mesmo resultado que chamar ``recommend``
    consulta a consulta com a mesma ``seed`` (garantido pela chave
    aleatória por permutação de todos os apps, seção 6.3).
    """
```

---

### 6.5 `debug.py` (opcional)

```python
def matrix_to_polars(
    data: InteractionData,
    matrix: np.ndarray | None = None,
    day_range: tuple[int, int] | None = None,
    apps: Sequence[str] | None = None,
) -> pl.DataFrame:
    """
    Converte um recorte da matriz (``M`` ou acumulada) em ``pl.DataFrame``
    indexado por data, com uma coluna por app, apenas para inspeção visual.

    Não usar em produção: com muitos apps a tabela larga é custosa.
    Exigir ``day_range`` ou ``apps`` quando ``A > 1000``.
    """
```

---

## 7. Testes obrigatórios (pytest)

### Exemplo de referência

Dias `d0..d3`, apps `A, B, C, D` (ordem alfabética = índices 0..3), `w = 3`:

| dia | A | B | C | D |
|-----|---|---|---|---|
| d0  | 1 | 2 | 0 | 0 |
| d1  | 1 | 0 | 1 | 0 |
| d2  | 1 | 1 | 2 | 0 |
| d3  | 5 | 5 | 5 | 5 |

Casos esperados:

1. `t = 3, w = 3, n = 5` → `["C", "A", "B"]`.
   Totais em d0..d2: A=3, B=3, C=3, D=0. D é excluído (total 0). Em d2 (ontem): C=2 vence. A e B empatam em d2 (1 e 1); em d1, A=1 > B=0, então A vem antes de B. A linha d3 não é contada.
2. Igual ao caso 1 com `black_list=["C"]` → `["A", "B"]`.
3. `t = 1, w = 3` (janela truncada, só d0) → `["B", "A"]`.
4. `t = 0` → `[]`.
5. `n = 1`, caso 1 → `["C"]` (o `argpartition` não pode perder o grupo empatado).
6. Empate total: adicionar app `E` com contagens idênticas às de A (1, 1, 1, 5). Com a mesma `seed`, o resultado é idêntico entre chamadas; A e E aparecem em posições adjacentes; com seeds diferentes, a ordem entre A e E pode variar.
7. `t = 4` (dia seguinte ao último) com `w = 1` → `["A", "B", "C", "D"]` em ordem definida só pela seed (todos com 5; nenhum desempate diário além de d3).

### Outros testes

- `build_prefix_sums`: `P[j] - P[i]` igual a `M[i:j].sum(axis=0)` para pares aleatórios `(i, j)`.
- `build_interaction_data`: dataset com um dia sem interações no meio gera uma linha de zeros (calendário contínuo); a soma total de `M` é igual ao número de linhas do dataset.
- `build_accumulated_matrix`: comparar com implementação ingênua com loops em matriz pequena.
- `rank_items`: comparar com uma implementação de referência ingênua (ordenação Python com `sorted` e chave em tupla) em matrizes aleatórias pequenas, para várias seeds, `t`, `w` e `k`.
- `recommend_batch`: resultado idêntico a `recommend` consulta a consulta.
- Validação: `t < 0`, `w < 1`, `n < 1` levantam `ValueError`.

---

## 8. Resumo das decisões de design

- A matriz de interação é um `np.ndarray` denso `(T, A)` `int32`, não um `pl.DataFrame` largo.
- A matriz acumulada é substituída por somas de prefixos `P`, válidas para qualquer `w`.
- O desempate recursivo é implementado como `np.lexsort` sobre `(total, t-1, ..., t-w, aleatório)`, aplicado só aos candidatos selecionados via `argpartition`.
- A aleatoriedade usa `default_rng([seed, t])` com permutação de todos os apps, garantindo reprodutibilidade e consistência entre modos individual e em lote.
- Numba é usado somente no filtro de black list da avaliação em lote.