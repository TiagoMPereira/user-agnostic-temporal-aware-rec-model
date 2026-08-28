"""Script de avaliacao de recomendacoes (Card 10).

Le o parquet de ground truth de teste e o parquet de predicoes,
executa o pipeline de metricas (metrics/) -- join, rank e metricas por
interacao -- e salva as estatisticas resumo em um arquivo Excel.

Com o split leave-one-out (Card 5), cada usuario contribui com
exatamente 1 interacao de teste, entao as estatisticas sao calculadas
direto sobre as metricas por interacao, sem estagio de agregacao por
usuario.
"""

from metrics import compute_rank, compute_summary, save_to_excel

GROUND_TRUTH_PATH = "data/predictions/test_ground_truth.parquet"
PREDICTIONS_PATH = "data/predictions/random.parquet"
OUTPUT_PATH = "metrics_output.xlsx"


def main():
    print(f"Lendo {GROUND_TRUTH_PATH} e {PREDICTIONS_PATH}, calculando rank e metricas por interacao...")
    interaction_metrics = compute_rank(GROUND_TRUTH_PATH, PREDICTIONS_PATH)

    print("Calculando estatisticas resumo...")
    summary = compute_summary(interaction_metrics)

    print(f"Salvando {OUTPUT_PATH}...")
    save_to_excel(summary, OUTPUT_PATH)

    print("Concluido!")


if __name__ == "__main__":
    main()
