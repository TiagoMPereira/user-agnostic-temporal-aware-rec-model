"""Script de avaliacao de recomendacoes (Card 10).

Le o parquet de ground truth de teste e o parquet de predicoes,
executa o pipeline de metricas (metrics/) -- join, rank, metricas por
interacao e agregacao macro em dois estagios -- e salva o resultado em
um arquivo Excel.

ATENCAO: os arquivos de teste/predicao tem ~3.5M linhas. Este script
processa o dataset inteiro e pode consumir bastante memoria e tempo --
nao deve ser executado em uma maquina fraca sem antes confirmar o
volume de dados disponivel.
"""

from metrics import aggregate_per_user, compute_rank, compute_summary, save_to_excel

GROUND_TRUTH_PATH = "data/predictions/test_ground_truth.parquet"
PREDICTIONS_PATH = "data/predictions/random.parquet"
OUTPUT_PATH = "metrics_output.xlsx"


def main():
    print(f"Lendo {GROUND_TRUTH_PATH} e {PREDICTIONS_PATH}, calculando rank e metricas por interacao...")
    interaction_metrics = compute_rank(GROUND_TRUTH_PATH, PREDICTIONS_PATH)

    print("Agregando metricas por usuario (Estagio 1 da macro-agregacao)...")
    per_user = aggregate_per_user(interaction_metrics)
    print(f"  {per_user.height} usuarios unicos no teste")

    print("Calculando estatisticas resumo (Estagio 2 da macro-agregacao)...")
    summary = compute_summary(per_user)

    print(f"Salvando {OUTPUT_PATH}...")
    save_to_excel(per_user, summary, OUTPUT_PATH)

    print("Concluido!")


if __name__ == "__main__":
    main()
