"""
RedesNeuraisAvancadas - Utilitários
Autor: Luiz Tiago Wilcke
"""

from .dados import (
    gerar_dados_regressao,
    gerar_dados_classificacao,
    gerar_sequencia,
    gerar_imagens,
    normalizar,
    dividir_dados,
    acuracia,
    r2_score
)

from .visualizacao import (
    plotar_historico_treino,
    plotar_gan_historico,
    plotar_som,
    plotar_grafo,
    imprimir_arquitetura
)
