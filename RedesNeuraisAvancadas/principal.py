#!/usr/bin/env python3
"""
RedesNeuraisAvancadas - Demonstração Principal
Autor: Luiz Tiago Wilcke

Este script demonstra o uso das 20 redes neurais implementadas na biblioteca.
"""

import numpy as np
import sys
import os

# Adiciona o diretório ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from redes import (
    # Arquiteturas básicas
    Perceptron, MLP, CNN, CamadaLSTM, CamadaGRU,
    # Generativas
    Autoencoder, VAE, GAN,
    # Avançadas
    Transformer, ResNet,
    # Especiais
    RedeHopfield, MapaKohonen, RedeRBF, MaquinaBoltzmann, RedeEchoState,
    # Grafos
    CapsuleNetwork, RedeGrafoNeuronal,
    # Otimizadores
    Adam, SGD, RMSprop,
    # Perdas
    ErroQuadraticoMedio, EntropiaCruzadaCategorica,
    # Utilitários
    listar_redes, CamadaDensa, CamadaReLU, CamadaSoftmax
)

from utils import (
    gerar_dados_regressao, gerar_dados_classificacao,
    gerar_sequencia, dividir_dados, acuracia, r2_score,
    imprimir_arquitetura
)


def demo_perceptron():
    """Demonstração do Perceptron."""
    print("\n" + "="*60)
    print("  DEMO: Perceptron")
    print("="*60)
    
    # Dados de treinamento (porta AND)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])
    
    perceptron = Perceptron(entrada_dim=2, funcao_ativacao='step')
    perceptron.treinar_perceptron(X, y, taxa=0.1, epocas=100)
    
    print("Porta AND treinada:")
    for xi in X:
        pred = perceptron.frente(xi.reshape(1, -1))
        print(f"  {xi} -> {int(pred[0, 0])}")


def demo_mlp():
    """Demonstração do MLP."""
    print("\n" + "="*60)
    print("  DEMO: Multi-Layer Perceptron (MLP)")
    print("="*60)
    
    # Gera dados de classificação
    X, y = gerar_dados_classificacao(n_amostras=500, n_classes=3, n_features=10)
    X_treino, X_teste, y_treino, y_teste = dividir_dados(X, y, prop_treino=0.8)
    
    # Cria modelo
    modelo = MLP.criar_classificador(entrada=10, classes=3, ocultas=[32, 16])
    modelo.compilar(
        otimizador=Adam(taxa_aprendizado=0.01),
        funcao_perda=EntropiaCruzadaCategorica()
    )
    
    imprimir_arquitetura(modelo)
    
    # Treina
    modelo.treinar(X_treino, y_treino, epocas=50, tamanho_lote=32, verbose=True)
    
    # Avalia
    pred = modelo.prever(X_teste)
    acc = acuracia(y_teste, pred)
    print(f"\nAcurácia no teste: {acc:.4f}")


def demo_lstm():
    """Demonstração da LSTM."""
    print("\n" + "="*60)
    print("  DEMO: LSTM (Long Short-Term Memory)")
    print("="*60)
    
    # Gera sequências
    X, y = gerar_sequencia(n_amostras=200, seq_len=10, n_features=5)
    X_treino, X_teste, y_treino, y_teste = dividir_dados(X, y, prop_treino=0.8)
    
    print(f"Formato entrada: {X_treino.shape}")
    print(f"Formato saída: {y_treino.shape}")
    
    # Cria camada LSTM
    lstm = CamadaLSTM(unidades=32, retornar_sequencias=False)
    
    # Forward pass
    saida = lstm.frente(X_treino[:5])
    print(f"Saída LSTM shape: {saida.shape}")
    print("LSTM executada com sucesso!")


def demo_gan():
    """Demonstração da GAN."""
    print("\n" + "="*60)
    print("  DEMO: GAN (Generative Adversarial Network)")
    print("="*60)
    
    # Dados simples (distribuição gaussiana)
    dados_reais = np.random.randn(500, 10) * 2 + 5
    
    # Cria GAN
    gan = GAN(dim_latente=20, dim_dados=10)
    
    print("Treinando GAN...")
    gan.treinar(dados_reais, epocas=50, tamanho_lote=32, verbose=True)
    
    # Gera amostras
    amostras = gan.gerar(5)
    print(f"\nAmostras geradas (média esperada ~5):")
    print(f"  Média: {np.mean(amostras):.4f}")
    print(f"  Std: {np.std(amostras):.4f}")


def demo_hopfield():
    """Demonstração da Rede de Hopfield."""
    print("\n" + "="*60)
    print("  DEMO: Rede de Hopfield (Memória Associativa)")
    print("="*60)
    
    # Padrões para memorizar (bipolares: -1 e +1)
    padroes = np.array([
        [1, 1, 1, -1, -1, -1, 1, 1],
        [-1, -1, 1, 1, 1, 1, -1, -1],
        [1, -1, 1, -1, 1, -1, 1, -1]
    ])
    
    rede = RedeHopfield(tamanho=8)
    rede.treinar(padroes)
    
    print("Padrões memorizados:")
    for i, p in enumerate(padroes):
        print(f"  Padrão {i+1}: {p}")
    
    # Testa recuperação com ruído
    teste = np.array([1, 1, 1, 1, -1, -1, 1, 1])  # Versão corrompida do padrão 1
    recuperado = rede.recuperar(teste)
    
    print(f"\nTeste de recuperação:")
    print(f"  Entrada (com ruído): {teste}")
    print(f"  Recuperado: {recuperado}")
    print(f"  Original: {padroes[0]}")


def demo_kohonen():
    """Demonstração do Mapa de Kohonen."""
    print("\n" + "="*60)
    print("  DEMO: Mapa Auto-Organizável de Kohonen (SOM)")
    print("="*60)
    
    # Dados de cores RGB
    cores = np.array([
        [1, 0, 0],    # Vermelho
        [0, 1, 0],    # Verde
        [0, 0, 1],    # Azul
        [1, 1, 0],    # Amarelo
        [1, 0, 1],    # Magenta
        [0, 1, 1],    # Ciano
        [0.5, 0.5, 0.5],  # Cinza
        [1, 0.5, 0],  # Laranja
    ])
    
    som = MapaKohonen(largura=5, altura=5, dim_entrada=3)
    
    print("Treinando SOM com cores RGB...")
    som.treinar(cores, epocas=100)
    
    # Encontra BMU para cada cor
    print("\nBest Matching Units:")
    nomes = ['Vermelho', 'Verde', 'Azul', 'Amarelo', 'Magenta', 'Ciano', 'Cinza', 'Laranja']
    for nome, cor in zip(nomes, cores):
        bmu = som.encontrar_bmu(cor)
        print(f"  {nome}: posição {bmu}")


def demo_transformer():
    """Demonstração do Transformer."""
    print("\n" + "="*60)
    print("  DEMO: Transformer")
    print("="*60)
    
    # Sequência de entrada
    batch_size = 4
    seq_len = 10
    dim_modelo = 64
    
    X = np.random.randn(batch_size, seq_len, dim_modelo)
    
    transformer = Transformer(dim_modelo=dim_modelo, num_camadas=2, num_cabecas=4)
    
    imprimir_arquitetura(transformer)
    
    saida = transformer.frente(X)
    print(f"Entrada: {X.shape}")
    print(f"Saída: {saida.shape}")
    print("Transformer executado com sucesso!")


def demo_gnn():
    """Demonstração da Graph Neural Network."""
    print("\n" + "="*60)
    print("  DEMO: Graph Neural Network (GNN)")
    print("="*60)
    
    # Grafo simples (5 nós)
    n_nos = 5
    
    # Matriz de adjacência
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=float)
    
    # Features dos nós
    X = np.random.randn(n_nos, 8)
    
    gnn = RedeGrafoNeuronal(dim_entrada=8, dims_ocultas=[16, 16], dim_saida=3)
    
    saida = gnn.frente(X, A)
    print(f"Número de nós: {n_nos}")
    print(f"Features por nó: 8")
    print(f"Classes de saída: 3")
    print(f"\nPrevisões por nó (softmax):")
    probs = gnn.prever_nos(X, A)
    for i in range(n_nos):
        classe = np.argmax(probs[i])
        print(f"  Nó {i}: Classe {classe} (prob: {probs[i, classe]:.4f})")


def demo_boltzmann():
    """Demonstração da Máquina de Boltzmann."""
    print("\n" + "="*60)
    print("  DEMO: Máquina de Boltzmann Restrita (RBM)")
    print("="*60)
    
    # Dados binários
    dados = (np.random.rand(100, 10) > 0.5).astype(float)
    
    rbm = MaquinaBoltzmann(n_visivel=10, n_oculto=5)
    
    print("Treinando RBM...")
    rbm.treinar(dados, epocas=50, taxa=0.1, k=1)
    
    # Testa reconstrução
    amostra = dados[0:1]
    h = rbm.prop_cima(amostra)
    reconstrucao = rbm.prop_baixo(h)
    
    print(f"\nOriginal: {amostra[0]}")
    print(f"Features ocultas: {h[0].round(2)}")
    print(f"Reconstrução: {reconstrucao[0].round(2)}")


def demo_esn():
    """Demonstração da Echo State Network."""
    print("\n" + "="*60)
    print("  DEMO: Echo State Network (ESN)")
    print("="*60)
    
    # Série temporal simples (seno)
    t = np.linspace(0, 10 * np.pi, 500)
    X = np.sin(t).reshape(-1, 1)
    y = np.sin(t + 0.1).reshape(-1, 1)  # Prever próximo valor
    
    esn = RedeEchoState(dim_entrada=1, dim_reservatorio=100, dim_saida=1)
    
    # Treina
    esn.treinar(X[:400], y[:400])
    
    # Testa
    pred = esn.prever(X[400:])
    
    erro = np.mean((y[400:] - pred) ** 2)
    print(f"Erro quadrático médio: {erro:.6f}")
    print("ESN treinada com sucesso!")


def main():
    """Função principal - executa todas as demos."""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#    REDES NEURAIS AVANÇADAS - DEMONSTRAÇÃO COMPLETA    #")
    print("#" + " "*58 + "#")
    print("#    Autor: Luiz Tiago Wilcke                           #")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    # Lista todas as redes
    listar_redes()
    
    # Executa demos
    demos = [
        demo_perceptron,
        demo_mlp,
        demo_lstm,
        demo_hopfield,
        demo_kohonen,
        demo_transformer,
        demo_gnn,
        demo_boltzmann,
        demo_esn,
        demo_gan,  # Por último pois é mais demorado
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n  ERRO na demo: {e}")
    
    print("\n" + "#"*60)
    print("#    DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!                #")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
