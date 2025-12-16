"""
RedesNeuraisAvancadas - Utilitários
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from typing import Tuple


def gerar_dados_regressao(n_amostras: int = 1000, n_features: int = 10, ruido: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Gera dados para teste de regressão."""
    X = np.random.randn(n_amostras, n_features)
    coeficientes = np.random.randn(n_features, 1)
    y = X @ coeficientes + ruido * np.random.randn(n_amostras, 1)
    return X, y


def gerar_dados_classificacao(n_amostras: int = 1000, n_classes: int = 3, n_features: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Gera dados para teste de classificação."""
    X = []
    y = []
    
    for classe in range(n_classes):
        centro = np.random.randn(n_features) * 2
        amostras = np.random.randn(n_amostras // n_classes, n_features) + centro
        X.append(amostras)
        y.extend([classe] * (n_amostras // n_classes))
    
    X = np.vstack(X)
    y = np.array(y)
    
    # One-hot encoding
    y_onehot = np.zeros((len(y), n_classes))
    y_onehot[np.arange(len(y)), y] = 1
    
    return X, y_onehot


def gerar_sequencia(n_amostras: int = 100, seq_len: int = 20, n_features: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Gera dados sequenciais para teste de RNN/LSTM."""
    X = np.random.randn(n_amostras, seq_len, n_features)
    # Target é uma função simples da sequência
    y = np.mean(X, axis=(1, 2), keepdims=True)
    return X, y


def gerar_imagens(n_amostras: int = 100, altura: int = 28, largura: int = 28, canais: int = 1) -> np.ndarray:
    """Gera imagens aleatórias para teste."""
    return np.random.rand(n_amostras, canais, altura, largura)


def normalizar(X: np.ndarray, metodo: str = 'minmax') -> Tuple[np.ndarray, dict]:
    """Normaliza dados."""
    if metodo == 'minmax':
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}
    else:  # zscore
        media = X.mean(axis=0)
        std = X.std(axis=0)
        X_norm = (X - media) / (std + 1e-8)
        params = {'media': media, 'std': std}
    
    return X_norm, params


def dividir_dados(X: np.ndarray, y: np.ndarray, prop_treino: float = 0.8) -> Tuple:
    """Divide dados em treino e teste."""
    n = X.shape[0]
    indices = np.random.permutation(n)
    corte = int(n * prop_treino)
    
    idx_treino = indices[:corte]
    idx_teste = indices[corte:]
    
    return X[idx_treino], X[idx_teste], y[idx_treino], y[idx_teste]


def acuracia(y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> float:
    """Calcula acurácia para classificação."""
    if y_verdadeiro.ndim > 1:
        y_verdadeiro = np.argmax(y_verdadeiro, axis=1)
    if y_previsto.ndim > 1:
        y_previsto = np.argmax(y_previsto, axis=1)
    return np.mean(y_verdadeiro == y_previsto)


def r2_score(y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> float:
    """Calcula coeficiente de determinação R²."""
    ss_res = np.sum((y_verdadeiro - y_previsto) ** 2)
    ss_tot = np.sum((y_verdadeiro - np.mean(y_verdadeiro)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)
