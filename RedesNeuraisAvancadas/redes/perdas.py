"""
RedesNeuraisAvancadas - Biblioteca de Redes Neurais
Autor: Luiz Tiago Wilcke
Módulo: Perdas - Funções de perda/custo
"""

import numpy as np
from abc import ABC, abstractmethod


class FuncaoPerda(ABC):
    """Classe abstrata base para funções de perda."""
    
    @abstractmethod
    def calcular(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> float:
        pass
    
    @abstractmethod
    def gradiente(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> np.ndarray:
        pass


class ErroQuadraticoMedio(FuncaoPerda):
    """
    Erro Quadrático Médio (MSE).
    L = (1/n) * Σ(y - ŷ)²
    """
    
    def calcular(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> float:
        return np.mean((y_verdadeiro - y_previsto) ** 2)
    
    def gradiente(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> np.ndarray:
        n = y_verdadeiro.shape[0]
        return -2 * (y_verdadeiro - y_previsto) / n


class ErroAbsolutoMedio(FuncaoPerda):
    """
    Erro Absoluto Médio (MAE).
    L = (1/n) * Σ|y - ŷ|
    """
    
    def calcular(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> float:
        return np.mean(np.abs(y_verdadeiro - y_previsto))
    
    def gradiente(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> np.ndarray:
        n = y_verdadeiro.shape[0]
        return -np.sign(y_verdadeiro - y_previsto) / n


class Huber(FuncaoPerda):
    """
    Perda de Huber (combinação de MSE e MAE).
    
    Args:
        delta: Threshold para troca entre MSE e MAE
    """
    
    def __init__(self, delta: float = 1.0):
        self.delta = delta
    
    def calcular(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> float:
        erro = y_verdadeiro - y_previsto
        abs_erro = np.abs(erro)
        
        quadratico = 0.5 * erro ** 2
        linear = self.delta * (abs_erro - 0.5 * self.delta)
        
        return np.mean(np.where(abs_erro <= self.delta, quadratico, linear))
    
    def gradiente(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> np.ndarray:
        n = y_verdadeiro.shape[0]
        erro = y_verdadeiro - y_previsto
        abs_erro = np.abs(erro)
        
        return -np.where(abs_erro <= self.delta, erro, self.delta * np.sign(erro)) / n


class EntropiaCruzadaBinaria(FuncaoPerda):
    """
    Entropia Cruzada Binária.
    L = -(1/n) * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
    """
    
    def __init__(self, epsilon: float = 1e-15):
        self.epsilon = epsilon
    
    def calcular(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> float:
        y_clip = np.clip(y_previsto, self.epsilon, 1 - self.epsilon)
        return -np.mean(
            y_verdadeiro * np.log(y_clip) +
            (1 - y_verdadeiro) * np.log(1 - y_clip)
        )
    
    def gradiente(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> np.ndarray:
        n = y_verdadeiro.shape[0]
        y_clip = np.clip(y_previsto, self.epsilon, 1 - self.epsilon)
        return -(y_verdadeiro / y_clip - (1 - y_verdadeiro) / (1 - y_clip)) / n


class EntropiaCruzadaCategorica(FuncaoPerda):
    """
    Entropia Cruzada Categórica (para classificação multi-classe).
    L = -(1/n) * Σ Σ y_ij * log(ŷ_ij)
    """
    
    def __init__(self, epsilon: float = 1e-15):
        self.epsilon = epsilon
    
    def calcular(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> float:
        y_clip = np.clip(y_previsto, self.epsilon, 1 - self.epsilon)
        return -np.mean(np.sum(y_verdadeiro * np.log(y_clip), axis=-1))
    
    def gradiente(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> np.ndarray:
        n = y_verdadeiro.shape[0]
        y_clip = np.clip(y_previsto, self.epsilon, 1 - self.epsilon)
        return -y_verdadeiro / y_clip / n


class EntropiaCruzadaSparse(FuncaoPerda):
    """
    Entropia Cruzada Sparse (quando y é índice, não one-hot).
    """
    
    def __init__(self, epsilon: float = 1e-15):
        self.epsilon = epsilon
    
    def calcular(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> float:
        n = y_verdadeiro.shape[0]
        y_clip = np.clip(y_previsto, self.epsilon, 1 - self.epsilon)
        return -np.mean(np.log(y_clip[np.arange(n), y_verdadeiro.astype(int).flatten()]))
    
    def gradiente(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> np.ndarray:
        n = y_verdadeiro.shape[0]
        y_clip = np.clip(y_previsto, self.epsilon, 1 - self.epsilon)
        grad = np.zeros_like(y_clip)
        indices = np.arange(n)
        y_indices = y_verdadeiro.astype(int).flatten()
        grad[indices, y_indices] = -1 / y_clip[indices, y_indices] / n
        return grad


class KLDivergencia(FuncaoPerda):
    """
    Divergência Kullback-Leibler.
    D_KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
    """
    
    def __init__(self, epsilon: float = 1e-15):
        self.epsilon = epsilon
    
    def calcular(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> float:
        p = np.clip(y_verdadeiro, self.epsilon, 1)
        q = np.clip(y_previsto, self.epsilon, 1)
        return np.mean(np.sum(p * np.log(p / q), axis=-1))
    
    def gradiente(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> np.ndarray:
        n = y_verdadeiro.shape[0]
        p = np.clip(y_verdadeiro, self.epsilon, 1)
        q = np.clip(y_previsto, self.epsilon, 1)
        return -p / q / n


class HingeLoss(FuncaoPerda):
    """
    Hinge Loss (para SVM).
    L = max(0, 1 - y * ŷ)
    """
    
    def calcular(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> float:
        return np.mean(np.maximum(0, 1 - y_verdadeiro * y_previsto))
    
    def gradiente(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> np.ndarray:
        n = y_verdadeiro.shape[0]
        return -np.where(1 - y_verdadeiro * y_previsto > 0, y_verdadeiro, 0) / n


class FocalLoss(FuncaoPerda):
    """
    Focal Loss (para dados desbalanceados).
    FL = -α * (1-p)^γ * log(p)
    
    Args:
        alfa: Peso balanceador
        gama: Fator de foco (γ >= 0)
    """
    
    def __init__(self, alfa: float = 0.25, gama: float = 2.0, epsilon: float = 1e-15):
        self.alfa = alfa
        self.gama = gama
        self.epsilon = epsilon
    
    def calcular(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> float:
        p = np.clip(y_previsto, self.epsilon, 1 - self.epsilon)
        pt = np.where(y_verdadeiro == 1, p, 1 - p)
        return -np.mean(self.alfa * ((1 - pt) ** self.gama) * np.log(pt))
    
    def gradiente(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> np.ndarray:
        n = y_verdadeiro.shape[0]
        p = np.clip(y_previsto, self.epsilon, 1 - self.epsilon)
        pt = np.where(y_verdadeiro == 1, p, 1 - p)
        
        # Derivada do focal loss
        fator = self.alfa * ((1 - pt) ** self.gama)
        grad_log = -1 / pt
        grad_fator = self.gama * ((1 - pt) ** (self.gama - 1))
        
        grad = fator * grad_log + self.alfa * grad_fator * np.log(pt)
        return np.where(y_verdadeiro == 1, grad, -grad) / n


class CosineSimilarityLoss(FuncaoPerda):
    """
    Perda baseada em Similaridade de Cosseno.
    L = 1 - cos(y, ŷ)
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
    
    def calcular(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> float:
        norma_y = np.linalg.norm(y_verdadeiro, axis=-1, keepdims=True)
        norma_p = np.linalg.norm(y_previsto, axis=-1, keepdims=True)
        
        similaridade = np.sum(y_verdadeiro * y_previsto, axis=-1, keepdims=True) / \
                       (norma_y * norma_p + self.epsilon)
        
        return np.mean(1 - similaridade)
    
    def gradiente(self, y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> np.ndarray:
        n = y_verdadeiro.shape[0]
        norma_y = np.linalg.norm(y_verdadeiro, axis=-1, keepdims=True)
        norma_p = np.linalg.norm(y_previsto, axis=-1, keepdims=True)
        
        prod_escalar = np.sum(y_verdadeiro * y_previsto, axis=-1, keepdims=True)
        denom = norma_y * norma_p + self.epsilon
        
        # d/dp [y·p / (||y||·||p||)]
        grad = -y_verdadeiro / denom + prod_escalar * y_previsto / (denom * norma_p ** 2 + self.epsilon)
        
        return grad / n


def obter_perda(nome: str, **kwargs) -> FuncaoPerda:
    """Retorna instância de função de perda pelo nome."""
    perdas = {
        'mse': ErroQuadraticoMedio,
        'erro_quadratico_medio': ErroQuadraticoMedio,
        'mae': ErroAbsolutoMedio,
        'erro_absoluto_medio': ErroAbsolutoMedio,
        'huber': Huber,
        'entropia_binaria': EntropiaCruzadaBinaria,
        'binary_crossentropy': EntropiaCruzadaBinaria,
        'entropia_categorica': EntropiaCruzadaCategorica,
        'categorical_crossentropy': EntropiaCruzadaCategorica,
        'entropia_sparse': EntropiaCruzadaSparse,
        'sparse_crossentropy': EntropiaCruzadaSparse,
        'kl': KLDivergencia,
        'kl_divergence': KLDivergencia,
        'hinge': HingeLoss,
        'focal': FocalLoss,
        'cosseno': CosineSimilarityLoss,
        'cosine': CosineSimilarityLoss
    }
    
    if nome.lower() not in perdas:
        raise ValueError(f"Função de perda '{nome}' não encontrada. Opções: {list(perdas.keys())}")
    
    return perdas[nome.lower()](**kwargs)
