"""
RedesNeuraisAvancadas - Biblioteca de Redes Neurais
Autor: Luiz Tiago Wilcke
Módulo: Capsule Network e Graph Neural Network
"""

import numpy as np
from .base import Camada, RedeNeural, CamadaDensa
from .ativacoes import CamadaReLU


# ==============================================
# CAPSULE NETWORK
# ==============================================

class CamadaCapsule(Camada):
    """Primary Capsule Layer."""
    
    def __init__(self, num_capsules: int, dim_capsule: int, nome: str = None):
        super().__init__(nome or "CapsuleLayer")
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.treinavel = True
        self.inicializado = False
    
    def _inicializar(self, dim_entrada: int):
        escala = np.sqrt(2 / dim_entrada)
        self.parametros['W'] = np.random.randn(
            dim_entrada, self.num_capsules * self.dim_capsule
        ) * escala
        self.inicializado = True
    
    def _squash(self, x: np.ndarray) -> np.ndarray:
        """Função squash para normalização de cápsulas."""
        norma_sq = np.sum(x ** 2, axis=-1, keepdims=True)
        norma = np.sqrt(norma_sq + 1e-8)
        return (norma_sq / (1 + norma_sq)) * (x / norma)
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        if not self.inicializado:
            self._inicializar(entrada.shape[-1])
        
        self.entrada = entrada
        batch = entrada.shape[0]
        
        u = entrada @ self.parametros['W']
        u = u.reshape(batch, self.num_capsules, self.dim_capsule)
        self.saida = self._squash(u)
        
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        self.gradientes['W'] = np.zeros_like(self.parametros['W'])
        self.gradiente_entrada = np.zeros_like(self.entrada)
        return self.gradiente_entrada


class CapsuleNetwork(RedeNeural):
    """
    Capsule Network.
    Autor: Luiz Tiago Wilcke
    """
    
    def __init__(self, dim_entrada: int, num_classes: int, dim_capsule: int = 16):
        super().__init__("CapsuleNetwork")
        self.adicionar(CamadaDensa(dim_entrada, 256))
        self.adicionar(CamadaReLU())
        self.adicionar(CamadaCapsule(num_classes, dim_capsule))


# ==============================================
# GRAPH NEURAL NETWORK
# ==============================================

class CamadaGrafo(Camada):
    """Camada de convolução em grafo (GCN)."""
    
    def __init__(self, dim_entrada: int, dim_saida: int, nome: str = None):
        super().__init__(nome or "GraphConv")
        self.dim_entrada = dim_entrada
        self.dim_saida = dim_saida
        self.treinavel = True
        
        escala = np.sqrt(2 / (dim_entrada + dim_saida))
        self.parametros['W'] = np.random.randn(dim_entrada, dim_saida) * escala
        self.parametros['b'] = np.zeros((1, dim_saida))
    
    def frente(self, entrada: tuple, treinamento: bool = True) -> np.ndarray:
        """
        entrada: (X, A_norm) onde:
            X: features dos nós (N, dim_entrada)
            A_norm: matriz de adjacência normalizada (N, N)
        """
        self.X, self.A_norm = entrada
        
        # Agregação de vizinhos
        agregado = self.A_norm @ self.X
        
        # Transformação
        self.saida = agregado @ self.parametros['W'] + self.parametros['b']
        
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        self.gradientes['W'] = (self.A_norm @ self.X).T @ gradiente_saida
        self.gradientes['b'] = np.sum(gradiente_saida, axis=0, keepdims=True)
        self.gradiente_entrada = self.A_norm.T @ gradiente_saida @ self.parametros['W'].T
        return self.gradiente_entrada


class RedeGrafoNeuronal:
    """
    Graph Neural Network (GNN).
    Autor: Luiz Tiago Wilcke
    """
    
    def __init__(self, dim_entrada: int, dims_ocultas: list, dim_saida: int):
        self.camadas = []
        dims = [dim_entrada] + dims_ocultas + [dim_saida]
        
        for i in range(len(dims) - 1):
            self.camadas.append(CamadaGrafo(dims[i], dims[i+1]))
    
    @staticmethod
    def normalizar_adjacencia(A: np.ndarray) -> np.ndarray:
        """Normalização simétrica: D^(-1/2) * A * D^(-1/2)."""
        A = A + np.eye(A.shape[0])  # Auto-loops
        D = np.diag(np.sum(A, axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
        return D_inv_sqrt @ A @ D_inv_sqrt
    
    def frente(self, X: np.ndarray, A: np.ndarray, treinamento: bool = True) -> np.ndarray:
        """Propagação para frente."""
        A_norm = self.normalizar_adjacencia(A)
        
        h = X
        for i, camada in enumerate(self.camadas):
            h = camada.frente((h, A_norm), treinamento)
            if i < len(self.camadas) - 1:
                h = np.maximum(0, h)  # ReLU
        
        return h
    
    def prever_nos(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Classificação de nós."""
        logits = self.frente(X, A, treinamento=False)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


# ==============================================
# GRAPH ATTENTION NETWORK
# ==============================================

class CamadaAtencaoGrafo(Camada):
    """Graph Attention Layer."""
    
    def __init__(self, dim_entrada: int, dim_saida: int, num_cabecas: int = 4, nome: str = None):
        super().__init__(nome or "GraphAttention")
        self.dim_entrada = dim_entrada
        self.dim_saida = dim_saida
        self.num_cabecas = num_cabecas
        self.dim_cabeca = dim_saida // num_cabecas
        self.treinavel = True
        
        escala = np.sqrt(2 / dim_entrada)
        self.parametros['W'] = np.random.randn(dim_entrada, dim_saida) * escala
        self.parametros['a'] = np.random.randn(2 * self.dim_cabeca, num_cabecas) * escala
    
    def frente(self, entrada: tuple, treinamento: bool = True) -> np.ndarray:
        self.X, self.A = entrada
        N = self.X.shape[0]
        
        # Transformação linear
        Wh = self.X @ self.parametros['W']
        Wh = Wh.reshape(N, self.num_cabecas, self.dim_cabeca)
        
        # Coeficientes de atenção
        e = np.zeros((N, N, self.num_cabecas))
        for i in range(N):
            for j in range(N):
                if self.A[i, j] > 0:
                    concat = np.concatenate([Wh[i], Wh[j]], axis=-1)
                    e[i, j] = np.sum(concat * self.parametros['a'].T, axis=-1)
        
        # Softmax sobre vizinhos
        e = np.where(self.A[:, :, np.newaxis] > 0, e, -1e9)
        alpha = np.exp(e - np.max(e, axis=1, keepdims=True))
        alpha = alpha / (np.sum(alpha, axis=1, keepdims=True) + 1e-8)
        
        # Agregação
        h_prime = np.zeros((N, self.num_cabecas, self.dim_cabeca))
        for h in range(self.num_cabecas):
            for i in range(N):
                h_prime[i, h] = np.sum(alpha[i, :, h:h+1] * Wh[:, h, :], axis=0)
        
        self.saida = h_prime.reshape(N, self.dim_saida)
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        self.gradientes['W'] = np.zeros_like(self.parametros['W'])
        self.gradientes['a'] = np.zeros_like(self.parametros['a'])
        return np.zeros_like(self.X)
