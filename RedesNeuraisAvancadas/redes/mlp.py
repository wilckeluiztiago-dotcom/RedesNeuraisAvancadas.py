"""
RedesNeuraisAvancadas - Biblioteca de Redes Neurais
Autor: Luiz Tiago Wilcke
Módulo: MLP - Perceptron Multicamadas
"""

import numpy as np
from typing import List, Tuple
from .base import RedeNeural, Camada, CamadaDensa
from .ativacoes import CamadaReLU, CamadaSigmoide, CamadaSoftmax
from .otimizadores import Adam
from .perdas import ErroQuadraticoMedio


class Perceptron(Camada):
    """
    Perceptron simples (unidade neural básica).
    
    Args:
        entrada_dim: Dimensão da entrada
        funcao_ativacao: 'step', 'sigmoid' ou 'linear'
    """
    
    def __init__(self, entrada_dim: int, funcao_ativacao: str = 'step', nome: str = None):
        super().__init__(nome or "Perceptron")
        self.entrada_dim = entrada_dim
        self.funcao_ativacao = funcao_ativacao
        self.treinavel = True
        self.parametros['pesos'] = np.random.randn(entrada_dim, 1) * 0.01
        self.parametros['vies'] = np.zeros((1, 1))
    
    def _ativacao(self, x: np.ndarray) -> np.ndarray:
        if self.funcao_ativacao == 'step':
            return np.where(x >= 0, 1, 0)
        elif self.funcao_ativacao == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return x
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        self.entrada = entrada
        linear = entrada @ self.parametros['pesos'] + self.parametros['vies']
        self.saida = self._ativacao(linear)
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        self.gradientes['pesos'] = self.entrada.T @ gradiente_saida
        self.gradientes['vies'] = np.sum(gradiente_saida, axis=0, keepdims=True)
        return gradiente_saida @ self.parametros['pesos'].T
    
    def treinar_perceptron(self, X: np.ndarray, y: np.ndarray, taxa: float = 0.1, epocas: int = 100):
        """Algoritmo de treinamento do Perceptron."""
        for _ in range(epocas):
            for xi, yi in zip(X, y):
                xi = xi.reshape(1, -1)
                pred = self.frente(xi)
                erro = yi - pred
                self.parametros['pesos'] += taxa * xi.T * erro
                self.parametros['vies'] += taxa * erro


class MLP(RedeNeural):
    """
    Multi-Layer Perceptron (Rede Neural Multicamadas).
    
    Args:
        arquitetura: Lista de dimensões [entrada, oculta1, oculta2, ..., saida]
        ativacao: Função de ativação para camadas ocultas
    """
    
    def __init__(self, arquitetura: List[int], ativacao: str = 'relu', nome: str = "MLP"):
        super().__init__(nome)
        
        for i in range(len(arquitetura) - 1):
            self.adicionar(CamadaDensa(arquitetura[i], arquitetura[i+1]))
            if i < len(arquitetura) - 2:
                if ativacao == 'relu':
                    self.adicionar(CamadaReLU())
                elif ativacao == 'sigmoid':
                    self.adicionar(CamadaSigmoide())
    
    @classmethod
    def criar_classificador(cls, entrada: int, classes: int, ocultas: List[int] = [64, 32]) -> 'MLP':
        """Cria MLP para classificação."""
        arquitetura = [entrada] + ocultas + [classes]
        modelo = cls(arquitetura, ativacao='relu', nome="MLPClassificador")
        modelo.adicionar(CamadaSoftmax())
        return modelo
    
    @classmethod
    def criar_regressor(cls, entrada: int, saida: int = 1, ocultas: List[int] = [64, 32]) -> 'MLP':
        """Cria MLP para regressão."""
        arquitetura = [entrada] + ocultas + [saida]
        return cls(arquitetura, ativacao='relu', nome="MLPRegressor")
