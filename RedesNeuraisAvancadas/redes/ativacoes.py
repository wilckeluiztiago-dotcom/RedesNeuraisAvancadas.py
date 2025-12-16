"""
RedesNeuraisAvancadas - Biblioteca de Redes Neurais
Autor: Luiz Tiago Wilcke
Módulo: Ativações - Funções de ativação
"""

import numpy as np
from abc import ABC, abstractmethod
from .base import Camada


class Ativacao(ABC):
    """Classe abstrata base para funções de ativação."""
    
    @abstractmethod
    def aplicar(self, x: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def derivada(self, x: np.ndarray) -> np.ndarray:
        pass


class ReLU(Ativacao):
    """Rectified Linear Unit."""
    
    def aplicar(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def derivada(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class LeakyReLU(Ativacao):
    """Leaky ReLU com inclinação negativa."""
    
    def __init__(self, alfa: float = 0.01):
        self.alfa = alfa
    
    def aplicar(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alfa * x)
    
    def derivada(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, self.alfa)


class ELU(Ativacao):
    """Exponential Linear Unit."""
    
    def __init__(self, alfa: float = 1.0):
        self.alfa = alfa
    
    def aplicar(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alfa * (np.exp(x) - 1))
    
    def derivada(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, self.aplicar(x) + self.alfa)


class SELU(Ativacao):
    """Scaled Exponential Linear Unit."""
    
    def __init__(self):
        self.alfa = 1.6732632423543772848170429916717
        self.escala = 1.0507009873554804934193349852946
    
    def aplicar(self, x: np.ndarray) -> np.ndarray:
        return self.escala * np.where(x > 0, x, self.alfa * (np.exp(x) - 1))
    
    def derivada(self, x: np.ndarray) -> np.ndarray:
        return self.escala * np.where(x > 0, 1.0, self.alfa * np.exp(x))


class Sigmoide(Ativacao):
    """Função sigmoide."""
    
    def aplicar(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def derivada(self, x: np.ndarray) -> np.ndarray:
        s = self.aplicar(x)
        return s * (1 - s)


class Tanh(Ativacao):
    """Tangente hiperbólica."""
    
    def aplicar(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def derivada(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class Softmax(Ativacao):
    """Softmax para classificação multi-classe."""
    
    def aplicar(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def derivada(self, x: np.ndarray) -> np.ndarray:
        s = self.aplicar(x)
        return s * (1 - s)


class Swish(Ativacao):
    """Swish: x * sigmoid(x)."""
    
    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.sigmoide = Sigmoide()
    
    def aplicar(self, x: np.ndarray) -> np.ndarray:
        return x * self.sigmoide.aplicar(self.beta * x)
    
    def derivada(self, x: np.ndarray) -> np.ndarray:
        sig = self.sigmoide.aplicar(self.beta * x)
        return sig + self.beta * x * sig * (1 - sig)


class GELU(Ativacao):
    """Gaussian Error Linear Unit."""
    
    def aplicar(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def derivada(self, x: np.ndarray) -> np.ndarray:
        sech2 = 1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))**2
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))) + \
               0.5 * x * sech2 * np.sqrt(2 / np.pi) * (1 + 0.134145 * x**2)


class Mish(Ativacao):
    """Mish: x * tanh(softplus(x))."""
    
    def aplicar(self, x: np.ndarray) -> np.ndarray:
        return x * np.tanh(np.log(1 + np.exp(x)))
    
    def derivada(self, x: np.ndarray) -> np.ndarray:
        sp = np.log(1 + np.exp(x))
        tanh_sp = np.tanh(sp)
        sig = 1 / (1 + np.exp(-x))
        return tanh_sp + x * sig * (1 - tanh_sp**2)


class Softplus(Ativacao):
    """Softplus: log(1 + exp(x))."""
    
    def aplicar(self, x: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(np.clip(x, -500, 500)))
    
    def derivada(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))


# ==============================================
# CAMADAS DE ATIVAÇÃO
# ==============================================

class CamadaAtivacao(Camada):
    """Camada que aplica uma função de ativação."""
    
    def __init__(self, ativacao: Ativacao, nome: str = None):
        super().__init__(nome or ativacao.__class__.__name__)
        self.ativacao = ativacao
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        self.entrada = entrada
        self.saida = self.ativacao.aplicar(entrada)
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        self.gradiente_entrada = gradiente_saida * self.ativacao.derivada(self.entrada)
        return self.gradiente_entrada


class CamadaReLU(CamadaAtivacao):
    def __init__(self, nome: str = None):
        super().__init__(ReLU(), nome)


class CamadaLeakyReLU(CamadaAtivacao):
    def __init__(self, alfa: float = 0.01, nome: str = None):
        super().__init__(LeakyReLU(alfa), nome)


class CamadaSigmoide(CamadaAtivacao):
    def __init__(self, nome: str = None):
        super().__init__(Sigmoide(), nome)


class CamadaTanh(CamadaAtivacao):
    def __init__(self, nome: str = None):
        super().__init__(Tanh(), nome)


class CamadaSoftmax(CamadaAtivacao):
    def __init__(self, nome: str = None):
        super().__init__(Softmax(), nome)


class CamadaSwish(CamadaAtivacao):
    def __init__(self, beta: float = 1.0, nome: str = None):
        super().__init__(Swish(beta), nome)


class CamadaGELU(CamadaAtivacao):
    def __init__(self, nome: str = None):
        super().__init__(GELU(), nome)


class CamadaMish(CamadaAtivacao):
    def __init__(self, nome: str = None):
        super().__init__(Mish(), nome)


# ==============================================
# FUNÇÃO AUXILIAR
# ==============================================

def obter_ativacao(nome: str) -> Ativacao:
    """Retorna instância de ativação pelo nome."""
    ativacoes = {
        'relu': ReLU,
        'leaky_relu': LeakyReLU,
        'elu': ELU,
        'selu': SELU,
        'sigmoide': Sigmoide,
        'sigmoid': Sigmoide,
        'tanh': Tanh,
        'softmax': Softmax,
        'swish': Swish,
        'gelu': GELU,
        'mish': Mish,
        'softplus': Softplus
    }
    
    if nome.lower() not in ativacoes:
        raise ValueError(f"Ativação '{nome}' não encontrada. Opções: {list(ativacoes.keys())}")
    
    return ativacoes[nome.lower()]()
