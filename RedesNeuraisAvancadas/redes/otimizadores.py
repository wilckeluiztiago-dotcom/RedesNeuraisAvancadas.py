"""
RedesNeuraisAvancadas - Biblioteca de Redes Neurais
Autor: Luiz Tiago Wilcke
Módulo: Otimizadores - Algoritmos de otimização
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict


class Otimizador(ABC):
    """Classe abstrata base para otimizadores."""
    
    def __init__(self, taxa_aprendizado: float = 0.01):
        self.taxa_aprendizado = taxa_aprendizado
    
    @abstractmethod
    def atualizar(self, parametros: Dict[str, np.ndarray], gradientes: Dict[str, np.ndarray]):
        pass


class SGD(Otimizador):
    """
    Gradiente Descendente Estocástico com momentum opcional.
    
    Args:
        taxa_aprendizado: Taxa de aprendizado
        momentum: Coeficiente de momentum (0 = sem momentum)
        nesterov: Se deve usar momentum de Nesterov
    """
    
    def __init__(
        self,
        taxa_aprendizado: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False
    ):
        super().__init__(taxa_aprendizado)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocidades: Dict[int, Dict[str, np.ndarray]] = {}
        self.contador = 0
    
    def atualizar(self, parametros: Dict[str, np.ndarray], gradientes: Dict[str, np.ndarray]):
        id_params = id(parametros)
        
        if id_params not in self.velocidades:
            self.velocidades[id_params] = {}
            for nome in parametros:
                self.velocidades[id_params][nome] = np.zeros_like(parametros[nome])
        
        for nome in parametros:
            if nome not in gradientes:
                continue
            
            v = self.velocidades[id_params][nome]
            g = gradientes[nome]
            
            v_novo = self.momentum * v - self.taxa_aprendizado * g
            self.velocidades[id_params][nome] = v_novo
            
            if self.nesterov:
                parametros[nome] += self.momentum * v_novo - self.taxa_aprendizado * g
            else:
                parametros[nome] += v_novo


class Adam(Otimizador):
    """
    Adam (Adaptive Moment Estimation).
    
    Args:
        taxa_aprendizado: Taxa de aprendizado
        beta1: Decaimento exponencial para primeiro momento
        beta2: Decaimento exponencial para segundo momento
        epsilon: Constante para estabilidade numérica
    """
    
    def __init__(
        self,
        taxa_aprendizado: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        super().__init__(taxa_aprendizado)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Dict[int, Dict[str, np.ndarray]] = {}
        self.v: Dict[int, Dict[str, np.ndarray]] = {}
        self.t: Dict[int, int] = {}
    
    def atualizar(self, parametros: Dict[str, np.ndarray], gradientes: Dict[str, np.ndarray]):
        id_params = id(parametros)
        
        if id_params not in self.m:
            self.m[id_params] = {}
            self.v[id_params] = {}
            self.t[id_params] = 0
            for nome in parametros:
                self.m[id_params][nome] = np.zeros_like(parametros[nome])
                self.v[id_params][nome] = np.zeros_like(parametros[nome])
        
        self.t[id_params] += 1
        t = self.t[id_params]
        
        for nome in parametros:
            if nome not in gradientes:
                continue
            
            g = gradientes[nome]
            
            # Atualiza momentos
            self.m[id_params][nome] = self.beta1 * self.m[id_params][nome] + (1 - self.beta1) * g
            self.v[id_params][nome] = self.beta2 * self.v[id_params][nome] + (1 - self.beta2) * g**2
            
            # Correção de viés
            m_corrigido = self.m[id_params][nome] / (1 - self.beta1**t)
            v_corrigido = self.v[id_params][nome] / (1 - self.beta2**t)
            
            # Atualiza parâmetros
            parametros[nome] -= self.taxa_aprendizado * m_corrigido / (np.sqrt(v_corrigido) + self.epsilon)


class AdamW(Otimizador):
    """
    AdamW (Adam com Weight Decay desacoplado).
    
    Args:
        taxa_aprendizado: Taxa de aprendizado
        beta1: Decaimento exponencial para primeiro momento
        beta2: Decaimento exponencial para segundo momento
        epsilon: Constante para estabilidade numérica
        decaimento_peso: Coeficiente de regularização L2
    """
    
    def __init__(
        self,
        taxa_aprendizado: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        decaimento_peso: float = 0.01
    ):
        super().__init__(taxa_aprendizado)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decaimento_peso = decaimento_peso
        self.m: Dict[int, Dict[str, np.ndarray]] = {}
        self.v: Dict[int, Dict[str, np.ndarray]] = {}
        self.t: Dict[int, int] = {}
    
    def atualizar(self, parametros: Dict[str, np.ndarray], gradientes: Dict[str, np.ndarray]):
        id_params = id(parametros)
        
        if id_params not in self.m:
            self.m[id_params] = {}
            self.v[id_params] = {}
            self.t[id_params] = 0
            for nome in parametros:
                self.m[id_params][nome] = np.zeros_like(parametros[nome])
                self.v[id_params][nome] = np.zeros_like(parametros[nome])
        
        self.t[id_params] += 1
        t = self.t[id_params]
        
        for nome in parametros:
            if nome not in gradientes:
                continue
            
            g = gradientes[nome]
            
            # Atualiza momentos
            self.m[id_params][nome] = self.beta1 * self.m[id_params][nome] + (1 - self.beta1) * g
            self.v[id_params][nome] = self.beta2 * self.v[id_params][nome] + (1 - self.beta2) * g**2
            
            # Correção de viés
            m_corrigido = self.m[id_params][nome] / (1 - self.beta1**t)
            v_corrigido = self.v[id_params][nome] / (1 - self.beta2**t)
            
            # Atualiza parâmetros com weight decay
            parametros[nome] -= self.taxa_aprendizado * (
                m_corrigido / (np.sqrt(v_corrigido) + self.epsilon) +
                self.decaimento_peso * parametros[nome]
            )


class RMSprop(Otimizador):
    """
    RMSprop (Root Mean Square Propagation).
    
    Args:
        taxa_aprendizado: Taxa de aprendizado
        rho: Fator de decaimento
        epsilon: Constante para estabilidade numérica
    """
    
    def __init__(
        self,
        taxa_aprendizado: float = 0.001,
        rho: float = 0.9,
        epsilon: float = 1e-8
    ):
        super().__init__(taxa_aprendizado)
        self.rho = rho
        self.epsilon = epsilon
        self.cache: Dict[int, Dict[str, np.ndarray]] = {}
    
    def atualizar(self, parametros: Dict[str, np.ndarray], gradientes: Dict[str, np.ndarray]):
        id_params = id(parametros)
        
        if id_params not in self.cache:
            self.cache[id_params] = {}
            for nome in parametros:
                self.cache[id_params][nome] = np.zeros_like(parametros[nome])
        
        for nome in parametros:
            if nome not in gradientes:
                continue
            
            g = gradientes[nome]
            
            # Atualiza cache
            self.cache[id_params][nome] = self.rho * self.cache[id_params][nome] + (1 - self.rho) * g**2
            
            # Atualiza parâmetros
            parametros[nome] -= self.taxa_aprendizado * g / (np.sqrt(self.cache[id_params][nome]) + self.epsilon)


class Adagrad(Otimizador):
    """
    Adagrad (Adaptive Gradient Algorithm).
    
    Args:
        taxa_aprendizado: Taxa de aprendizado
        epsilon: Constante para estabilidade numérica
    """
    
    def __init__(
        self,
        taxa_aprendizado: float = 0.01,
        epsilon: float = 1e-8
    ):
        super().__init__(taxa_aprendizado)
        self.epsilon = epsilon
        self.cache: Dict[int, Dict[str, np.ndarray]] = {}
    
    def atualizar(self, parametros: Dict[str, np.ndarray], gradientes: Dict[str, np.ndarray]):
        id_params = id(parametros)
        
        if id_params not in self.cache:
            self.cache[id_params] = {}
            for nome in parametros:
                self.cache[id_params][nome] = np.zeros_like(parametros[nome])
        
        for nome in parametros:
            if nome not in gradientes:
                continue
            
            g = gradientes[nome]
            
            # Acumula gradientes quadrados
            self.cache[id_params][nome] += g**2
            
            # Atualiza parâmetros
            parametros[nome] -= self.taxa_aprendizado * g / (np.sqrt(self.cache[id_params][nome]) + self.epsilon)


class Adadelta(Otimizador):
    """
    Adadelta.
    
    Args:
        rho: Fator de decaimento
        epsilon: Constante para estabilidade numérica
    """
    
    def __init__(
        self,
        rho: float = 0.95,
        epsilon: float = 1e-6
    ):
        super().__init__(1.0)  # Adadelta não usa taxa de aprendizado explícita
        self.rho = rho
        self.epsilon = epsilon
        self.cache_g: Dict[int, Dict[str, np.ndarray]] = {}
        self.cache_x: Dict[int, Dict[str, np.ndarray]] = {}
    
    def atualizar(self, parametros: Dict[str, np.ndarray], gradientes: Dict[str, np.ndarray]):
        id_params = id(parametros)
        
        if id_params not in self.cache_g:
            self.cache_g[id_params] = {}
            self.cache_x[id_params] = {}
            for nome in parametros:
                self.cache_g[id_params][nome] = np.zeros_like(parametros[nome])
                self.cache_x[id_params][nome] = np.zeros_like(parametros[nome])
        
        for nome in parametros:
            if nome not in gradientes:
                continue
            
            g = gradientes[nome]
            
            # Atualiza média móvel de gradientes quadrados
            self.cache_g[id_params][nome] = self.rho * self.cache_g[id_params][nome] + (1 - self.rho) * g**2
            
            # Calcula atualização
            rms_g = np.sqrt(self.cache_g[id_params][nome] + self.epsilon)
            rms_x = np.sqrt(self.cache_x[id_params][nome] + self.epsilon)
            delta = -(rms_x / rms_g) * g
            
            # Atualiza média móvel de atualizações quadradas
            self.cache_x[id_params][nome] = self.rho * self.cache_x[id_params][nome] + (1 - self.rho) * delta**2
            
            # Atualiza parâmetros
            parametros[nome] += delta


class NAdam(Otimizador):
    """
    NAdam (Nesterov-accelerated Adam).
    
    Args:
        taxa_aprendizado: Taxa de aprendizado
        beta1: Decaimento exponencial para primeiro momento
        beta2: Decaimento exponencial para segundo momento
        epsilon: Constante para estabilidade numérica
    """
    
    def __init__(
        self,
        taxa_aprendizado: float = 0.002,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        super().__init__(taxa_aprendizado)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Dict[int, Dict[str, np.ndarray]] = {}
        self.v: Dict[int, Dict[str, np.ndarray]] = {}
        self.t: Dict[int, int] = {}
    
    def atualizar(self, parametros: Dict[str, np.ndarray], gradientes: Dict[str, np.ndarray]):
        id_params = id(parametros)
        
        if id_params not in self.m:
            self.m[id_params] = {}
            self.v[id_params] = {}
            self.t[id_params] = 0
            for nome in parametros:
                self.m[id_params][nome] = np.zeros_like(parametros[nome])
                self.v[id_params][nome] = np.zeros_like(parametros[nome])
        
        self.t[id_params] += 1
        t = self.t[id_params]
        
        for nome in parametros:
            if nome not in gradientes:
                continue
            
            g = gradientes[nome]
            
            # Atualiza momentos
            self.m[id_params][nome] = self.beta1 * self.m[id_params][nome] + (1 - self.beta1) * g
            self.v[id_params][nome] = self.beta2 * self.v[id_params][nome] + (1 - self.beta2) * g**2
            
            # Correção de viés
            m_corrigido = self.m[id_params][nome] / (1 - self.beta1**t)
            v_corrigido = self.v[id_params][nome] / (1 - self.beta2**t)
            
            # Nesterov momentum
            m_nesterov = self.beta1 * m_corrigido + (1 - self.beta1) * g / (1 - self.beta1**t)
            
            # Atualiza parâmetros
            parametros[nome] -= self.taxa_aprendizado * m_nesterov / (np.sqrt(v_corrigido) + self.epsilon)


class RAdam(Otimizador):
    """
    RAdam (Rectified Adam).
    
    Args:
        taxa_aprendizado: Taxa de aprendizado
        beta1: Decaimento exponencial para primeiro momento
        beta2: Decaimento exponencial para segundo momento
        epsilon: Constante para estabilidade numérica
    """
    
    def __init__(
        self,
        taxa_aprendizado: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        super().__init__(taxa_aprendizado)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.rho_inf = 2 / (1 - beta2) - 1
        self.m: Dict[int, Dict[str, np.ndarray]] = {}
        self.v: Dict[int, Dict[str, np.ndarray]] = {}
        self.t: Dict[int, int] = {}
    
    def atualizar(self, parametros: Dict[str, np.ndarray], gradientes: Dict[str, np.ndarray]):
        id_params = id(parametros)
        
        if id_params not in self.m:
            self.m[id_params] = {}
            self.v[id_params] = {}
            self.t[id_params] = 0
            for nome in parametros:
                self.m[id_params][nome] = np.zeros_like(parametros[nome])
                self.v[id_params][nome] = np.zeros_like(parametros[nome])
        
        self.t[id_params] += 1
        t = self.t[id_params]
        
        for nome in parametros:
            if nome not in gradientes:
                continue
            
            g = gradientes[nome]
            
            # Atualiza momentos
            self.m[id_params][nome] = self.beta1 * self.m[id_params][nome] + (1 - self.beta1) * g
            self.v[id_params][nome] = self.beta2 * self.v[id_params][nome] + (1 - self.beta2) * g**2
            
            # Correção de viés para primeiro momento
            m_corrigido = self.m[id_params][nome] / (1 - self.beta1**t)
            
            # Calcula comprimento do SMA
            rho_t = self.rho_inf - 2 * t * (self.beta2**t) / (1 - self.beta2**t)
            
            if rho_t > 5:
                # Variância tratável
                v_corrigido = np.sqrt(self.v[id_params][nome] / (1 - self.beta2**t))
                r_t = np.sqrt(
                    ((rho_t - 4) * (rho_t - 2) * self.rho_inf) /
                    ((self.rho_inf - 4) * (self.rho_inf - 2) * rho_t)
                )
                parametros[nome] -= self.taxa_aprendizado * r_t * m_corrigido / (v_corrigido + self.epsilon)
            else:
                # Usa SGD com momentum
                parametros[nome] -= self.taxa_aprendizado * m_corrigido


def obter_otimizador(nome: str, **kwargs) -> Otimizador:
    """Retorna instância de otimizador pelo nome."""
    otimizadores = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'nadam': NAdam,
        'radam': RAdam
    }
    
    if nome.lower() not in otimizadores:
        raise ValueError(f"Otimizador '{nome}' não encontrado. Opções: {list(otimizadores.keys())}")
    
    return otimizadores[nome.lower()](**kwargs)
