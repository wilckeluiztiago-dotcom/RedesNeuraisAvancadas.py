"""
RedesNeuraisAvancadas - Biblioteca de Redes Neurais
Autor: Luiz Tiago Wilcke
Módulo: CNN - Rede Neural Convolucional
"""

import numpy as np
from typing import List, Tuple
from .base import RedeNeural, Camada, CamadaConvolucional
from .ativacoes import CamadaReLU, CamadaSoftmax


class CamadaPooling(Camada):
    """
    Camada de Pooling (Max ou Average).
    """
    
    def __init__(self, tamanho: int = 2, modo: str = 'max', nome: str = None):
        super().__init__(nome or f"{modo.capitalize()}Pooling")
        self.tamanho = tamanho
        self.modo = modo
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        self.entrada = entrada
        batch, canais, alt, larg = entrada.shape
        alt_saida = alt // self.tamanho
        larg_saida = larg // self.tamanho
        
        self.saida = np.zeros((batch, canais, alt_saida, larg_saida))
        
        for i in range(alt_saida):
            for j in range(larg_saida):
                regiao = entrada[:, :, 
                    i*self.tamanho:(i+1)*self.tamanho,
                    j*self.tamanho:(j+1)*self.tamanho]
                
                if self.modo == 'max':
                    self.saida[:, :, i, j] = np.max(regiao, axis=(2, 3))
                else:
                    self.saida[:, :, i, j] = np.mean(regiao, axis=(2, 3))
        
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        batch, canais, alt, larg = self.entrada.shape
        self.gradiente_entrada = np.zeros_like(self.entrada)
        alt_saida = alt // self.tamanho
        larg_saida = larg // self.tamanho
        
        for i in range(alt_saida):
            for j in range(larg_saida):
                if self.modo == 'max':
                    regiao = self.entrada[:, :,
                        i*self.tamanho:(i+1)*self.tamanho,
                        j*self.tamanho:(j+1)*self.tamanho]
                    mascara = regiao == np.max(regiao, axis=(2, 3), keepdims=True)
                    self.gradiente_entrada[:, :,
                        i*self.tamanho:(i+1)*self.tamanho,
                        j*self.tamanho:(j+1)*self.tamanho] = \
                        mascara * gradiente_saida[:, :, i:i+1, j:j+1]
                else:
                    self.gradiente_entrada[:, :,
                        i*self.tamanho:(i+1)*self.tamanho,
                        j*self.tamanho:(j+1)*self.tamanho] = \
                        gradiente_saida[:, :, i:i+1, j:j+1] / (self.tamanho ** 2)
        
        return self.gradiente_entrada


class CamadaFlatten(Camada):
    """Achata tensor para vetor."""
    
    def __init__(self, nome: str = None):
        super().__init__(nome or "Flatten")
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        self.entrada_shape = entrada.shape
        self.saida = entrada.reshape(entrada.shape[0], -1)
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        return gradiente_saida.reshape(self.entrada_shape)


class CNN(RedeNeural):
    """
    Rede Neural Convolucional.
    """
    
    def __init__(self, nome: str = "CNN"):
        super().__init__(nome)
    
    @classmethod
    def criar_classificador_imagem(
        cls, 
        canais_entrada: int,
        altura: int,
        largura: int,
        classes: int
    ) -> 'CNN':
        """Cria CNN para classificação de imagens."""
        from .base import CamadaDensa
        
        modelo = cls("CNNClassificador")
        modelo.adicionar(CamadaConvolucional(32, (3, 3)))
        modelo.adicionar(CamadaReLU())
        modelo.adicionar(CamadaPooling(2))
        modelo.adicionar(CamadaConvolucional(64, (3, 3)))
        modelo.adicionar(CamadaReLU())
        modelo.adicionar(CamadaPooling(2))
        modelo.adicionar(CamadaFlatten())
        modelo.adicionar(CamadaDensa(64 * (altura // 4) * (largura // 4), 128))
        modelo.adicionar(CamadaReLU())
        modelo.adicionar(CamadaDensa(128, classes))
        modelo.adicionar(CamadaSoftmax())
        
        return modelo


class BatchNormalizacao(Camada):
    """Normalização em lote."""
    
    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.9, nome: str = None):
        super().__init__(nome or "BatchNorm")
        self.epsilon = epsilon
        self.momentum = momentum
        self.treinavel = True
        self.inicializado = False
    
    def _inicializar(self, dim: int):
        self.parametros['gamma'] = np.ones(dim)
        self.parametros['beta'] = np.zeros(dim)
        self.media_mov = np.zeros(dim)
        self.var_mov = np.ones(dim)
        self.inicializado = True
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        if not self.inicializado:
            self._inicializar(entrada.shape[-1])
        
        self.entrada = entrada
        
        if treinamento:
            self.media = np.mean(entrada, axis=0)
            self.var = np.var(entrada, axis=0)
            self.media_mov = self.momentum * self.media_mov + (1 - self.momentum) * self.media
            self.var_mov = self.momentum * self.var_mov + (1 - self.momentum) * self.var
        else:
            self.media = self.media_mov
            self.var = self.var_mov
        
        self.x_norm = (entrada - self.media) / np.sqrt(self.var + self.epsilon)
        self.saida = self.parametros['gamma'] * self.x_norm + self.parametros['beta']
        
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        n = self.entrada.shape[0]
        
        self.gradientes['gamma'] = np.sum(gradiente_saida * self.x_norm, axis=0)
        self.gradientes['beta'] = np.sum(gradiente_saida, axis=0)
        
        std_inv = 1 / np.sqrt(self.var + self.epsilon)
        dx_norm = gradiente_saida * self.parametros['gamma']
        
        dvar = np.sum(dx_norm * (self.entrada - self.media) * -0.5 * std_inv**3, axis=0)
        dmedia = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2 * (self.entrada - self.media), axis=0)
        
        self.gradiente_entrada = dx_norm * std_inv + dvar * 2 * (self.entrada - self.media) / n + dmedia / n
        
        return self.gradiente_entrada
