"""
RedesNeuraisAvancadas - Biblioteca de Redes Neurais
Autor: Luiz Tiago Wilcke
Módulo: Autoencoder e VAE
"""

import numpy as np
from .base import RedeNeural, CamadaDensa
from .ativacoes import CamadaReLU, CamadaSigmoide


class Autoencoder(RedeNeural):
    """Autoencoder para compressão/reconstrução."""
    
    def __init__(self, dim_entrada: int, dim_latente: int, dims_ocultas: list = [128, 64]):
        super().__init__("Autoencoder")
        
        # Encoder
        dims = [dim_entrada] + dims_ocultas + [dim_latente]
        for i in range(len(dims) - 1):
            self.adicionar(CamadaDensa(dims[i], dims[i+1]))
            self.adicionar(CamadaReLU())
        
        # Decoder
        dims_dec = dims[::-1]
        for i in range(len(dims_dec) - 1):
            self.adicionar(CamadaDensa(dims_dec[i], dims_dec[i+1]))
            if i < len(dims_dec) - 2:
                self.adicionar(CamadaReLU())
            else:
                self.adicionar(CamadaSigmoide())


class VAE(RedeNeural):
    """Variational Autoencoder."""
    
    def __init__(self, dim_entrada: int, dim_latente: int, dims_ocultas: list = [128, 64]):
        super().__init__("VAE")
        self.dim_latente = dim_latente
        self.dim_entrada = dim_entrada
        
        # Encoder - produz média e log-variância
        dims = [dim_entrada] + dims_ocultas
        for i in range(len(dims) - 1):
            self.adicionar(CamadaDensa(dims[i], dims[i+1]))
            self.adicionar(CamadaReLU())
        
        self.camada_media = CamadaDensa(dims[-1], dim_latente)
        self.camada_logvar = CamadaDensa(dims[-1], dim_latente)
        
        # Decoder
        self.camadas_decoder = []
        dims_dec = [dim_latente] + dims_ocultas[::-1] + [dim_entrada]
        for i in range(len(dims_dec) - 1):
            self.camadas_decoder.append(CamadaDensa(dims_dec[i], dims_dec[i+1]))
    
    def reparametrizacao(self, media: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """Truque da reparametrização."""
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*media.shape)
        return media + eps * std
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        # Encoder
        h = entrada
        for camada in self.camadas:
            h = camada.frente(h, treinamento)
        
        self.media = self.camada_media.frente(h, treinamento)
        self.logvar = self.camada_logvar.frente(h, treinamento)
        
        # Amostragem
        z = self.reparametrizacao(self.media, self.logvar)
        
        # Decoder
        reconstrucao = z
        for camada in self.camadas_decoder:
            reconstrucao = camada.frente(reconstrucao, treinamento)
        
        self.saida = reconstrucao
        return self.saida
    
    def perda_vae(self, entrada: np.ndarray, reconstrucao: np.ndarray) -> float:
        """Perda do VAE = reconstrução + KL divergence."""
        perda_rec = np.mean((entrada - reconstrucao) ** 2)
        perda_kl = -0.5 * np.mean(1 + self.logvar - self.media**2 - np.exp(self.logvar))
        return perda_rec + perda_kl
