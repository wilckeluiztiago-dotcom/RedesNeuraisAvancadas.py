"""
RedesNeuraisAvancadas - Biblioteca de Redes Neurais
Autor: Luiz Tiago Wilcke
Módulo: GAN - Generative Adversarial Network
"""

import numpy as np
from .base import RedeNeural, CamadaDensa
from .ativacoes import CamadaReLU, CamadaSigmoide, CamadaLeakyReLU
from .otimizadores import Adam
from .perdas import EntropiaCruzadaBinaria


class Gerador(RedeNeural):
    """Rede geradora para GAN."""
    
    def __init__(self, dim_latente: int, dim_saida: int, dims_ocultas: list = [128, 256]):
        super().__init__("Gerador")
        dims = [dim_latente] + dims_ocultas + [dim_saida]
        for i in range(len(dims) - 1):
            self.adicionar(CamadaDensa(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                self.adicionar(CamadaReLU())
            else:
                self.adicionar(CamadaSigmoide())


class Discriminador(RedeNeural):
    """Rede discriminadora para GAN."""
    
    def __init__(self, dim_entrada: int, dims_ocultas: list = [256, 128]):
        super().__init__("Discriminador")
        dims = [dim_entrada] + dims_ocultas + [1]
        for i in range(len(dims) - 1):
            self.adicionar(CamadaDensa(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                self.adicionar(CamadaLeakyReLU(0.2))
            else:
                self.adicionar(CamadaSigmoide())


class GAN:
    """
    Generative Adversarial Network.
    
    Autor: Luiz Tiago Wilcke
    """
    
    def __init__(self, dim_latente: int, dim_dados: int):
        self.dim_latente = dim_latente
        self.gerador = Gerador(dim_latente, dim_dados)
        self.discriminador = Discriminador(dim_dados)
        
        self.otimizador_g = Adam(taxa_aprendizado=0.0002, beta1=0.5)
        self.otimizador_d = Adam(taxa_aprendizado=0.0002, beta1=0.5)
        self.funcao_perda = EntropiaCruzadaBinaria()
        
        self.historico = {'perda_g': [], 'perda_d': []}
    
    def gerar(self, n_amostras: int) -> np.ndarray:
        """Gera novas amostras."""
        z = np.random.randn(n_amostras, self.dim_latente)
        return self.gerador.frente(z, treinamento=False)
    
    def treinar_passo(self, dados_reais: np.ndarray) -> tuple:
        """Um passo de treinamento."""
        batch_size = dados_reais.shape[0]
        
        # Labels
        reais = np.ones((batch_size, 1))
        falsos = np.zeros((batch_size, 1))
        
        # Treina Discriminador
        z = np.random.randn(batch_size, self.dim_latente)
        dados_falsos = self.gerador.frente(z, treinamento=False)
        
        pred_reais = self.discriminador.frente(dados_reais, treinamento=True)
        pred_falsos = self.discriminador.frente(dados_falsos, treinamento=True)
        
        perda_d_real = self.funcao_perda.calcular(reais, pred_reais)
        perda_d_falso = self.funcao_perda.calcular(falsos, pred_falsos)
        perda_d = (perda_d_real + perda_d_falso) / 2
        
        grad_d = (self.funcao_perda.gradiente(reais, pred_reais) + 
                  self.funcao_perda.gradiente(falsos, pred_falsos)) / 2
        self.discriminador.tras(grad_d)
        
        for camada in self.discriminador.camadas:
            if camada.treinavel:
                self.otimizador_d.atualizar(camada.parametros, camada.gradientes)
        
        # Treina Gerador
        z = np.random.randn(batch_size, self.dim_latente)
        dados_gerados = self.gerador.frente(z, treinamento=True)
        pred = self.discriminador.frente(dados_gerados, treinamento=False)
        
        perda_g = self.funcao_perda.calcular(reais, pred)
        
        grad_g = self.funcao_perda.gradiente(reais, pred)
        self.discriminador.tras(grad_g)
        grad_gerador = self.discriminador.camadas[0].gradiente_entrada
        self.gerador.tras(grad_gerador)
        
        for camada in self.gerador.camadas:
            if camada.treinavel:
                self.otimizador_g.atualizar(camada.parametros, camada.gradientes)
        
        return perda_d, perda_g
    
    def treinar(self, dados: np.ndarray, epocas: int = 100, tamanho_lote: int = 32, verbose: bool = True):
        """Treina a GAN."""
        n_amostras = dados.shape[0]
        n_lotes = n_amostras // tamanho_lote
        
        for epoca in range(epocas):
            indices = np.random.permutation(n_amostras)
            
            perda_d_epoca = 0
            perda_g_epoca = 0
            
            for i in range(n_lotes):
                lote = dados[indices[i*tamanho_lote:(i+1)*tamanho_lote]]
                perda_d, perda_g = self.treinar_passo(lote)
                perda_d_epoca += perda_d
                perda_g_epoca += perda_g
            
            self.historico['perda_d'].append(perda_d_epoca / n_lotes)
            self.historico['perda_g'].append(perda_g_epoca / n_lotes)
            
            if verbose and (epoca + 1) % 10 == 0:
                print(f"Época {epoca+1}/{epocas} - D: {perda_d_epoca/n_lotes:.4f}, G: {perda_g_epoca/n_lotes:.4f}")
