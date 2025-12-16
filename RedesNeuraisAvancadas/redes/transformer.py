"""
RedesNeuraisAvancadas - Biblioteca de Redes Neurais
Autor: Luiz Tiago Wilcke
Módulo: Transformer - Mecanismo de Atenção
"""

import numpy as np
from .base import Camada, RedeNeural, CamadaDensa
from .ativacoes import CamadaReLU


class AtencaoMultiCabeca(Camada):
    """Multi-Head Self-Attention."""
    
    def __init__(self, dim_modelo: int, num_cabecas: int = 8, nome: str = None):
        super().__init__(nome or "MultiHeadAttention")
        self.dim_modelo = dim_modelo
        self.num_cabecas = num_cabecas
        self.dim_cabeca = dim_modelo // num_cabecas
        self.treinavel = True
        
        escala = np.sqrt(2 / dim_modelo)
        self.parametros['Wq'] = np.random.randn(dim_modelo, dim_modelo) * escala
        self.parametros['Wk'] = np.random.randn(dim_modelo, dim_modelo) * escala
        self.parametros['Wv'] = np.random.randn(dim_modelo, dim_modelo) * escala
        self.parametros['Wo'] = np.random.randn(dim_modelo, dim_modelo) * escala
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        self.entrada = entrada
        batch, seq_len, _ = entrada.shape
        
        # Projeções
        Q = entrada @ self.parametros['Wq']
        K = entrada @ self.parametros['Wk']
        V = entrada @ self.parametros['Wv']
        
        # Reshape para múltiplas cabeças
        Q = Q.reshape(batch, seq_len, self.num_cabecas, self.dim_cabeca).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.num_cabecas, self.dim_cabeca).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.num_cabecas, self.dim_cabeca).transpose(0, 2, 1, 3)
        
        # Atenção
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.dim_cabeca)
        self.atencao = self._softmax(scores)
        
        contexto = self.atencao @ V
        contexto = contexto.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.dim_modelo)
        
        self.saida = contexto @ self.parametros['Wo']
        return self.saida
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        for key in self.parametros:
            self.gradientes[key] = np.zeros_like(self.parametros[key])
        self.gradiente_entrada = np.zeros_like(self.entrada)
        return self.gradiente_entrada


class CamadaFeedForward(Camada):
    """Feed-Forward Network para Transformer."""
    
    def __init__(self, dim_modelo: int, dim_ff: int = None, nome: str = None):
        super().__init__(nome or "FeedForward")
        dim_ff = dim_ff or dim_modelo * 4
        self.treinavel = True
        
        escala = np.sqrt(2 / dim_modelo)
        self.parametros['W1'] = np.random.randn(dim_modelo, dim_ff) * escala
        self.parametros['b1'] = np.zeros((1, dim_ff))
        self.parametros['W2'] = np.random.randn(dim_ff, dim_modelo) * escala
        self.parametros['b2'] = np.zeros((1, dim_modelo))
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        self.entrada = entrada
        self.h = np.maximum(0, entrada @ self.parametros['W1'] + self.parametros['b1'])
        self.saida = self.h @ self.parametros['W2'] + self.parametros['b2']
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        self.gradientes['W2'] = np.sum(self.h.transpose(0, 2, 1) @ gradiente_saida, axis=0)
        self.gradientes['b2'] = np.sum(gradiente_saida, axis=(0, 1), keepdims=True).reshape(1, -1)
        
        dh = gradiente_saida @ self.parametros['W2'].T
        dh = dh * (self.h > 0)
        
        self.gradientes['W1'] = np.sum(self.entrada.transpose(0, 2, 1) @ dh, axis=0)
        self.gradientes['b1'] = np.sum(dh, axis=(0, 1), keepdims=True).reshape(1, -1)
        
        self.gradiente_entrada = dh @ self.parametros['W1'].T
        return self.gradiente_entrada


class CamadaNormalizacao(Camada):
    """Layer Normalization."""
    
    def __init__(self, dim_modelo: int, epsilon: float = 1e-6, nome: str = None):
        super().__init__(nome or "LayerNorm")
        self.epsilon = epsilon
        self.treinavel = True
        self.parametros['gamma'] = np.ones(dim_modelo)
        self.parametros['beta'] = np.zeros(dim_modelo)
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        self.entrada = entrada
        self.media = np.mean(entrada, axis=-1, keepdims=True)
        self.var = np.var(entrada, axis=-1, keepdims=True)
        self.x_norm = (entrada - self.media) / np.sqrt(self.var + self.epsilon)
        self.saida = self.parametros['gamma'] * self.x_norm + self.parametros['beta']
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        self.gradientes['gamma'] = np.sum(gradiente_saida * self.x_norm, axis=(0, 1))
        self.gradientes['beta'] = np.sum(gradiente_saida, axis=(0, 1))
        self.gradiente_entrada = gradiente_saida
        return self.gradiente_entrada


class BlocoTransformer(Camada):
    """Um bloco encoder do Transformer."""
    
    def __init__(self, dim_modelo: int, num_cabecas: int = 8, nome: str = None):
        super().__init__(nome or "TransformerBlock")
        self.atencao = AtencaoMultiCabeca(dim_modelo, num_cabecas)
        self.norm1 = CamadaNormalizacao(dim_modelo)
        self.ff = CamadaFeedForward(dim_modelo)
        self.norm2 = CamadaNormalizacao(dim_modelo)
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        attn_out = self.atencao.frente(entrada, treinamento)
        x = self.norm1.frente(entrada + attn_out, treinamento)
        ff_out = self.ff.frente(x, treinamento)
        self.saida = self.norm2.frente(x + ff_out, treinamento)
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        return gradiente_saida


class Transformer(RedeNeural):
    """Transformer Encoder."""
    
    def __init__(self, dim_modelo: int = 512, num_camadas: int = 6, num_cabecas: int = 8):
        super().__init__("Transformer")
        for i in range(num_camadas):
            self.adicionar(BlocoTransformer(dim_modelo, num_cabecas, f"Bloco{i+1}"))


def codificacao_posicional(seq_len: int, dim_modelo: int) -> np.ndarray:
    """Gera codificação posicional."""
    posicao = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, dim_modelo, 2) * (-np.log(10000.0) / dim_modelo))
    
    pe = np.zeros((seq_len, dim_modelo))
    pe[:, 0::2] = np.sin(posicao * div_term)
    pe[:, 1::2] = np.cos(posicao * div_term)
    
    return pe
