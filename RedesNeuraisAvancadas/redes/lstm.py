"""
RedesNeuraisAvancadas - Biblioteca de Redes Neurais
Autor: Luiz Tiago Wilcke
Módulo: LSTM - Long Short-Term Memory
"""

import numpy as np
from .base import Camada, RedeNeural


class CamadaLSTM(Camada):
    """
    Long Short-Term Memory.
    """
    
    def __init__(self, unidades: int, retornar_sequencias: bool = False, nome: str = None):
        super().__init__(nome or "LSTM")
        self.unidades = unidades
        self.retornar_sequencias = retornar_sequencias
        self.treinavel = True
        self.inicializado = False
    
    def _inicializar(self, dim_entrada: int):
        d, h = dim_entrada, self.unidades
        escala = np.sqrt(2 / (d + h))
        
        # Pesos: forget, input, cell, output gates
        self.parametros['Wf'] = np.random.randn(d + h, h) * escala
        self.parametros['Wi'] = np.random.randn(d + h, h) * escala
        self.parametros['Wc'] = np.random.randn(d + h, h) * escala
        self.parametros['Wo'] = np.random.randn(d + h, h) * escala
        
        self.parametros['bf'] = np.zeros((1, h))
        self.parametros['bi'] = np.zeros((1, h))
        self.parametros['bc'] = np.zeros((1, h))
        self.parametros['bo'] = np.zeros((1, h))
        
        self.inicializado = True
    
    def _sigmoide(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        if not self.inicializado:
            self._inicializar(entrada.shape[2])
        
        self.entrada = entrada
        batch, seq_len, _ = entrada.shape
        h = self.unidades
        
        # Estados
        self.h = np.zeros((batch, seq_len + 1, h))
        self.c = np.zeros((batch, seq_len + 1, h))
        self.cache = []
        
        for t in range(seq_len):
            concat = np.concatenate([self.h[:, t], entrada[:, t]], axis=1)
            
            f = self._sigmoide(concat @ self.parametros['Wf'] + self.parametros['bf'])
            i = self._sigmoide(concat @ self.parametros['Wi'] + self.parametros['bi'])
            c_til = np.tanh(concat @ self.parametros['Wc'] + self.parametros['bc'])
            o = self._sigmoide(concat @ self.parametros['Wo'] + self.parametros['bo'])
            
            self.c[:, t + 1] = f * self.c[:, t] + i * c_til
            self.h[:, t + 1] = o * np.tanh(self.c[:, t + 1])
            
            self.cache.append((concat, f, i, c_til, o))
        
        if self.retornar_sequencias:
            self.saida = self.h[:, 1:]
        else:
            self.saida = self.h[:, -1]
        
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        batch, seq_len, dim = self.entrada.shape
        h = self.unidades
        
        for key in ['Wf', 'Wi', 'Wc', 'Wo', 'bf', 'bi', 'bc', 'bo']:
            self.gradientes[key] = np.zeros_like(self.parametros[key])
        
        gradiente_entrada = np.zeros_like(self.entrada)
        
        if not self.retornar_sequencias:
            grad_h = gradiente_saida.copy()
            grad_seq = np.zeros((batch, seq_len, h))
            grad_seq[:, -1] = grad_h
        else:
            grad_seq = gradiente_saida
        
        dh_prox = np.zeros((batch, h))
        dc_prox = np.zeros((batch, h))
        
        for t in reversed(range(seq_len)):
            concat, f, i, c_til, o = self.cache[t]
            
            dh = grad_seq[:, t] + dh_prox
            dc = dh * o * (1 - np.tanh(self.c[:, t + 1])**2) + dc_prox
            
            do = dh * np.tanh(self.c[:, t + 1]) * o * (1 - o)
            df = dc * self.c[:, t] * f * (1 - f)
            di = dc * c_til * i * (1 - i)
            dc_til = dc * i * (1 - c_til**2)
            
            self.gradientes['Wo'] += concat.T @ do
            self.gradientes['Wf'] += concat.T @ df
            self.gradientes['Wi'] += concat.T @ di
            self.gradientes['Wc'] += concat.T @ dc_til
            
            self.gradientes['bo'] += np.sum(do, axis=0, keepdims=True)
            self.gradientes['bf'] += np.sum(df, axis=0, keepdims=True)
            self.gradientes['bi'] += np.sum(di, axis=0, keepdims=True)
            self.gradientes['bc'] += np.sum(dc_til, axis=0, keepdims=True)
            
            d_concat = do @ self.parametros['Wo'].T + df @ self.parametros['Wf'].T + \
                       di @ self.parametros['Wi'].T + dc_til @ self.parametros['Wc'].T
            
            dh_prox = d_concat[:, :h]
            dc_prox = dc * f
            gradiente_entrada[:, t] = d_concat[:, h:]
        
        self.gradiente_entrada = gradiente_entrada
        return self.gradiente_entrada


class CamadaGRU(Camada):
    """Gated Recurrent Unit."""
    
    def __init__(self, unidades: int, retornar_sequencias: bool = False, nome: str = None):
        super().__init__(nome or "GRU")
        self.unidades = unidades
        self.retornar_sequencias = retornar_sequencias
        self.treinavel = True
        self.inicializado = False
    
    def _inicializar(self, dim_entrada: int):
        d, h = dim_entrada, self.unidades
        escala = np.sqrt(2 / (d + h))
        
        self.parametros['Wz'] = np.random.randn(d + h, h) * escala
        self.parametros['Wr'] = np.random.randn(d + h, h) * escala
        self.parametros['Wh'] = np.random.randn(d + h, h) * escala
        
        self.parametros['bz'] = np.zeros((1, h))
        self.parametros['br'] = np.zeros((1, h))
        self.parametros['bh'] = np.zeros((1, h))
        
        self.inicializado = True
    
    def _sigmoide(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        if not self.inicializado:
            self._inicializar(entrada.shape[2])
        
        self.entrada = entrada
        batch, seq_len, _ = entrada.shape
        h = self.unidades
        
        self.h = np.zeros((batch, seq_len + 1, h))
        
        for t in range(seq_len):
            concat = np.concatenate([self.h[:, t], entrada[:, t]], axis=1)
            
            z = self._sigmoide(concat @ self.parametros['Wz'] + self.parametros['bz'])
            r = self._sigmoide(concat @ self.parametros['Wr'] + self.parametros['br'])
            
            concat_r = np.concatenate([r * self.h[:, t], entrada[:, t]], axis=1)
            h_til = np.tanh(concat_r @ self.parametros['Wh'] + self.parametros['bh'])
            
            self.h[:, t + 1] = (1 - z) * self.h[:, t] + z * h_til
        
        if self.retornar_sequencias:
            self.saida = self.h[:, 1:]
        else:
            self.saida = self.h[:, -1]
        
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        # Simplificado
        batch, seq_len, dim = self.entrada.shape
        self.gradiente_entrada = np.zeros_like(self.entrada)
        
        for key in self.parametros:
            self.gradientes[key] = np.zeros_like(self.parametros[key])
        
        return self.gradiente_entrada
