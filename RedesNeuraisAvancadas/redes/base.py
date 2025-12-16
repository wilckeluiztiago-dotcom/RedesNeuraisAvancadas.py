"""
RedesNeuraisAvancadas - Biblioteca de Redes Neurais
Autor: Luiz Tiago Wilcke
Módulo: Base - Classes fundamentais para redes neurais
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable, Dict, Any
import pickle
import json
from datetime import datetime


class Camada(ABC):
    """
    Classe abstrata base para todas as camadas da rede neural.
    
    Atributos:
        entrada: Dados de entrada da camada
        saida: Dados de saída da camada
        gradiente_entrada: Gradiente em relação à entrada
        treinavel: Se a camada possui parâmetros treináveis
    """
    
    def __init__(self, nome: str = None):
        self.nome = nome or self.__class__.__name__
        self.entrada = None
        self.saida = None
        self.gradiente_entrada = None
        self.treinavel = False
        self.parametros: Dict[str, np.ndarray] = {}
        self.gradientes: Dict[str, np.ndarray] = {}
    
    @abstractmethod
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        """Propagação para frente."""
        pass
    
    @abstractmethod
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        """Retropropagação."""
        pass
    
    def obter_parametros(self) -> Dict[str, np.ndarray]:
        """Retorna parâmetros da camada."""
        return self.parametros
    
    def obter_gradientes(self) -> Dict[str, np.ndarray]:
        """Retorna gradientes da camada."""
        return self.gradientes
    
    def contar_parametros(self) -> int:
        """Conta número total de parâmetros."""
        total = 0
        for p in self.parametros.values():
            total += p.size
        return total


class CamadaDensa(Camada):
    """
    Camada totalmente conectada (Dense/Fully Connected).
    
    Args:
        entrada_dim: Dimensão da entrada
        saida_dim: Dimensão da saída
        inicializacao: Método de inicialização ('xavier', 'he', 'uniforme')
        usar_vies: Se deve usar termo de viés (bias)
    """
    
    def __init__(
        self,
        entrada_dim: int,
        saida_dim: int,
        inicializacao: str = 'xavier',
        usar_vies: bool = True,
        nome: str = None
    ):
        super().__init__(nome)
        self.entrada_dim = entrada_dim
        self.saida_dim = saida_dim
        self.usar_vies = usar_vies
        self.treinavel = True
        
        # Inicialização dos pesos
        self._inicializar_pesos(inicializacao)
    
    def _inicializar_pesos(self, metodo: str):
        """Inicializa pesos com o método especificado."""
        if metodo == 'xavier':
            # Inicialização Xavier/Glorot
            limite = np.sqrt(6 / (self.entrada_dim + self.saida_dim))
            self.parametros['pesos'] = np.random.uniform(
                -limite, limite, (self.entrada_dim, self.saida_dim)
            )
        elif metodo == 'he':
            # Inicialização He (para ReLU)
            desvio = np.sqrt(2 / self.entrada_dim)
            self.parametros['pesos'] = np.random.normal(
                0, desvio, (self.entrada_dim, self.saida_dim)
            )
        else:
            # Inicialização uniforme simples
            self.parametros['pesos'] = np.random.randn(
                self.entrada_dim, self.saida_dim
            ) * 0.01
        
        if self.usar_vies:
            self.parametros['vies'] = np.zeros((1, self.saida_dim))
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        """
        Propagação para frente: saida = entrada @ pesos + vies
        """
        self.entrada = entrada
        self.saida = entrada @ self.parametros['pesos']
        if self.usar_vies:
            self.saida += self.parametros['vies']
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        """
        Retropropagação: calcula gradientes dos pesos e entrada.
        """
        # Gradiente dos pesos
        self.gradientes['pesos'] = self.entrada.T @ gradiente_saida
        
        # Gradiente do viés
        if self.usar_vies:
            self.gradientes['vies'] = np.sum(gradiente_saida, axis=0, keepdims=True)
        
        # Gradiente da entrada
        self.gradiente_entrada = gradiente_saida @ self.parametros['pesos'].T
        return self.gradiente_entrada


class CamadaConvolucional(Camada):
    """
    Camada convolucional 2D.
    
    Args:
        filtros: Número de filtros
        tamanho_kernel: Tamanho do kernel (altura, largura)
        passo: Passo da convolução (stride)
        padding: Tipo de padding ('same', 'valid')
    """
    
    def __init__(
        self,
        filtros: int,
        tamanho_kernel: Tuple[int, int] = (3, 3),
        passo: int = 1,
        padding: str = 'same',
        nome: str = None
    ):
        super().__init__(nome)
        self.filtros = filtros
        self.tamanho_kernel = tamanho_kernel
        self.passo = passo
        self.padding = padding
        self.treinavel = True
        self.inicializado = False
    
    def _inicializar(self, canais_entrada: int):
        """Inicializa kernels quando conhecemos os canais de entrada."""
        limite = np.sqrt(6 / (canais_entrada * np.prod(self.tamanho_kernel) + self.filtros))
        self.parametros['kernels'] = np.random.uniform(
            -limite, limite,
            (self.filtros, canais_entrada, *self.tamanho_kernel)
        )
        self.parametros['vies'] = np.zeros(self.filtros)
        self.inicializado = True
    
    def _aplicar_padding(self, entrada: np.ndarray) -> np.ndarray:
        """Aplica padding à entrada."""
        if self.padding == 'valid':
            return entrada
        
        # Padding 'same'
        pad_h = (self.tamanho_kernel[0] - 1) // 2
        pad_w = (self.tamanho_kernel[1] - 1) // 2
        
        return np.pad(
            entrada,
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode='constant'
        )
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        """
        Convolução 2D.
        entrada shape: (batch, canais, altura, largura)
        """
        if not self.inicializado:
            self._inicializar(entrada.shape[1])
        
        self.entrada = entrada
        entrada_pad = self._aplicar_padding(entrada)
        
        batch, canais, altura, largura = entrada_pad.shape
        kh, kw = self.tamanho_kernel
        
        # Dimensões da saída
        altura_saida = (altura - kh) // self.passo + 1
        largura_saida = (largura - kw) // self.passo + 1
        
        self.saida = np.zeros((batch, self.filtros, altura_saida, largura_saida))
        
        # Convolução
        for i in range(altura_saida):
            for j in range(largura_saida):
                regiao = entrada_pad[
                    :, :,
                    i * self.passo:i * self.passo + kh,
                    j * self.passo:j * self.passo + kw
                ]
                for f in range(self.filtros):
                    self.saida[:, f, i, j] = np.sum(
                        regiao * self.parametros['kernels'][f], axis=(1, 2, 3)
                    ) + self.parametros['vies'][f]
        
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        """Retropropagação da convolução."""
        batch, canais, altura, largura = self.entrada.shape
        kh, kw = self.tamanho_kernel
        
        self.gradientes['kernels'] = np.zeros_like(self.parametros['kernels'])
        self.gradientes['vies'] = np.sum(gradiente_saida, axis=(0, 2, 3))
        
        entrada_pad = self._aplicar_padding(self.entrada)
        gradiente_entrada_pad = np.zeros_like(entrada_pad)
        
        _, _, altura_saida, largura_saida = gradiente_saida.shape
        
        for i in range(altura_saida):
            for j in range(largura_saida):
                regiao = entrada_pad[
                    :, :,
                    i * self.passo:i * self.passo + kh,
                    j * self.passo:j * self.passo + kw
                ]
                for f in range(self.filtros):
                    self.gradientes['kernels'][f] += np.sum(
                        regiao * gradiente_saida[:, f, i, j].reshape(-1, 1, 1, 1),
                        axis=0
                    )
                    gradiente_entrada_pad[
                        :, :,
                        i * self.passo:i * self.passo + kh,
                        j * self.passo:j * self.passo + kw
                    ] += self.parametros['kernels'][f] * gradiente_saida[:, f, i, j].reshape(-1, 1, 1, 1)
        
        # Remover padding
        if self.padding == 'same':
            pad_h = (kh - 1) // 2
            pad_w = (kw - 1) // 2
            if pad_h > 0 and pad_w > 0:
                self.gradiente_entrada = gradiente_entrada_pad[:, :, pad_h:-pad_h, pad_w:-pad_w]
            else:
                self.gradiente_entrada = gradiente_entrada_pad
        else:
            self.gradiente_entrada = gradiente_entrada_pad
        
        return self.gradiente_entrada


class CamadaRecorrente(Camada):
    """
    Camada recorrente simples (RNN vanilla).
    
    Args:
        unidades: Número de unidades ocultas
        retornar_sequencias: Se deve retornar toda a sequência
    """
    
    def __init__(
        self,
        unidades: int,
        retornar_sequencias: bool = False,
        nome: str = None
    ):
        super().__init__(nome)
        self.unidades = unidades
        self.retornar_sequencias = retornar_sequencias
        self.treinavel = True
        self.inicializado = False
    
    def _inicializar(self, entrada_dim: int):
        """Inicializa pesos."""
        limite_x = np.sqrt(6 / (entrada_dim + self.unidades))
        limite_h = np.sqrt(6 / (self.unidades + self.unidades))
        
        self.parametros['Wx'] = np.random.uniform(-limite_x, limite_x, (entrada_dim, self.unidades))
        self.parametros['Wh'] = np.random.uniform(-limite_h, limite_h, (self.unidades, self.unidades))
        self.parametros['bh'] = np.zeros((1, self.unidades))
        self.inicializado = True
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        """
        Propagação para frente da RNN.
        entrada shape: (batch, sequencia, features)
        """
        if not self.inicializado:
            self._inicializar(entrada.shape[2])
        
        self.entrada = entrada
        batch, seq_len, _ = entrada.shape
        
        self.estados_ocultos = np.zeros((batch, seq_len + 1, self.unidades))
        self.saidas = np.zeros((batch, seq_len, self.unidades))
        
        for t in range(seq_len):
            self.estados_ocultos[:, t + 1] = np.tanh(
                entrada[:, t] @ self.parametros['Wx'] +
                self.estados_ocultos[:, t] @ self.parametros['Wh'] +
                self.parametros['bh']
            )
            self.saidas[:, t] = self.estados_ocultos[:, t + 1]
        
        if self.retornar_sequencias:
            self.saida = self.saidas
        else:
            self.saida = self.saidas[:, -1]
        
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        """Retropropagação através do tempo (BPTT)."""
        batch, seq_len, _ = self.entrada.shape
        
        self.gradientes['Wx'] = np.zeros_like(self.parametros['Wx'])
        self.gradientes['Wh'] = np.zeros_like(self.parametros['Wh'])
        self.gradientes['bh'] = np.zeros_like(self.parametros['bh'])
        
        gradiente_entrada = np.zeros_like(self.entrada)
        
        if not self.retornar_sequencias:
            gradiente_sequencia = np.zeros((batch, seq_len, self.unidades))
            gradiente_sequencia[:, -1] = gradiente_saida
        else:
            gradiente_sequencia = gradiente_saida
        
        gradiente_h_proximo = np.zeros((batch, self.unidades))
        
        for t in reversed(range(seq_len)):
            gradiente_h = gradiente_sequencia[:, t] + gradiente_h_proximo
            gradiente_tanh = gradiente_h * (1 - self.estados_ocultos[:, t + 1] ** 2)
            
            self.gradientes['Wx'] += self.entrada[:, t].T @ gradiente_tanh
            self.gradientes['Wh'] += self.estados_ocultos[:, t].T @ gradiente_tanh
            self.gradientes['bh'] += np.sum(gradiente_tanh, axis=0, keepdims=True)
            
            gradiente_entrada[:, t] = gradiente_tanh @ self.parametros['Wx'].T
            gradiente_h_proximo = gradiente_tanh @ self.parametros['Wh'].T
        
        self.gradiente_entrada = gradiente_entrada
        return self.gradiente_entrada


class RedeNeural:
    """
    Classe principal para construção e treinamento de redes neurais.
    
    Autor: Luiz Tiago Wilcke
    """
    
    def __init__(self, nome: str = "RedeNeural"):
        self.nome = nome
        self.camadas: List[Camada] = []
        self.otimizador = None
        self.funcao_perda = None
        self.metricas = []
        self.historico = {
            'perda_treino': [],
            'perda_validacao': [],
            'metricas': {}
        }
        self.data_criacao = datetime.now().isoformat()
        self.autor = "Luiz Tiago Wilcke"
    
    def adicionar(self, camada: Camada) -> 'RedeNeural':
        """Adiciona uma camada à rede."""
        self.camadas.append(camada)
        return self
    
    def compilar(self, otimizador, funcao_perda, metricas: List = None):
        """Compila o modelo com otimizador e função de perda."""
        self.otimizador = otimizador
        self.funcao_perda = funcao_perda
        self.metricas = metricas or []
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        """Propagação para frente através de todas as camadas."""
        saida = entrada
        for camada in self.camadas:
            saida = camada.frente(saida, treinamento)
        return saida
    
    def tras(self, gradiente: np.ndarray):
        """Retropropagação através de todas as camadas."""
        for camada in reversed(self.camadas):
            gradiente = camada.tras(gradiente)
    
    def treinar_passo(self, X: np.ndarray, y: np.ndarray) -> float:
        """Um passo de treinamento."""
        # Forward
        previsao = self.frente(X, treinamento=True)
        
        # Calcula perda
        perda = self.funcao_perda.calcular(y, previsao)
        
        # Backward
        gradiente = self.funcao_perda.gradiente(y, previsao)
        self.tras(gradiente)
        
        # Atualiza parâmetros
        for camada in self.camadas:
            if camada.treinavel:
                self.otimizador.atualizar(camada.parametros, camada.gradientes)
        
        return perda
    
    def treinar(
        self,
        X_treino: np.ndarray,
        y_treino: np.ndarray,
        epocas: int = 100,
        tamanho_lote: int = 32,
        X_validacao: np.ndarray = None,
        y_validacao: np.ndarray = None,
        verbose: bool = True
    ):
        """Treina a rede neural."""
        num_amostras = X_treino.shape[0]
        num_lotes = num_amostras // tamanho_lote
        
        for epoca in range(epocas):
            # Embaralha dados
            indices = np.random.permutation(num_amostras)
            X_embaralhado = X_treino[indices]
            y_embaralhado = y_treino[indices]
            
            perda_epoca = 0
            for i in range(num_lotes):
                inicio = i * tamanho_lote
                fim = inicio + tamanho_lote
                X_lote = X_embaralhado[inicio:fim]
                y_lote = y_embaralhado[inicio:fim]
                
                perda_lote = self.treinar_passo(X_lote, y_lote)
                perda_epoca += perda_lote
            
            perda_media = perda_epoca / num_lotes
            self.historico['perda_treino'].append(perda_media)
            
            # Validação
            if X_validacao is not None:
                previsao_val = self.frente(X_validacao, treinamento=False)
                perda_val = self.funcao_perda.calcular(y_validacao, previsao_val)
                self.historico['perda_validacao'].append(perda_val)
            
            if verbose and (epoca + 1) % 10 == 0:
                msg = f"Época {epoca + 1}/{epocas} - Perda: {perda_media:.6f}"
                if X_validacao is not None:
                    msg += f" - Val Perda: {perda_val:.6f}"
                print(msg)
    
    def prever(self, X: np.ndarray) -> np.ndarray:
        """Faz previsões."""
        return self.frente(X, treinamento=False)
    
    def avaliar(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Avalia o modelo."""
        previsao = self.prever(X)
        resultados = {'perda': self.funcao_perda.calcular(y, previsao)}
        
        for metrica in self.metricas:
            nome = metrica.__class__.__name__
            resultados[nome] = metrica.calcular(y, previsao)
        
        return resultados
    
    def resumo(self):
        """Exibe resumo da arquitetura."""
        print(f"\n{'='*60}")
        print(f"Modelo: {self.nome}")
        print(f"Autor: {self.autor}")
        print(f"Criado em: {self.data_criacao}")
        print(f"{'='*60}")
        print(f"{'Camada':<25} {'Saída':<20} {'Parâmetros'}")
        print(f"{'-'*60}")
        
        total_parametros = 0
        for camada in self.camadas:
            num_params = camada.contar_parametros()
            total_parametros += num_params
            print(f"{camada.nome:<25} {'N/A':<20} {num_params:,}")
        
        print(f"{'='*60}")
        print(f"Total de parâmetros: {total_parametros:,}")
        print(f"{'='*60}\n")
    
    def salvar(self, caminho: str):
        """Salva o modelo em arquivo."""
        dados = {
            'nome': self.nome,
            'autor': self.autor,
            'data_criacao': self.data_criacao,
            'camadas': [],
            'historico': self.historico
        }
        
        for camada in self.camadas:
            dados_camada = {
                'tipo': camada.__class__.__name__,
                'nome': camada.nome,
                'parametros': {k: v.tolist() for k, v in camada.parametros.items()}
            }
            dados['camadas'].append(dados_camada)
        
        with open(caminho, 'wb') as f:
            pickle.dump(dados, f)
        
        print(f"Modelo salvo em: {caminho}")
    
    @classmethod
    def carregar(cls, caminho: str) -> 'RedeNeural':
        """Carrega modelo de arquivo."""
        with open(caminho, 'rb') as f:
            dados = pickle.load(f)
        
        modelo = cls(dados['nome'])
        modelo.autor = dados['autor']
        modelo.data_criacao = dados['data_criacao']
        modelo.historico = dados['historico']
        
        print(f"Modelo '{dados['nome']}' carregado com sucesso!")
        return modelo
