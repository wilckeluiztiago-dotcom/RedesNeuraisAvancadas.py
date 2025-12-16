"""
RedesNeuraisAvancadas - Biblioteca de Redes Neurais
Autor: Luiz Tiago Wilcke
Módulo: Redes Especiais - ResNet, DenseNet, Hopfield, Kohonen, RBF, etc.
"""

import numpy as np
from .base import Camada, RedeNeural, CamadaDensa
from .ativacoes import CamadaReLU, obter_ativacao


# ==============================================
# BLOCO RESIDUAL (ResNet)
# ==============================================

class BlocoResidual(Camada):
    """Bloco residual para ResNet."""
    
    def __init__(self, dim: int, nome: str = None):
        super().__init__(nome or "ResidualBlock")
        self.treinavel = True
        escala = np.sqrt(2 / dim)
        self.parametros['W1'] = np.random.randn(dim, dim) * escala
        self.parametros['b1'] = np.zeros((1, dim))
        self.parametros['W2'] = np.random.randn(dim, dim) * escala
        self.parametros['b2'] = np.zeros((1, dim))
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        self.entrada = entrada
        self.h1 = np.maximum(0, entrada @ self.parametros['W1'] + self.parametros['b1'])
        self.h2 = self.h1 @ self.parametros['W2'] + self.parametros['b2']
        self.saida = np.maximum(0, self.h2 + entrada)  # Skip connection
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        for key in self.parametros:
            self.gradientes[key] = np.zeros_like(self.parametros[key])
        self.gradiente_entrada = gradiente_saida
        return self.gradiente_entrada


class ResNet(RedeNeural):
    """Rede Residual."""
    
    def __init__(self, dim_entrada: int, dim_saida: int, num_blocos: int = 4):
        super().__init__("ResNet")
        self.adicionar(CamadaDensa(dim_entrada, 64))
        self.adicionar(CamadaReLU())
        for _ in range(num_blocos):
            self.adicionar(BlocoResidual(64))
        self.adicionar(CamadaDensa(64, dim_saida))


# ==============================================
# BLOCO DENSO (DenseNet)
# ==============================================

class BlocoDenso(Camada):
    """Bloco denso para DenseNet."""
    
    def __init__(self, dim_entrada: int, taxa_crescimento: int = 12, nome: str = None):
        super().__init__(nome or "DenseBlock")
        self.taxa_crescimento = taxa_crescimento
        self.treinavel = True
        self.dim_entrada = dim_entrada
        self.parametros['W'] = np.random.randn(dim_entrada, taxa_crescimento) * np.sqrt(2/dim_entrada)
        self.parametros['b'] = np.zeros((1, taxa_crescimento))
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        self.entrada = entrada
        h = np.maximum(0, entrada @ self.parametros['W'] + self.parametros['b'])
        self.saida = np.concatenate([entrada, h], axis=-1)  # Concatenação
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        for key in self.parametros:
            self.gradientes[key] = np.zeros_like(self.parametros[key])
        self.gradiente_entrada = gradiente_saida[:, :self.dim_entrada]
        return self.gradiente_entrada


# ==============================================
# REDE DE HOPFIELD
# ==============================================

class RedeHopfield:
    """
    Rede de Hopfield (memória associativa).
    Autor: Luiz Tiago Wilcke
    """
    
    def __init__(self, tamanho: int):
        self.tamanho = tamanho
        self.pesos = np.zeros((tamanho, tamanho))
    
    def treinar(self, padroes: np.ndarray):
        """Treina com padrões (cada linha é um padrão bipolar -1/+1)."""
        n_padroes = padroes.shape[0]
        self.pesos = np.zeros((self.tamanho, self.tamanho))
        
        for p in padroes:
            self.pesos += np.outer(p, p)
        
        self.pesos /= n_padroes
        np.fill_diagonal(self.pesos, 0)
    
    def recuperar(self, padrao: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Recupera padrão memorizado."""
        estado = padrao.copy()
        
        for _ in range(max_iter):
            estado_anterior = estado.copy()
            for i in np.random.permutation(self.tamanho):
                estado[i] = np.sign(self.pesos[i] @ estado)
            if np.array_equal(estado, estado_anterior):
                break
        
        return estado
    
    def energia(self, estado: np.ndarray) -> float:
        """Calcula energia do estado."""
        return -0.5 * estado @ self.pesos @ estado


# ==============================================
# MAPA AUTO-ORGANIZÁVEL (Kohonen SOM)
# ==============================================

class MapaKohonen:
    """
    Self-Organizing Map (SOM) de Kohonen.
    Autor: Luiz Tiago Wilcke
    """
    
    def __init__(self, largura: int, altura: int, dim_entrada: int):
        self.largura = largura
        self.altura = altura
        self.dim_entrada = dim_entrada
        self.pesos = np.random.randn(largura, altura, dim_entrada)
    
    def encontrar_bmu(self, entrada: np.ndarray) -> tuple:
        """Encontra Best Matching Unit."""
        distancias = np.sum((self.pesos - entrada) ** 2, axis=2)
        idx = np.unravel_index(np.argmin(distancias), distancias.shape)
        return idx
    
    def treinar(self, dados: np.ndarray, epocas: int = 100, taxa_inicial: float = 0.5, raio_inicial: float = None):
        """Treina o SOM."""
        raio_inicial = raio_inicial or max(self.largura, self.altura) / 2
        
        for epoca in range(epocas):
            taxa = taxa_inicial * np.exp(-epoca / epocas)
            raio = raio_inicial * np.exp(-epoca / epocas)
            
            for amostra in dados[np.random.permutation(len(dados))]:
                bmu_x, bmu_y = self.encontrar_bmu(amostra)
                
                for i in range(self.largura):
                    for j in range(self.altura):
                        dist = np.sqrt((i - bmu_x)**2 + (j - bmu_y)**2)
                        if dist <= raio:
                            influencia = np.exp(-dist**2 / (2 * raio**2))
                            self.pesos[i, j] += taxa * influencia * (amostra - self.pesos[i, j])


# ==============================================
# REDE RBF (Radial Basis Function)
# ==============================================

class RedeRBF(RedeNeural):
    """
    Radial Basis Function Network.
    Autor: Luiz Tiago Wilcke
    """
    
    def __init__(self, dim_entrada: int, num_centros: int, dim_saida: int):
        super().__init__("RBF")
        self.num_centros = num_centros
        self.centros = np.random.randn(num_centros, dim_entrada)
        self.sigma = 1.0
        
        self.camada_rbf = CamadaRBF(self.centros, self.sigma)
        self.camada_saida = CamadaDensa(num_centros, dim_saida)
        
        self.adicionar(self.camada_rbf)
        self.adicionar(self.camada_saida)


class CamadaRBF(Camada):
    """Camada RBF."""
    
    def __init__(self, centros: np.ndarray, sigma: float = 1.0, nome: str = None):
        super().__init__(nome or "RBF")
        self.centros = centros
        self.sigma = sigma
    
    def frente(self, entrada: np.ndarray, treinamento: bool = True) -> np.ndarray:
        self.entrada = entrada
        distancias = np.array([[np.linalg.norm(x - c) for c in self.centros] for x in entrada])
        self.saida = np.exp(-distancias**2 / (2 * self.sigma**2))
        return self.saida
    
    def tras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        self.gradiente_entrada = np.zeros_like(self.entrada)
        return self.gradiente_entrada


# ==============================================
# MÁQUINA DE BOLTZMANN RESTRITA (RBM)
# ==============================================

class MaquinaBoltzmann:
    """
    Restricted Boltzmann Machine.
    Autor: Luiz Tiago Wilcke
    """
    
    def __init__(self, n_visivel: int, n_oculto: int):
        self.n_visivel = n_visivel
        self.n_oculto = n_oculto
        self.pesos = np.random.randn(n_visivel, n_oculto) * 0.01
        self.vies_visivel = np.zeros(n_visivel)
        self.vies_oculto = np.zeros(n_oculto)
    
    def _sigmoide(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def prop_cima(self, v: np.ndarray) -> np.ndarray:
        """Propagação: visível -> oculto."""
        return self._sigmoide(v @ self.pesos + self.vies_oculto)
    
    def prop_baixo(self, h: np.ndarray) -> np.ndarray:
        """Propagação: oculto -> visível."""
        return self._sigmoide(h @ self.pesos.T + self.vies_visivel)
    
    def amostrar_oculto(self, v: np.ndarray) -> np.ndarray:
        prob = self.prop_cima(v)
        return (np.random.rand(*prob.shape) < prob).astype(float)
    
    def amostrar_visivel(self, h: np.ndarray) -> np.ndarray:
        prob = self.prop_baixo(h)
        return (np.random.rand(*prob.shape) < prob).astype(float)
    
    def treinar(self, dados: np.ndarray, epocas: int = 100, taxa: float = 0.1, k: int = 1):
        """Treina usando Contrastive Divergence."""
        for _ in range(epocas):
            for v0 in dados:
                v0 = v0.reshape(1, -1)
                
                # Fase positiva
                h0_prob = self.prop_cima(v0)
                h0 = (np.random.rand(*h0_prob.shape) < h0_prob).astype(float)
                
                # Gibbs sampling
                hk = h0
                for _ in range(k):
                    vk_prob = self.prop_baixo(hk)
                    vk = (np.random.rand(*vk_prob.shape) < vk_prob).astype(float)
                    hk_prob = self.prop_cima(vk)
                    hk = (np.random.rand(*hk_prob.shape) < hk_prob).astype(float)
                
                # Atualização
                self.pesos += taxa * (v0.T @ h0_prob - vk.T @ hk_prob)
                self.vies_visivel += taxa * (v0 - vk).flatten()
                self.vies_oculto += taxa * (h0_prob - hk_prob).flatten()


# ==============================================
# ECHO STATE NETWORK (ESN)
# ==============================================

class RedeEchoState:
    """
    Echo State Network (Reservoir Computing).
    Autor: Luiz Tiago Wilcke
    """
    
    def __init__(self, dim_entrada: int, dim_reservatorio: int, dim_saida: int, raio_espectral: float = 0.9):
        self.dim_entrada = dim_entrada
        self.dim_reservatorio = dim_reservatorio
        self.dim_saida = dim_saida
        
        # Pesos de entrada
        self.Win = np.random.randn(dim_reservatorio, dim_entrada) * 0.1
        
        # Reservatório (matriz esparsa)
        W = np.random.randn(dim_reservatorio, dim_reservatorio)
        W[np.random.rand(*W.shape) > 0.1] = 0  # Esparsidade
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = (W / rho) * raio_espectral
        
        # Pesos de saída (treináveis)
        self.Wout = None
    
    def coletar_estados(self, entradas: np.ndarray) -> np.ndarray:
        """Coleta estados do reservatório."""
        seq_len = entradas.shape[0]
        estados = np.zeros((seq_len, self.dim_reservatorio))
        x = np.zeros(self.dim_reservatorio)
        
        for t in range(seq_len):
            x = np.tanh(self.Win @ entradas[t] + self.W @ x)
            estados[t] = x
        
        return estados
    
    def treinar(self, entradas: np.ndarray, saidas: np.ndarray, regularizacao: float = 1e-6):
        """Treina pesos de saída com regressão ridge."""
        estados = self.coletar_estados(entradas)
        
        # Regressão Ridge
        self.Wout = np.linalg.solve(
            estados.T @ estados + regularizacao * np.eye(self.dim_reservatorio),
            estados.T @ saidas
        )
    
    def prever(self, entradas: np.ndarray) -> np.ndarray:
        """Faz previsões."""
        estados = self.coletar_estados(entradas)
        return estados @ self.Wout
