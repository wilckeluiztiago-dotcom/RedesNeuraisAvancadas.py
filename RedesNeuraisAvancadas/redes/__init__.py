"""
RedesNeuraisAvancadas - Biblioteca de Redes Neurais
Autor: Luiz Tiago Wilcke

Uma biblioteca completa de redes neurais implementadas do zero em Python/NumPy.
Contém 20+ arquiteturas diferentes, todas com variáveis em português.
"""

__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

# Base
from .base import Camada, CamadaDensa, CamadaConvolucional, CamadaRecorrente, RedeNeural

# Ativações
from .ativacoes import (
    ReLU, LeakyReLU, ELU, SELU, Sigmoide, Tanh, Softmax, Swish, GELU, Mish,
    CamadaReLU, CamadaSigmoide, CamadaTanh, CamadaSoftmax, CamadaSwish, CamadaGELU,
    obter_ativacao
)

# Otimizadores
from .otimizadores import SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, NAdam, RAdam, obter_otimizador

# Funções de Perda
from .perdas import (
    ErroQuadraticoMedio, ErroAbsolutoMedio, Huber,
    EntropiaCruzadaBinaria, EntropiaCruzadaCategorica, EntropiaCruzadaSparse,
    KLDivergencia, HingeLoss, FocalLoss, CosineSimilarityLoss,
    obter_perda
)

# Arquiteturas
from .mlp import Perceptron, MLP
from .cnn import CNN, CamadaPooling, CamadaFlatten, BatchNormalizacao
from .lstm import CamadaLSTM, CamadaGRU
from .autoencoder import Autoencoder, VAE
from .gan import GAN, Gerador, Discriminador
from .transformer import Transformer, AtencaoMultiCabeca, BlocoTransformer, codificacao_posicional
from .especiais import (
    ResNet, BlocoResidual,
    BlocoDenso,
    RedeHopfield,
    MapaKohonen,
    RedeRBF, CamadaRBF,
    MaquinaBoltzmann,
    RedeEchoState
)
from .grafos import CapsuleNetwork, CamadaCapsule, RedeGrafoNeuronal, CamadaGrafo, CamadaAtencaoGrafo


# Lista de todas as redes disponíveis
REDES_DISPONIVEIS = [
    "Perceptron",
    "MLP (Multi-Layer Perceptron)",
    "CNN (Rede Convolucional)",
    "RNN (Rede Recorrente)",
    "LSTM (Long Short-Term Memory)",
    "GRU (Gated Recurrent Unit)",
    "Autoencoder",
    "VAE (Variational Autoencoder)",
    "GAN (Generative Adversarial Network)",
    "Transformer",
    "ResNet (Rede Residual)",
    "DenseNet (Bloco Denso)",
    "Rede de Hopfield",
    "Mapa de Kohonen (SOM)",
    "Rede RBF (Radial Basis Function)",
    "Máquina de Boltzmann Restrita",
    "Echo State Network",
    "Capsule Network",
    "Graph Neural Network (GNN)",
    "Graph Attention Network (GAT)"
]


def listar_redes():
    """Lista todas as redes neurais disponíveis."""
    print(f"\n{'='*50}")
    print(f"RedesNeuraisAvancadas v{__version__}")
    print(f"Autor: {__author__}")
    print(f"{'='*50}")
    print(f"\n{len(REDES_DISPONIVEIS)} Redes Neurais Disponíveis:\n")
    for i, rede in enumerate(REDES_DISPONIVEIS, 1):
        print(f"  {i:2d}. {rede}")
    print()
