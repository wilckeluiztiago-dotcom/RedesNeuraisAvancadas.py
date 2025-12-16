# RedesNeuraisAvancadas

**Autor:** Luiz Tiago Wilcke

Uma biblioteca completa de redes neurais implementadas do zero em Python/NumPy. Contém 20+ arquiteturas diferentes.

## 🧠 Redes Neurais Disponíveis

| # | Rede | Descrição |
|---|------|-----------|
| 1 | **Perceptron** | Neurônio artificial básico |
| 2 | **MLP** | Multi-Layer Perceptron |
| 3 | **CNN** | Rede Convolucional |
| 4 | **RNN** | Rede Recorrente |
| 5 | **LSTM** | Long Short-Term Memory |
| 6 | **GRU** | Gated Recurrent Unit |
| 7 | **Autoencoder** | Compressão/Reconstrução |
| 8 | **VAE** | Variational Autoencoder |
| 9 | **GAN** | Generative Adversarial Network |
| 10 | **Transformer** | Mecanismo de Atenção |
| 11 | **ResNet** | Rede Residual |
| 12 | **DenseNet** | Blocos Densos |
| 13 | **Hopfield** | Memória Associativa |
| 14 | **Kohonen SOM** | Mapa Auto-Organizável |
| 15 | **RBF** | Radial Basis Function |
| 16 | **Boltzmann** | Máquina de Boltzmann Restrita |
| 17 | **ESN** | Echo State Network |
| 18 | **Capsule** | Capsule Network |
| 19 | **GNN** | Graph Neural Network |
| 20 | **GAT** | Graph Attention Network |

## 📁 Estrutura do Projeto

```
RedesNeuraisAvancadas/
├── redes/
│   ├── __init__.py      # Exportações
│   ├── base.py          # Classes fundamentais
│   ├── ativacoes.py     # 12 funções de ativação
│   ├── otimizadores.py  # 8 otimizadores
│   ├── perdas.py        # 10 funções de perda
│   ├── mlp.py           # Perceptron, MLP
│   ├── cnn.py           # CNN, Pooling, BatchNorm
│   ├── lstm.py          # LSTM, GRU
│   ├── autoencoder.py   # Autoencoder, VAE
│   ├── gan.py           # GAN
│   ├── transformer.py   # Transformer, Atenção
│   ├── especiais.py     # ResNet, Hopfield, SOM, RBF, ESN
│   └── grafos.py        # Capsule, GNN, GAT
├── utils/
│   ├── dados.py         # Geração de dados
│   └── visualizacao.py  # Gráficos
├── principal.py         # Demo completa
└── README.md
```

## 🚀 Início Rápido

### Instalação

```bash
# Clone o repositório
cd RedesNeuraisAvancadas

# Apenas NumPy é necessário
pip install numpy
```

### Exemplo: MLP para Classificação

```python
from redes import MLP, Adam, EntropiaCruzadaCategorica
from utils import gerar_dados_classificacao, dividir_dados, acuracia

# Gera dados
X, y = gerar_dados_classificacao(n_amostras=1000, n_classes=3)
X_treino, X_teste, y_treino, y_teste = dividir_dados(X, y)

# Cria modelo
modelo = MLP.criar_classificador(entrada=10, classes=3, ocultas=[64, 32])
modelo.compilar(Adam(taxa_aprendizado=0.01), EntropiaCruzadaCategorica())

# Treina
modelo.treinar(X_treino, y_treino, epocas=100, tamanho_lote=32)

# Avalia
pred = modelo.prever(X_teste)
print(f"Acurácia: {acuracia(y_teste, pred):.2%}")
```

### Exemplo: GAN

```python
from redes import GAN
import numpy as np

# Dados reais
dados = np.random.randn(1000, 10) * 2 + 5

# Treina GAN
gan = GAN(dim_latente=20, dim_dados=10)
gan.treinar(dados, epocas=100)

# Gera novas amostras
novas = gan.gerar(100)
```

### Exemplo: Rede de Hopfield

```python
from redes import RedeHopfield
import numpy as np

# Padrões para memorizar
padroes = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]])

rede = RedeHopfield(tamanho=4)
rede.treinar(padroes)

# Recupera padrão corrompido
entrada = np.array([1, -1, 1, 1])  # Com ruído
recuperado = rede.recuperar(entrada)
```

## 🛠️ Componentes

### Funções de Ativação
- ReLU, LeakyReLU, ELU, SELU
- Sigmoide, Tanh, Softmax
- Swish, GELU, Mish

### Otimizadores
- SGD (com momentum)
- Adam, AdamW, NAdam, RAdam
- RMSprop, Adagrad, Adadelta

### Funções de Perda
- MSE, MAE, Huber
- Entropia Cruzada (binária, categórica)
- KL Divergence, Hinge, Focal Loss

## ▶️ Executar Demo

```bash
python principal.py
```

## 📋 Requisitos

- Python 3.8+
- NumPy
- Matplotlib (opcional, para visualização)

## 📝 Licença

MIT License

## 👤 Autor

**Luiz Tiago Wilcke**

---

*Biblioteca desenvolvida para fins educacionais, demonstrando implementação de redes neurais do zero.*
