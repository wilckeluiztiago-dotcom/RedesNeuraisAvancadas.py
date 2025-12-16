"""
RedesNeuraisAvancadas - Visualização
Autor: Luiz Tiago Wilcke
"""

import numpy as np


def plotar_historico_treino(historico: dict, salvar: str = None):
    """Plota curvas de perda do treinamento."""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'perda_treino' in historico:
            ax.plot(historico['perda_treino'], label='Perda Treino', color='blue')
        if 'perda_validacao' in historico:
            ax.plot(historico['perda_validacao'], label='Perda Validação', color='orange')
        
        ax.set_xlabel('Época')
        ax.set_ylabel('Perda')
        ax.set_title('Histórico de Treinamento')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if salvar:
            plt.savefig(salvar, dpi=150, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("Matplotlib não instalado. Use: pip install matplotlib")


def plotar_gan_historico(historico: dict, salvar: str = None):
    """Plota curvas de perda do GAN."""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(historico['perda_d'], label='Discriminador', color='red')
        ax.plot(historico['perda_g'], label='Gerador', color='blue')
        
        ax.set_xlabel('Época')
        ax.set_ylabel('Perda')
        ax.set_title('Treinamento GAN')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if salvar:
            plt.savefig(salvar, dpi=150, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("Matplotlib não instalado.")


def plotar_som(som, dados: np.ndarray = None, salvar: str = None):
    """Plota mapa auto-organizável (Kohonen)."""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plota pesos como imagem
        pesos_norm = som.pesos.copy()
        pesos_norm = (pesos_norm - pesos_norm.min()) / (pesos_norm.max() - pesos_norm.min() + 1e-8)
        
        # Usa primeiras 3 dimensões como RGB se possível
        if som.dim_entrada >= 3:
            imagem = pesos_norm[:, :, :3]
        else:
            imagem = np.mean(pesos_norm, axis=2)
        
        ax.imshow(imagem)
        ax.set_title('Mapa Auto-Organizável de Kohonen')
        ax.axis('off')
        
        if salvar:
            plt.savefig(salvar, dpi=150, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("Matplotlib não instalado.")


def plotar_grafo(A: np.ndarray, X: np.ndarray = None, labels: np.ndarray = None, salvar: str = None):
    """Plota grafo com suas conexões."""
    try:
        import matplotlib.pyplot as plt
        
        n = A.shape[0]
        
        # Posições dos nós (layout circular simples)
        angulos = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos_x = np.cos(angulos)
        pos_y = np.sin(angulos)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Desenha arestas
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] > 0:
                    ax.plot([pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]], 'gray', alpha=0.5)
        
        # Desenha nós
        cores = labels if labels is not None else 'blue'
        ax.scatter(pos_x, pos_y, c=cores, s=200, zorder=5, cmap='tab10')
        
        for i in range(n):
            ax.annotate(str(i), (pos_x[i], pos_y[i]), ha='center', va='center', fontsize=8)
        
        ax.set_title('Visualização do Grafo')
        ax.axis('equal')
        ax.axis('off')
        
        if salvar:
            plt.savefig(salvar, dpi=150, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("Matplotlib não instalado.")


def imprimir_arquitetura(modelo):
    """Imprime arquitetura do modelo de forma visual."""
    print("\n" + "=" * 60)
    print(f"  ARQUITETURA: {modelo.nome}")
    print("=" * 60)
    
    for i, camada in enumerate(modelo.camadas):
        params = camada.contar_parametros()
        print(f"  │")
        print(f"  ├─ [{i+1}] {camada.nome}")
        if params > 0:
            print(f"  │      └─ Parâmetros: {params:,}")
    
    print(f"  │")
    print(f"  └─ SAÍDA")
    print("=" * 60)
    
    total = sum(c.contar_parametros() for c in modelo.camadas)
    print(f"  Total de Parâmetros: {total:,}")
    print("=" * 60 + "\n")
