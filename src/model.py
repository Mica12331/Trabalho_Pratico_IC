"""
===============================================================================
FASE 4 — Arquitetura da Rede Neuronal Baseline (model.py)
===============================================================================

OBJETIVO:
    Construir uma Convolutional Neural Network (CNN) base para classificação
    multi-label de 102 classes de alimentos.

JUSTIFICAÇÃO DOS COMPONENTES E HIPERPARÂMETROS:
    1. Entrada (224, 224, 3):
       - Imagens RGB redimensionadas na Fase 3.
    2. Camadas Conv2D (filtros 32 -> 64 -> 128 com kernel 3x3):
       - Extração hierárquica de características: arestas -> texturas -> formas.
    3. Função de Ativação ReLU (Rectified Linear Unit):
       - f(x) = max(0, x). Introduz não-linearidade sem sofrer de saturação de gradientes
         (conceito explicado no Cap. 2 do livro de Müller & Guido para redes neuronais).
    4. MaxPooling2D (pool 2x2):
       - Reduz a resolução espacial para metade a cada bloco, diminuindo os parâmetros
         e conferindo invariância a pequenas translações no prato.
    5. GlobalAveragePooling2D:
       - Em vez de um Flatten tradicional (que geraria centenas de milhares de conexões
         e causaria overfitting severo), tira a média espacial de cada canal.
    6. Dropout (0.3):
       - Técnica de regularização: desliga aleatoriamente 30% dos neurónios durante o treino,
         forçando a rede a aprender representações redundantes e robustas (Livro, Cap. 2).
    7. Camada de Saída Dense(102, activation='sigmoid'):
       - 102 neurónios independentes (um por alimento). O Sigmoid mapeia a saída de cada
         neurónio para o intervalo [0.0, 1.0], permitindo múltiplas classes ativas em simultâneo.
    8. Função de Perda (Binary Cross-Entropy):
       - Mede o erro de cada uma das 102 decisões binárias de forma independente.
    9. Otimizador Adam (Adaptive Moment Estimation):
       - Ajusta taxas de aprendizagem individuais para cada peso com momentum adaptativo.

===============================================================================
"""

import os
import sys

# Garantir UTF-8 no stdout/stderr no Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics

NUM_CLASSES = 102
IMG_SHAPE = (224, 224, 3)


def build_baseline_cnn(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES, learning_rate=1e-3):
    """
    Constrói e compila a CNN Baseline para classificação multi-label.

    Parâmetros:
        input_shape (tuple): forma da imagem de entrada (H, W, C)
        num_classes (int): número de classes de saída (102)
        learning_rate (float): taxa de aprendizagem inicial do Adam

    Retorna:
        tf.keras.Model: modelo compilado pronto para treino
    """
    model = models.Sequential(name="Baseline_CNN_FoodSeg102")

    # Camada de Entrada
    model.add(layers.Input(shape=input_shape, name="input_image"))

    # Bloco Convolucional 1: Extração de arestas e cores básicas
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1"))
    model.add(layers.MaxPooling2D((2, 2), name="pool1"))

    # Bloco Convolucional 2: Extração de texturas de alimentos
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2"))
    model.add(layers.MaxPooling2D((2, 2), name="pool2"))

    # Bloco Convolucional 3: Extração de padrões e formas complexas
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3"))
    model.add(layers.MaxPooling2D((2, 2), name="pool3"))

    # Transição: Global Average Pooling (leve e resistente a overfitting)
    model.add(layers.GlobalAveragePooling2D(name="gap"))

    # Camada Densa Intermédia de Combinação
    model.add(layers.Dense(128, activation="relu", name="dense_features"))
    model.add(layers.Dropout(0.3, name="dropout_regularization"))

    # Camada de Saída Multi-Label: 102 neurónios independentes com Sigmoid
    model.add(layers.Dense(num_classes, activation="sigmoid", name="multi_label_output"))

    # Compilação do Modelo
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryAccuracy(name="binary_accuracy"),
            metrics.AUC(name="auc", multi_label=True)
        ]
    )

    return model


if __name__ == "__main__":
    print("=" * 70)
    print("SUMÁRIO DA ARQUITETURA DA CNN BASELINE")
    print("=" * 70)

    model = build_baseline_cnn()
    model.summary()

    # Validação da forma de saída para um lote fictício
    dummy_input = tf.random.uniform((4, 224, 224, 3), minval=0.0, maxval=1.0)
    dummy_output = model(dummy_input)

    print("\nTeste com Lote Simulado:")
    print(f"  Entrada simulada: {dummy_input.shape}")
    print(f"  Saída obtida:     {dummy_output.shape} (Esperado: (4, 102))")
    print(f"  Intervalo saída:  Min = {float(tf.reduce_min(dummy_output)):.4f}, Max = {float(tf.reduce_max(dummy_output)):.4f}")
    assert dummy_output.shape == (4, NUM_CLASSES), "Forma da saída incorreta!"
    print("  -> [OK] Camada Sigmoid produz probabilidades válidas entre 0.0 e 1.0!")
