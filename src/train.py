"""
===============================================================================
FASE 5 & 6 — Treino e Monitorização do Modelo Baseline (train.py)
===============================================================================

OBJETIVO:
    Treinar a CNN Baseline no conjunto de treino, avaliar no conjunto de validação
    a cada época, guardar o melhor modelo e gerar gráficos de evolução.

CONCEITOS DE MACHINE LEARNING (Müller & Guido, Cap. 2):
    1. Época (Epoch):
       - Uma passagem completa por todas as 4.983 imagens de treino.
    2. Perda de Treino vs. Perda de Validação (Training vs. Validation Loss):
       - Se a perda de treino desce e a de validação também desce: o modelo está a aprender!
       - Se a perda de treino desce mas a de validação começa a subir: OVERFITTING!
    3. Paragem Precoce (Early Stopping):
       - Monitoriza a perda de validação. Se ela não melhorar durante 3 épocas consecutivas,
         o treino pára automaticamente para não desperdiçar tempo nem entrar em overfitting.
    4. Guardar o Melhor Modelo (ModelCheckpoint):
       - Guarda no disco apenas a versão do modelo com o menor erro na validação.

===============================================================================
"""

import os
import sys

# Garantir UTF-8 no stdout/stderr no Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from data_loader import get_data_loaders, IMG_SIZE, BATCH_SIZE
from model import build_baseline_cnn

# ============================================================================
# CONFIGURAÇÃO SIMPLES
# ============================================================================
EPOCHS = 5             # 5 épocas para a baseline inicial (rápido e suficiente para referência)
LEARNING_RATE = 1e-3   # Taxa de aprendizagem do otimizador Adam (0.001)

MODELS_DIR = "models"
FIGURES_DIR = "results/figures"
METRICS_DIR = "results/metrics"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def plot_training_curves(history, output_path):
    """
    Desenha e guarda os gráficos de evolução de Loss e AUC ao longo das épocas.
    """
    epochs_range = range(1, len(history.history['loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Função de Perda (Binary Cross-Entropy)
    ax1 = axes[0]
    ax1.plot(epochs_range, history.history['loss'], 'b-o', label='Treino (Train Loss)')
    ax1.plot(epochs_range, history.history['val_loss'], 'r-s', label='Validação (Val Loss)')
    ax1.set_title('Evolução da Perda (Loss)', fontsize=12)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Binary Cross-Entropy')
    ax1.legend()
    ax1.grid(True)

    # Gráfico 2: Métrica de Discriminação (AUC - Area Under ROC Curve)
    ax2 = axes[1]
    ax2.plot(epochs_range, history.history['auc'], 'b-o', label='Treino (Train AUC)')
    ax2.plot(epochs_range, history.history['val_auc'], 'r-s', label='Validação (Val AUC)')
    ax2.set_title('Evolução do ROC-AUC', fontsize=12)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('AUC (0.0 a 1.0)')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle('Curvas de Aprendizagem — Modelo Baseline', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n[OK] Gráfico das curvas de treino guardado em: {output_path}")


def train_baseline(epochs=EPOCHS):
    print("=" * 70)
    print("INÍCIO DO TREINO DO MODELO BASELINE")
    print("=" * 70)

    # PASSO 1: Carregar os dados
    print("\n[Passo 1/4] A carregar os dados de treino e validação...")
    train_ds, val_ds, test_ds, encoder, info = get_data_loaders(batch_size=BATCH_SIZE, target_size=IMG_SIZE)
    print(f"  Treino:    {info['num_train']} imagens")
    print(f"  Validação: {info['num_val']} imagens")

    # PASSO 2: Construir a rede neuronal
    print("\n[Passo 2/4] A construir a CNN Baseline...")
    model = build_baseline_cnn(learning_rate=LEARNING_RATE)

    # PASSO 3: Configurar os mecanismos de proteção (Callbacks)
    print("\n[Passo 3/4] A configurar mecanismos de paragem e salvaguarda...")
    best_model_path = os.path.join(MODELS_DIR, "baseline_best.keras")

    callbacks = [
        # Guarda apenas o modelo da melhor época na validação
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        # Pára se a perda de validação não melhorar durante 2 épocas consecutivas
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduz o learning rate para metade se o progresso abrandar
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            verbose=1
        )
    ]

    # PASSO 4: Executar o treino
    print(f"\n[Passo 4/4] A treinar o modelo durante {epochs} épocas...")
    print("=" * 70)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Guardar métricas em formato JSON simples para histórico
    history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
    history_path = os.path.join(METRICS_DIR, "baseline_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=4)
    print(f"\n[OK] Histórico numérico guardado em: {history_path}")

    # Gerar e guardar o gráfico
    curves_path = os.path.join(FIGURES_DIR, "baseline_training_curves.png")
    plot_training_curves(history, curves_path)

    print("\n" + "=" * 70)
    print(f"[SUCESSO] Treino da Baseline concluído!")
    print(f"Melhor modelo guardado em: {best_model_path}")
    print("=" * 70)

    return model, history


if __name__ == "__main__":
    train_baseline()
