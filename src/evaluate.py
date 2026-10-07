"""
===============================================================================
FASE 6 — Avaliação Rigorosa do Modelo no Conjunto de Teste (evaluate.py)
===============================================================================

OBJETIVO:
    Avaliar o modelo treinado (baseline_best.keras) no conjunto de TESTE independente
    (1.068 imagens nunca vistas).

MÉTRICAS EXIGIDAS PELO ENUNCIADO DO TRABALHO PRÁTICO:
    1. Accuracy (Explicando porque 96% de Binary Accuracy é uma ilusão devido ao desbalanceamento).
    2. Sensibilidade (Recall): Entre os alimentos reais, quantos foram detetados?
    3. Especificidade (Specificity): Entre os alimentos ausentes, quantos foram rejeitados?
    4. F-measure (F1-score): Média harmónica entre Precisão e Sensibilidade (Micro e Macro).
    5. AUC (Area Under the ROC Curve): Capacidade de discriminação global.
    6. Matriz de Confusão: Agregação de TP, FP, TN, FN para todas as 102 classes.

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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, multilabel_confusion_matrix
)

from data_loader import get_data_loaders, NUM_CLASSES, IMG_SIZE, BATCH_SIZE


def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Calcula as métricas exigidas pelo enunciado usando um limiar (threshold) de decisão.
    """
    y_pred_bin = (y_pred_probs >= threshold).astype(np.float32)

    # 1. Accuracy
    # - Subset Accuracy: acerto exato de todos os alimentos da imagem (muito exigente)
    subset_acc = accuracy_score(y_true, y_pred_bin)
    # - Binary / Hamming Accuracy: percentagem global de 0s e 1s acertados
    hamming_acc = np.mean(y_true == y_pred_bin)

    # 2. Sensibilidade / Recall
    recall_micro = recall_score(y_true, y_pred_bin, average="micro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred_bin, average="macro", zero_division=0)

    # 3. Precisão
    prec_micro = precision_score(y_true, y_pred_bin, average="micro", zero_division=0)
    prec_macro = precision_score(y_true, y_pred_bin, average="macro", zero_division=0)

    # 4. F1-Score
    f1_micro = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)

    # 5. AUC (ROC-AUC)
    try:
        auc_micro = roc_auc_score(y_true, y_pred_probs, average="micro")
        auc_macro = roc_auc_score(y_true, y_pred_probs, average="macro")
    except Exception:
        auc_micro = 0.5
        auc_macro = 0.5

    # 6. Matriz de Confusão Agregada e Especificidade
    # multilabel_confusion_matrix devolve uma matriz 2x2 para cada classe:
    # [[TN, FP], [FN, TP]]
    mcm = multilabel_confusion_matrix(y_true, y_pred_bin)
    tn = mcm[:, 0, 0].sum()
    fp = mcm[:, 0, 1].sum()
    fn = mcm[:, 1, 0].sum()
    tp = mcm[:, 1, 1].sum()

    # Especificidade = TN / (TN + FP)
    specificity_micro = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics_dict = {
        "threshold": float(threshold),
        "subset_accuracy": float(subset_acc),
        "hamming_accuracy": float(hamming_acc),
        "precision_micro": float(prec_micro),
        "precision_macro": float(prec_macro),
        "recall_micro": float(recall_micro),
        "recall_macro": float(recall_macro),
        "specificity_micro": float(specificity_micro),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "auc_micro": float(auc_micro),
        "auc_macro": float(auc_macro),
        "confusion_matrix": {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn)
        }
    }

    return metrics_dict


def plot_confusion_matrix_summary(cm_dict, output_path):
    """
    Desenha um gráfico com a Matriz de Confusão Global Agregada (TP, FP, FN, TN).
    """
    tp = cm_dict["TP"]
    fp = cm_dict["FP"]
    tn = cm_dict["TN"]
    fn = cm_dict["FN"]

    matrix = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Previsto: Ausente (0)", "Previsto: Presente (1)"],
        yticklabels=["Real: Ausente (0)", "Real: Presente (1)"]
    )
    plt.title("Matriz de Confusão Agregada (Conjunto de Teste - 102 Classes)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Gráfico da Matriz de Confusão guardado em: {output_path}")


def evaluate_model(model_path="models/baseline_best.keras"):
    print("=" * 70)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE INDEPENDENTE")
    print("=" * 70)

    # 1. Carregar o modelo treinado
    print(f"\n[1/4] A carregar o modelo de: {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # 2. Carregar o conjunto de teste
    print("\n[2/4] A carregar o conjunto de Teste (1.068 imagens)...")
    _, _, test_ds, encoder, info = get_data_loaders(batch_size=BATCH_SIZE, target_size=IMG_SIZE)

    # 3. Gerar previsões para todo o conjunto de teste
    print("\n[3/4] A calcular previsões da rede para todo o teste...")
    y_true_list = []
    y_pred_list = []

    for images_batch, labels_batch in test_ds:
        preds = model(images_batch, training=False)
        y_pred_list.append(preds.numpy())
        y_true_list.append(labels_batch.numpy())

    y_true = np.vstack(y_true_list)
    y_pred_probs = np.vstack(y_pred_list)

    print(f"  Total de amostras avaliadas: {y_true.shape[0]}")
    print(f"  Forma dos rótulos reais:      {y_true.shape}")
    print(f"  Forma das previsões da rede:  {y_pred_probs.shape}")

    # 4. Calcular métricas para threshold padrão (0.5) e threshold ajustado (0.2)
    print("\n[4/4] A calcular métricas oficiais do enunciado...")

    metrics_05 = calculate_metrics(y_true, y_pred_probs, threshold=0.5)
    metrics_02 = calculate_metrics(y_true, y_pred_probs, threshold=0.2)

    # Mostrar resultados no terminal
    print("\n" + "=" * 70)
    print("RESULTADOS OBTIDOS NO TEST SET (THRESHOLD = 0.5)")
    print("=" * 70)
    print(f"  Binary / Hamming Accuracy: {metrics_05['hamming_accuracy'] * 100:.2f}%")
    print(f"  Subset Accuracy:           {metrics_05['subset_accuracy'] * 100:.2f}%")
    print(f"  Sensibilidade (Recall):    Micro = {metrics_05['recall_micro']:.4f} | Macro = {metrics_05['recall_macro']:.4f}")
    print(f"  Especificidade:            Micro = {metrics_05['specificity_micro']:.4f}")
    print(f"  Precisão (Precision):      Micro = {metrics_05['precision_micro']:.4f} | Macro = {metrics_05['precision_macro']:.4f}")
    print(f"  F1-Score (F-measure):      Micro = {metrics_05['f1_micro']:.4f} | Macro = {metrics_05['f1_macro']:.4f}")
    print(f"  AUC (Area Under ROC):      Micro = {metrics_05['auc_micro']:.4f} | Macro = {metrics_05['auc_macro']:.4f}")
    print(f"  Matriz de Confusão:")
    print(f"    Verdadeiros Positivos (TP): {metrics_05['confusion_matrix']['TP']}")
    print(f"    Falsos Positivos (FP):      {metrics_05['confusion_matrix']['FP']}")
    print(f"    Verdadeiros Negativos (TN): {metrics_05['confusion_matrix']['TN']}")
    print(f"    Falsos Negativos (FN):      {metrics_05['confusion_matrix']['FN']}")

    print("\n" + "=" * 70)
    print("RESULTADOS COMPARATIVOS COM THRESHOLD AJUSTADO = 0.2")
    print("=" * 70)
    print(f"  F1-Score Micro (th=0.2):   {metrics_02['f1_micro']:.4f}")
    print(f"  Sensibilidade (th=0.2):    {metrics_02['recall_micro']:.4f}")
    print(f"  Precisão (th=0.2):         {metrics_02['precision_micro']:.4f}")

    # Guardar métricas num ficheiro JSON
    results_path = "results/metrics/baseline_test_metrics.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"threshold_0.5": metrics_05, "threshold_0.2": metrics_02}, f, indent=4)
    print(f"\n[OK] Métricas salvas em: {results_path}")

    # Guardar gráfico da matriz de confusão
    cm_path = "results/figures/baseline_confusion_matrix.png"
    plot_confusion_matrix_summary(metrics_05['confusion_matrix'], cm_path)

    return metrics_05, metrics_02


if __name__ == "__main__":
    evaluate_model()
