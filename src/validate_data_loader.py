"""
===============================================================================
FASE 3 — Validação do Pipeline de Dados e Pré-processamento
===============================================================================

OBJETIVO:
    Testar o data_loader.py, verificar formas de tensores (shapes), intervalo
    de píxeis normalizados [0.0, 1.0], integridade dos lotes (batches) e
    guardar uma figura com pré-visualização de um lote real.

===============================================================================
"""

import os
import sys

# Garantir UTF-8 no stdout/stderr no Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from data_loader import get_data_loaders, NUM_CLASSES, IMG_SIZE, BATCH_SIZE


def validate_pipeline():
    print("=" * 70)
    print("FASE 3: VALIDAÇÃO DO PIPELINE DE DADOS (DATA LOADER)")
    print("=" * 70)

    print("\nA inicializar os Data Loaders...")
    train_ds, val_ds, test_ds, encoder, info = get_data_loaders(batch_size=8, target_size=IMG_SIZE)

    print("\nInformacao das Divisoes:")
    print(f"  Treino:     {info['num_train']} imagens")
    print(f"  Validacao:  {info['num_val']} imagens")
    print(f"  Teste:      {info['num_test']} imagens")
    print(f"  Total:      {info['num_train'] + info['num_val'] + info['num_test']} imagens")
    assert info['num_train'] + info['num_val'] + info['num_test'] == 7118, "Total deve ser 7118!"

    # 1. Testar primeiro lote do Treino
    print("\nA extrair um lote (batch) do conjunto de treino...")
    for images_batch, labels_batch in train_ds.take(1):
        imgs = images_batch.numpy()
        lbls = labels_batch.numpy()

        print(f"  Forma do lote de imagens: {imgs.shape} (Esperado: (8, 224, 224, 3))")
        print(f"  Forma do lote de labels:  {lbls.shape} (Esperado: (8, 102))")
        print(f"  Tipo dos dados:           {imgs.dtype} / {lbls.dtype}")
        print(f"  Intervalo dos píxeis:     Min = {imgs.min():.3f}, Max = {imgs.max():.3f}, Media = {imgs.mean():.3f}")

        assert imgs.shape == (8, 224, 224, 3), f"Forma incorreta: {imgs.shape}"
        assert lbls.shape == (8, NUM_CLASSES), f"Forma incorreta de labels: {lbls.shape}"
        assert 0.0 <= imgs.min() and imgs.max() <= 1.0, "Píxeis devem estar estritamente no intervalo [0.0, 1.0]!"
        print("  -> [OK] Formas, tipos e normalização de píxeis validados com sucesso!")

        # 2. Visualizar as 8 imagens do lote e descodificar os labels
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        print("\nDescodificação das amostras do lote:")
        for i in range(8):
            decoded = encoder.decode(lbls[i])
            food_names = [name for name, _ in decoded]

            ax = axes[i]
            ax.imshow(imgs[i])
            title_text = f"Exemplo {i+1}:\n" + "\n".join(food_names)
            ax.set_title(title_text, fontsize=8)
            ax.axis('off')

            print(f"  [Amostra {i+1}] Alimentos ativos ({int(lbls[i].sum())}): {', '.join(food_names)}")

        plt.suptitle("Pré-visualização de um Batch Normalizado (224x224, [0, 1])", fontsize=13)
        plt.tight_layout()
        preview_path = "results/figures/batch_preview.png"
        plt.savefig(preview_path, dpi=150)
        plt.close()
        print(f"\n[OK] Gráfico do lote guardado em: {preview_path}")

    # 3. Testar primeiro lote da Validação e Teste
    print("\nA validar extração de lotes dos conjuntos de Validação e Teste...")
    for images_val, labels_val in val_ds.take(1):
        assert images_val.shape == (8, 224, 224, 3)
        assert labels_val.shape == (8, NUM_CLASSES)
        print("  -> [OK] Lote de Validação extraído com sucesso!")

    for images_test, labels_test in test_ds.take(1):
        assert images_test.shape == (8, 224, 224, 3)
        assert labels_test.shape == (8, NUM_CLASSES)
        print("  -> [OK] Lote de Teste extraído com sucesso!")

    print("\n" + "=" * 70)
    print("[SUCESSO] Pipeline de Dados (FASE 3) 100% validado e pronto para a Baseline!")
    print("=" * 70)


if __name__ == "__main__":
    validate_pipeline()
