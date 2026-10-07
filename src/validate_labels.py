"""
===============================================================================
FASE 2 — Validação e Teste da Codificação Multi-Hot
===============================================================================

OBJETIVO:
    Testar rigorosamente o FoodLabelEncoder em casos sintéticos e em imagens
    reais do FoodSeg103 para garantir 100% de fiabilidade matemática antes
    de qualquer etapa de treino.

===============================================================================
"""

import os
import sys

# Garantir UTF-8 no stdout/stderr no Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from datasets import load_dataset
from preprocessing import FoodLabelEncoder, NUM_CLASSES


def test_synthetic_cases(encoder):
    """
    Testa casos controlados para validar o comportamento dos cantos (edge cases).
    """
    print("=" * 70)
    print("TESTE 1: CASOS DE TESTE CONTROLADOS (EDGE CASES)")
    print("=" * 70)

    # Caso 1: Prato típico [0, 48, 66, 90] (background, chicken duck, rice, snow peas)
    raw = [0, 48, 66, 90]
    vec = encoder.encode(raw)
    decoded = encoder.decode(vec)
    print(f"\nCaso 1: Prato misto")
    print(f"  Entrada bruta:          {raw}")
    print(f"  Vetor forma / tipo:     {vec.shape} / {vec.dtype}")
    print(f"  Número de 1s no vetor:  {int(vec.sum())}")
    print(f"  Posições ativas (1.0):  {[i for i, v in enumerate(vec) if v == 1.0]}")
    print(f"  Alimentos descodificados: {[name for name, _ in decoded]}")
    assert vec.shape == (NUM_CLASSES,), "Forma do vetor incorreta!"
    assert int(vec.sum()) == 3, "Deveriam ser exatamente 3 classes ativas (background excluído)!"
    assert [name for name, _ in decoded] == ["chicken duck", "rice", "snow peas"]
    print("  -> [OK] Caso 1 validado com sucesso!")

    # Caso 2: Prato que contém classe 103 (other ingredients) e classe 0 (background)
    raw_with_103 = [0, 58, 84, 103]
    vec_103 = encoder.encode(raw_with_103)
    decoded_103 = encoder.decode(vec_103)
    print(f"\nCaso 2: Presenca de 'other ingredients' (ID 103) e background (ID 0)")
    print(f"  Entrada bruta:          {raw_with_103}")
    print(f"  Número de 1s no vetor:  {int(vec_103.sum())}")
    print(f"  Alimentos descodificados: {[name for name, _ in decoded_103]}")
    assert int(vec_103.sum()) == 2, "Deveriam ser exatamente 2 classes ativas (0 e 103 excluídos)!"
    assert [name for name, _ in decoded_103] == ["bread", "carrot"]
    print("  -> [OK] Caso 2 validado com sucesso! (ID 0 e ID 103 corretamente ignorados)")

    # Caso 3: Imagem que conteria apenas background (ID 0)
    raw_only_bg = [0]
    vec_only_bg = encoder.encode(raw_only_bg)
    print(f"\nCaso 3: Apenas background")
    print(f"  Entrada bruta:          {raw_only_bg}")
    print(f"  Número de 1s no vetor:  {int(vec_only_bg.sum())}")
    assert int(vec_only_bg.sum()) == 0, "Deveriam ser 0 classes ativas!"
    print("  -> [OK] Caso 3 validado com sucesso! (vetor totalmente preenchido com 0s)")


def test_real_dataset_samples(encoder, num_samples=10):
    """
    Testa a transformação em amostras reais retiradas diretamente do FoodSeg103.
    """
    print("\n" + "=" * 70)
    print(f"TESTE 2: VALIDAÇÃO EM {num_samples} AMOSTRAS REAIS DO DATASET")
    print("=" * 70)

    ds = load_dataset("json9473/FoodSeg103")

    for i in range(num_samples):
        sample = ds["train"][i]
        img_id = sample["id"]
        raw_classes = sample["classes_on_image"]

        # Codificar para vetor multi-hot
        vec = encoder.encode(raw_classes)

        # Descodificar de volta
        decoded = encoder.decode(vec)
        decoded_names = [name for name, _ in decoded]

        # Nomes esperados pelo filtro manual (excluindo 0 e 103)
        expected_names = [encoder.id_to_name[cid] for cid in raw_classes if 1 <= cid <= 102]

        # Ordenar ambas as listas para comparação estrita
        assert sorted(decoded_names) == sorted(expected_names), (
            f"Discrepancia na imagem ID {img_id}!\n"
            f"Esperado: {expected_names}\nObtido: {decoded_names}"
        )

        print(f"\n[Amostra {i+1}/{num_samples}] Imagem ID: {img_id}")
        print(f"  classes_on_image:       {raw_classes}")
        print(f"  Alimentos esperados:    {expected_names}")
        print(f"  Alimentos no multi-hot: {decoded_names}")
        print(f"  Posicoes (indices):     {[encoder.name_to_idx[n] for n in decoded_names]}")
        print(f"  Soma do vetor:          {int(vec.sum())} / {len(expected_names)} esperados")

    print("\n" + "=" * 70)
    print(f"[SUCESSO] Todas as {num_samples} amostras reais foram convertidas e validadas perfeitamente!")
    print("=" * 70)


if __name__ == "__main__":
    encoder = FoodLabelEncoder()
    print(f"Encoder inicializado com {len(encoder.class_names)} classes.")
    print(f"Primeiras 5 classes: {encoder.class_names[:5]}")
    print(f"Ultimas 5 classes:   {encoder.class_names[-5:]}")

    test_synthetic_cases(encoder)
    test_real_dataset_samples(encoder, num_samples=10)
