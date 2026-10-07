"""
===============================================================================
FASE 3 — Pipeline de Dados e Pré-processamento de Imagens (data_loader.py)
===============================================================================

OBJETIVO:
    Carregar, redimensionar, normalizar e agrupar em lotes (batches) as imagens
    do FoodSeg103 e os respetivos vetores multi-hot para treino, validação e teste.

DECISÕES TÉCNICAS E CONCEITOS DE MACHINE LEARNING:
    1. Dimensões de Entrada (IMG_SIZE = 224x224):
       - As imagens originais variam entre 113x122 e 5312x4208 pixels (analisado na Fase 1).
       - Redes convolucionais requerem dimensões uniformes (H x W x C).
       - 224x224 é a resolução padrão da literatura (ImageNet), equilibrando detalhe
         visual de ingredientes e eficiência computacional.
       - [Nota]: Esta é uma decisão técnica de Visão Computacional, não coberta no livro.

    2. Normalização dos Píxeis (Escalamento [0.0, 1.0]):
       - [Livro de Referência: Müller & Guido, Cap. 3 — Preprocessing and Scaling]:
         Redes neuronais convergem muito melhor quando as features de entrada estão
         numa escala restrita (como [0, 1]). Píxeis originais estão em [0, 255].
         Dividir por 255.0 evita gradientes gigantescos e acelera a convergência.

    3. Tamanho do Lote (BATCH_SIZE = 32):
       - Mini-batch Stochastic Gradient Descent: equilibra estabilidade do gradiente
         e utilização de memória RAM/VRAM.

    4. Divisão dos Dados (Splits):
       - Train: 4.983 imagens (100% do split de treino original).
       - Validação: 1.067 imagens (50% do split de validação original, seed=42).
       - Teste: 1.068 imagens (50% restantes, totalmente isolados para avaliação final).

    5. Pipeline sob demanda (tf.data.Dataset):
       - Carregamento iterativo por lotes (evita sobrecarregar a memória RAM).

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
import tensorflow as tf
from datasets import load_dataset
from preprocessing import FoodLabelEncoder, NUM_CLASSES
from PIL import Image

# ============================================================================
# HIPERPARÂMETROS DO PRÉ-PROCESSAMENTO
# ============================================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42


def get_dataset_splits():
    """
    Carrega o FoodSeg103 e cria as 3 divisões: Train, Validation e Test.

    Retorna:
        train_data, val_data, test_data (listas de dicionários do dataset)
    """
    raw_dataset = load_dataset("json9473/FoodSeg103")
    train_data = raw_dataset["train"]

    # Dividir a partição 'validation' original (2.135 imagens) em 50% Val e 50% Test
    val_raw = raw_dataset["validation"]
    total_val_raw = len(val_raw)

    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(total_val_raw)

    split_point = total_val_raw // 2  # 1067 para val, 1068 para test
    val_indices = indices[:split_point]
    test_indices = indices[split_point:]

    # Criar subconjuntos indexados
    val_data = val_raw.select(val_indices)
    test_data = val_raw.select(test_indices)

    return train_data, val_data, test_data


def process_image(pil_img, target_size=IMG_SIZE):
    """
    Pré-processa uma única imagem PIL:
    1. Converte para RGB (caso venha em escala de cinza ou RGBA).
    2. Redimensiona para target_size (224x224) usando interpolação bilinear.
    3. Converte para array float32 e normaliza os píxeis para [0.0, 1.0].
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(pil_img, dtype=np.float32) / 255.0
    return img_array


def create_data_generator(data_split, encoder, target_size=IMG_SIZE, shuffle=False):
    """
    Cria um gerador Python para iterar sobre imagens e vetores multi-hot.
    """
    def generator():
        num_samples = len(data_split)
        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            sample = data_split[int(idx)]
            img_array = process_image(sample["image"], target_size=target_size)
            multi_hot = encoder.encode(sample["classes_on_image"])
            yield img_array, multi_hot

    return generator


def build_tf_dataset(data_split, encoder, batch_size=BATCH_SIZE, target_size=IMG_SIZE, shuffle=False):
    """
    Constrói um tf.data.Dataset de alto desempenho pronto para consumo pela rede neuronal.
    """
    generator = create_data_generator(data_split, encoder, target_size=target_size, shuffle=shuffle)

    output_signature = (
        tf.TensorSpec(shape=(target_size[0], target_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=128)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def get_data_loaders(batch_size=BATCH_SIZE, target_size=IMG_SIZE):
    """
    Função principal de conveniência que orquestra tudo e devolve os 3 datasets.

    Retorna:
        train_ds, val_ds, test_ds, encoder, splits_info
    """
    encoder = FoodLabelEncoder()
    train_data, val_data, test_data = get_dataset_splits()

    splits_info = {
        "num_train": len(train_data),
        "num_val": len(val_data),
        "num_test": len(test_data),
        "img_size": target_size,
        "batch_size": batch_size,
        "num_classes": NUM_CLASSES
    }

    train_ds = build_tf_dataset(train_data, encoder, batch_size=batch_size, target_size=target_size, shuffle=True)
    val_ds = build_tf_dataset(val_data, encoder, batch_size=batch_size, target_size=target_size, shuffle=False)
    test_ds = build_tf_dataset(test_data, encoder, batch_size=batch_size, target_size=target_size, shuffle=False)

    return train_ds, val_ds, test_ds, encoder, splits_info
