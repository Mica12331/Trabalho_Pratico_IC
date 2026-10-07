"""
===============================================================================
FASE 2 — Módulo de Pré-processamento e Codificação Multi-Hot
===============================================================================

OBJETIVO:
    Transformar as anotações brutas do FoodSeg103 (`classes_on_image`) num
    vetor binário multi-hot utilizável por uma rede neuronal.

DECISÕES TÉCNICAS INCORPORADAS:
    - ID 0 (background) excluído (não é alimento).
    - ID 103 (other ingredients) excluído (categoria genérica/residual).
    - 102 classes alimentares puras mapeadas para índices de 0 a 101.
    - Vetor resultante: numpy array de tamanho 102 com tipo float32.

RELAÇÃO COM CONCEITOS DE ML (Müller & Guido, Cap. 2):
    Em classificação tradicional (multiclass), cada amostra tem uma única
    classe (one-hot vector com soma = 1).
    Em classificação multi-label (extensão deste projeto para visão computacional),
    uma imagem pode conter múltiplos alimentos em simultâneo. Por isso, usamos
    Multi-Hot Encoding, onde cada posição i do vetor representa a presença (1.0)
    ou ausência (0.0) independente da classe i.

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
from huggingface_hub import hf_hub_download

# Número total de classes alimentares consideradas no projeto
NUM_CLASSES = 102


class FoodLabelEncoder:
    """
    Codificador e descodificador bidirecional para as classes do FoodSeg103.

    Garante rastreabilidade total:
        ID original do dataset <-> Nome da classe <-> Índice no vetor multi-hot
    """

    def __init__(self):
        # 1. Carregar o id2label oficial do Hugging Face
        filepath = hf_hub_download(
            repo_id="json9473/FoodSeg103",
            filename="id2label.json",
            repo_type="dataset"
        )
        with open(filepath, "r", encoding="utf-8") as f:
            raw_id2label = json.load(f)

        # 2. Filtrar apenas as classes válidas (IDs de 1 a 102)
        # ID 0 (background) e ID 103 (other ingredients) são excluídos
        self.valid_ids = sorted([int(k) for k in raw_id2label.keys() if 1 <= int(k) <= 102])

        # 3. Construir dicionários de mapeamento
        # id_to_name: {1: 'candy', 2: 'egg tart', ..., 102: 'salad'}
        self.id_to_name = {cid: raw_id2label[str(cid)].strip() for cid in self.valid_ids}

        # id_to_idx: mapeia o ID do dataset (1..102) para o índice no vetor (0..101)
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.valid_ids)}

        # idx_to_id: mapeamento inverso do índice (0..101) para o ID (1..102)
        self.idx_to_id = {idx: cid for idx, cid in enumerate(self.valid_ids)}

        # idx_to_name: índice do vetor -> nome da classe
        self.idx_to_name = {idx: self.id_to_name[cid] for idx, cid in self.idx_to_id.items()}

        # name_to_idx: nome da classe -> índice do vetor
        self.name_to_idx = {name: idx for idx, name in self.idx_to_name.items()}

        # Lista ordenada de todos os nomes de classes (ordem correspondente aos índices 0..101)
        self.class_names = [self.idx_to_name[i] for i in range(NUM_CLASSES)]

    def encode(self, classes_on_image):
        """
        Converte uma lista de IDs (ex: [0, 48, 66, 103]) num vetor multi-hot (102,).

        Parâmetros:
            classes_on_image (list): lista de IDs presentes na imagem

        Retorno:
            np.ndarray: vetor binário de tamanho (102,) com tipo float32 (1.0 = presente, 0.0 = ausente)
        """
        multi_hot = np.zeros(NUM_CLASSES, dtype=np.float32)

        for class_id in classes_on_image:
            if class_id in self.id_to_idx:
                idx = self.id_to_idx[class_id]
                multi_hot[idx] = 1.0

        return multi_hot

    def decode(self, multi_hot_vector, threshold=0.5):
        """
        Converte um vetor de probabilidades ou binário (102,) numa lista de nomes de alimentos.

        Parâmetros:
            multi_hot_vector (np.ndarray): vetor de tamanho (102,) com 0s/1s ou probabilidades [0.0, 1.0]
            threshold (float): limiar de decisão (por defeito 0.5)

        Retorno:
            list de tuples: [(nome_da_classe, score), ...]
        """
        results = []
        for idx, score in enumerate(multi_hot_vector):
            if score >= threshold:
                results.append((self.idx_to_name[idx], float(score)))

        # Ordenar por maior confiança/score
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_class_names(self):
        """Retorna a lista ordenada dos 102 nomes de classes."""
        return self.class_names
