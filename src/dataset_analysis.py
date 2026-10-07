"""
===============================================================================
FASE 1 — Análise do Dataset FoodSeg103
===============================================================================

OBJETIVO:
    Carregar o dataset FoodSeg103 do Hugging Face e realizar uma análise
    exploratória completa ANTES de qualquer modelação.

PORQUÊ:
    Como explica o livro de referência (Müller & Guido, Cap. 1 — "Knowing
    Your Task and Knowing Your Data"), o primeiro passo em qualquer projeto
    de Machine Learning é compreender os dados. Não faz sentido construir
    um modelo sem saber:
    - Quantas imagens temos
    - Quantas classes existem
    - Como estão distribuídas
    - Qual o aspeto das imagens
    - Se existem problemas nos dados

O QUE ESTE SCRIPT FAZ:
    1. Carrega o dataset do Hugging Face
    2. Carrega o mapeamento ID → nome da classe (id2label.json)
    3. Mostra a estrutura e estatísticas básicas
    4. Analisa a distribuição das classes
    5. Analisa as dimensões das imagens
    6. Mostra exemplos visuais de imagens + alimentos
    7. Guarda todas as figuras em results/figures/

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
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para guardar figuras
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datasets import load_dataset
from PIL import Image

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
# Definir caminhos relativos à raiz do projeto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Estilo dos gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "figure.figsize": (12, 6)})


def load_id2label():
    """
    Carrega o mapeamento oficial ID → nome da classe.

    RACIOCÍNIO:
        O dataset FoodSeg103 usa IDs numéricos (0, 1, 2, ..., 103) para
        identificar cada classe. Para nós, humanos, é muito mais útil ver
        "rice" do que "66". O ficheiro id2label.json contém esta tradução.

        Carregamos directamente do Hugging Face para garantir que usamos
        a versão oficial e não inventamos nenhum mapeamento.
    """
    from huggingface_hub import hf_hub_download

    # Descarregar o ficheiro id2label.json do repositório oficial
    filepath = hf_hub_download(
        repo_id="json9473/FoodSeg103",
        filename="id2label.json",
        repo_type="dataset"
    )

    with open(filepath, "r", encoding="utf-8") as f:
        id2label = json.load(f)

    # As chaves vêm como strings ("0", "1", ...), converter para inteiros
    id2label = {int(k): v.strip() for k, v in id2label.items()}

    return id2label


def load_foodseg103():
    """
    Carrega o dataset FoodSeg103 do Hugging Face.

    RACIOCÍNIO:
        A biblioteca 'datasets' do Hugging Face permite carregar o dataset
        de forma programática, sem necessidade de download manual.
        O dataset é descarregado e cacheado localmente na primeira utilização.

    RETORNA:
        dataset: objecto DatasetDict com splits 'train' e 'validation'
    """
    print("=" * 70)
    print("A carregar o dataset FoodSeg103 do Hugging Face...")
    print("(Isto pode demorar na primeira execução — ~1.25 GB)")
    print("=" * 70)

    dataset = load_dataset("json9473/FoodSeg103")

    return dataset


def analyze_structure(dataset, id2label):
    """
    Analisa a estrutura básica do dataset.

    RACIOCÍNIO (Livro, Cap. 1):
        "It is good practice to inspect your data before building a model."
        Precisamos de saber exatamente o que temos antes de qualquer decisão.
    """
    print("\n" + "=" * 70)
    print("1. ESTRUTURA DO DATASET")
    print("=" * 70)

    # Splits disponíveis
    print(f"\nSplits disponíveis: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} imagens")

    # Colunas
    print(f"\nColunas: {dataset['train'].column_names}")

    # Features (tipos de dados)
    print(f"\nTipos de dados:")
    for col_name, feature in dataset['train'].features.items():
        print(f"  {col_name}: {feature}")

    # id2label
    print(f"\nNúmero total de categorias (incluindo background): {len(id2label)}")
    print(f"Número de categorias de alimentos (excluindo background): {len(id2label) - 1}")

    # Mostrar algumas classes como exemplo
    print(f"\nExemplos de classes:")
    for i in [0, 1, 24, 48, 54, 61, 66, 95, 103]:
        if i in id2label:
            marker = " <- BACKGROUND (excluir)" if i == 0 else ""
            marker = " <- categoria generica" if i == 103 else marker
            print(f"  ID {i:>3} -> {id2label[i]}{marker}")


def analyze_classes_on_image(dataset, id2label):
    """
    Analisa a coluna 'classes_on_image' — a chave do nosso problema.

    RACIOCÍNIO:
        Esta coluna contém os IDs das classes presentes em cada imagem.
        É a partir dela que vamos construir os vetores multi-hot.
        Precisamos de perceber:
        - Quantos alimentos tem cada imagem (mín, máx, média)
        - Quais as classes mais/menos frequentes
        - Se existem imagens sem alimentos
        - Se o background (ID 0) aparece e com que frequência
    """
    print("\n" + "=" * 70)
    print("2. ANÁLISE DE classes_on_image")
    print("=" * 70)

    # Recolher estatísticas de ambos os splits
    all_stats = {}
    all_class_counts = Counter()

    for split_name in dataset.keys():
        classes_lists = dataset[split_name]["classes_on_image"]

        # Número de classes por imagem
        num_classes_per_image = [len(c) for c in classes_lists]
        # Excluindo background
        num_food_per_image = [len([x for x in c if x != 0]) for c in classes_lists]

        # Contar frequência de cada classe
        for class_list in classes_lists:
            for class_id in class_list:
                all_class_counts[class_id] += 1

        # Imagens que contêm background
        has_background = sum(1 for c in classes_lists if 0 in c)

        stats = {
            "total": len(classes_lists),
            "num_classes_all": num_classes_per_image,
            "num_food_only": num_food_per_image,
            "has_background": has_background,
        }
        all_stats[split_name] = stats

        print(f"\n--- {split_name.upper()} ---")
        print(f"  Total de imagens: {stats['total']}")
        print(f"  Imagens com background (ID 0): {has_background} "
              f"({has_background/stats['total']*100:.1f}%)")
        print(f"\n  Classes por imagem (incluindo background):")
        print(f"    Mínimo:  {min(num_classes_per_image)}")
        print(f"    Máximo:  {max(num_classes_per_image)}")
        print(f"    Média:   {np.mean(num_classes_per_image):.2f}")
        print(f"    Mediana: {np.median(num_classes_per_image):.1f}")
        print(f"\n  Alimentos por imagem (excluindo background):")
        print(f"    Mínimo:  {min(num_food_per_image)}")
        print(f"    Máximo:  {max(num_food_per_image)}")
        print(f"    Média:   {np.mean(num_food_per_image):.2f}")
        print(f"    Mediana: {np.median(num_food_per_image):.1f}")

    return all_stats, all_class_counts


def plot_classes_distribution(all_class_counts, id2label):
    """
    Cria gráfico da distribuição de classes (frequência de cada alimento).

    RACIOCÍNIO:
        Precisamos de saber quais classes são comuns e quais são raras.
        Classes muito raras podem ser difíceis de aprender para o modelo
        (o modelo precisa de exemplos suficientes para aprender cada classe).
        Este é um conceito que o livro aborda quando fala de dados
        desequilibrados e representatividade dos dados de treino.
    """
    print("\n" + "=" * 70)
    print("3. DISTRIBUIÇÃO DAS CLASSES")
    print("=" * 70)

    # Excluir background
    food_counts = {k: v for k, v in all_class_counts.items() if k != 0}

    # Ordenar por frequência
    sorted_classes = sorted(food_counts.items(), key=lambda x: x[1], reverse=True)

    # Mostrar top 10 e bottom 10
    print("\nTop 10 classes mais frequentes:")
    for class_id, count in sorted_classes[:10]:
        print(f"  ID {class_id:>3} ({id2label[class_id]:>25}) → {count} imagens")

    print("\nTop 10 classes menos frequentes:")
    for class_id, count in sorted_classes[-10:]:
        print(f"  ID {class_id:>3} ({id2label[class_id]:>25}) → {count} imagens")

    # --- Gráfico de barras horizontal (todas as classes) ---
    fig, ax = plt.subplots(figsize=(14, 22))
    class_names = [id2label[cid] for cid, _ in sorted_classes]
    counts = [c for _, c in sorted_classes]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(class_names)))
    ax.barh(range(len(class_names)), counts, color=colors)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Número de imagens que contêm esta classe")
    ax.set_title("Distribuição das classes no FoodSeg103\n(ambos os splits, excluindo background)")
    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, "class_distribution.png")
    plt.savefig(filepath)
    plt.close()
    print(f"\nGráfico guardado: {filepath}")

    return sorted_classes


def plot_foods_per_image(all_stats):
    """
    Cria histograma do número de alimentos por imagem.

    RACIOCÍNIO:
        Num problema multi-label, cada imagem pode ter um número diferente
        de classes activas. Precisamos de perceber a distribuição:
        - A maioria das imagens tem poucos alimentos? Muitos?
        - Existem outliers com demasiados ou poucos alimentos?
    """
    print("\n" + "=" * 70)
    print("4. DISTRIBUIÇÃO DO NÚMERO DE ALIMENTOS POR IMAGEM")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (split_name, stats) in enumerate(all_stats.items()):
        ax = axes[idx]
        data = stats["num_food_only"]
        ax.hist(data, bins=range(0, max(data) + 2), edgecolor="black",
                alpha=0.7, color="steelblue")
        ax.set_xlabel("Número de alimentos por imagem")
        ax.set_ylabel("Número de imagens")
        ax.set_title(f"{split_name.upper()}\n"
                     f"(média={np.mean(data):.2f}, mediana={np.median(data):.1f})")
        ax.axvline(np.mean(data), color="red", linestyle="--",
                   label=f"Média: {np.mean(data):.2f}")
        ax.legend()

    plt.suptitle("Número de alimentos por imagem (excluindo background)", fontsize=13)
    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, "foods_per_image.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Gráfico guardado: {filepath}")


def analyze_image_dimensions(dataset):
    """
    Analisa as dimensões das imagens do dataset.

    RACIOCÍNIO:
        Redes neuronais normalmente exigem que todas as imagens tenham o
        mesmo tamanho (ex: 224×224 pixels). Antes de decidir qual o tamanho
        a utilizar, precisamos de saber quais são as dimensões originais.
        Se as imagens já forem todas do mesmo tamanho, o pré-processamento
        é mais simples.
    """
    print("\n" + "=" * 70)
    print("5. DIMENSÕES DAS IMAGENS")
    print("=" * 70)

    # Amostrar imagens para análise (não precisamos de ver todas)
    sample_size = min(500, len(dataset["train"]))
    indices = np.random.choice(len(dataset["train"]), sample_size, replace=False)

    widths = []
    heights = []

    for i in indices:
        img = dataset["train"][int(i)]["image"]
        w, h = img.size  # PIL Image: .size retorna (largura, altura)
        widths.append(w)
        heights.append(h)

    print(f"\nAnálise de {sample_size} imagens do split 'train':")
    print(f"  Largura  - min: {min(widths)}, max: {max(widths)}, "
          f"média: {np.mean(widths):.0f}, mediana: {np.median(widths):.0f}")
    print(f"  Altura   - min: {min(heights)}, max: {max(heights)}, "
          f"média: {np.mean(heights):.0f}, mediana: {np.median(heights):.0f}")

    # Verificar se todas são iguais
    unique_sizes = set(zip(widths, heights))
    if len(unique_sizes) == 1:
        print(f"\n  [OK] Todas as imagens amostradas tem o mesmo tamanho: "
              f"{widths[0]}x{heights[0]}")
    else:
        print(f"\n  [!] Existem {len(unique_sizes)} tamanhos diferentes")
        # Mostrar os 5 mais comuns
        size_counts = Counter(zip(widths, heights))
        print(f"  Top 5 tamanhos mais comuns:")
        for size, count in size_counts.most_common(5):
            print(f"    {size[0]}x{size[1]}: {count} imagens")

    # Modo (canais de cor)
    sample_img = dataset["train"][0]["image"]
    print(f"\n  Modo da imagem: {sample_img.mode}")
    if sample_img.mode == "RGB":
        print(f"  -> 3 canais (Vermelho, Verde, Azul)")
    elif sample_img.mode == "L":
        print(f"  -> 1 canal (escala de cinzento)")

    # Gráfico scatter de dimensões
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(widths, heights, alpha=0.3, s=10, color="steelblue")
    ax.set_xlabel("Largura (pixels)")
    ax.set_ylabel("Altura (pixels)")
    ax.set_title(f"Dimensões das imagens (amostra de {sample_size})")
    ax.axhline(np.mean(heights), color="red", linestyle="--", alpha=0.5)
    ax.axvline(np.mean(widths), color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, "image_dimensions.png")
    plt.savefig(filepath)
    plt.close()
    print(f"\nGráfico guardado: {filepath}")

    return widths, heights


def show_examples(dataset, id2label, num_examples=8):
    """
    Mostra exemplos de imagens com os nomes dos alimentos presentes.

    RACIOCÍNIO:
        Ver os dados reais é essencial. Não basta ver números e estatísticas.
        Precisamos de confirmar visualmente que:
        - As imagens são de comida
        - Os labels fazem sentido
        - Não existem erros óbvios
    """
    print("\n" + "=" * 70)
    print("6. EXEMPLOS VISUAIS")
    print("=" * 70)

    # Seleccionar exemplos aleatórios do treino
    indices = np.random.choice(len(dataset["train"]), num_examples, replace=False)

    # Criar grid de imagens
    cols = 4
    rows = (num_examples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for idx, i in enumerate(indices):
        sample = dataset["train"][int(i)]
        img = sample["image"]
        classes = sample["classes_on_image"]

        # Converter IDs para nomes (excluindo background)
        food_names = [id2label[c] for c in classes if c != 0]

        ax = axes[idx]
        ax.imshow(img)
        ax.set_title("\n".join(food_names) if food_names else "Sem alimentos",
                     fontsize=8, ha="center")
        ax.axis("off")

        # Imprimir no terminal também
        print(f"\n  Imagem {int(i)} (id={sample['id']}):")
        print(f"    classes_on_image = {classes}")
        print(f"    Alimentos: {', '.join(food_names) if food_names else 'Nenhum'}")

    # Esconder eixos vazios
    for idx in range(num_examples, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Exemplos de imagens do FoodSeg103 com alimentos identificados",
                 fontsize=13)
    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, "example_images.png")
    plt.savefig(filepath)
    plt.close()
    print(f"\nGráfico guardado: {filepath}")


def analyze_background_and_special(dataset, id2label):
    """
    Analisa especificamente o background (ID 0) e 'other ingredients' (ID 103).

    RACIOCÍNIO:
        ID 0 = "background" → não é um alimento, deve ser excluído dos labels
        ID 103 = "other ingredients" → categoria genérica
        Precisamos de decidir se mantemos o ID 103 ou o excluímos.
    """
    print("\n" + "=" * 70)
    print("7. ANÁLISE DO BACKGROUND E CATEGORIAS ESPECIAIS")
    print("=" * 70)

    for split_name in dataset.keys():
        classes_lists = dataset[split_name]["classes_on_image"]

        has_bg = sum(1 for c in classes_lists if 0 in c)
        has_other = sum(1 for c in classes_lists if 103 in c)

        print(f"\n--- {split_name.upper()} ---")
        print(f"  Imagens com background (ID 0):         {has_bg}/{len(classes_lists)} "
              f"({has_bg/len(classes_lists)*100:.1f}%)")
        print(f"  Imagens com 'other ingredients' (ID 103): {has_other}/{len(classes_lists)} "
              f"({has_other/len(classes_lists)*100:.1f}%)")

        # Imagens que SÓ têm background (sem alimentos)
        only_bg = sum(1 for c in classes_lists if c == [0])
        print(f"  Imagens que SÓ contêm background:      {only_bg}/{len(classes_lists)}")


def generate_summary(dataset, id2label, all_class_counts):
    """
    Gera um resumo final das descobertas.
    """
    print("\n" + "=" * 70)
    print("RESUMO DA ANÁLISE")
    print("=" * 70)

    total_images = sum(len(dataset[s]) for s in dataset.keys())
    food_classes = {k: v for k, v in id2label.items() if k != 0}

    # Classes que aparecem no dataset
    food_counts = {k: v for k, v in all_class_counts.items() if k != 0}
    classes_present = len([c for c in food_counts.values() if c > 0])

    print(f"""
    +---------------------------------------------+
    |           FoodSeg103 - RESUMO               |
    +---------------------------------------------+
    | Total de imagens:          {total_images:>6}            |
    |   - Train:                 {len(dataset['train']):>6}            |
    |   - Validation:            {len(dataset['validation']):>6}            |
    |   - Test:                  Nao existe        |
    |                                             |
    | Categorias no id2label:    {len(id2label):>6}            |
    | Categorias de alimentos:   {len(food_classes):>6}            |
    | Classes com dados:         {classes_present:>6}            |
    |                                             |
    | Problema: MULTI-LABEL CLASSIFICATION        |
    +---------------------------------------------+
    """)


# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    # Seed para reprodutibilidade dos exemplos aleatórios
    np.random.seed(42)

    # 1. Carregar o mapeamento ID → Label
    print("A carregar id2label.json...")
    id2label = load_id2label()
    print(f"[OK] Carregado: {len(id2label)} categorias")

    # 2. Carregar o dataset
    dataset = load_foodseg103()

    # 3. Analisar estrutura
    analyze_structure(dataset, id2label)

    # 4. Analisar classes_on_image
    all_stats, all_class_counts = analyze_classes_on_image(dataset, id2label)

    # 5. Distribuição das classes
    sorted_classes = plot_classes_distribution(all_class_counts, id2label)

    # 6. Alimentos por imagem
    plot_foods_per_image(all_stats)

    # 7. Dimensões das imagens
    widths, heights = analyze_image_dimensions(dataset)

    # 8. Exemplos visuais
    show_examples(dataset, id2label)

    # 9. Background e categorias especiais
    analyze_background_and_special(dataset, id2label)

    # 10. Resumo
    generate_summary(dataset, id2label, all_class_counts)

    print("\n[OK] Analise completa! Verifica os graficos em results/figures/")
