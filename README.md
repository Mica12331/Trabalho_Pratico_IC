# Classificação Multi-Label de Alimentos em Fotografias de Pratos
**Unidade Curricular:** Inteligência Computacional  
**Dataset:** FoodSeg103 (Adaptado para Multi-Label)  
**Referência Teórica Principal:** Andreas C. Müller & Sarah Guido — *Introduction to Machine Learning with Python*

---

## 📌 Visão Geral do Projeto
O objetivo deste projeto consiste em desenvolver uma rede neuronal capaz de analisar uma fotografia de um prato de comida e identificar quais os alimentos/ingredientes presentes na imagem.

Como um prato pode conter múltiplos alimentos em simultâneo (ex: arroz, frango e legumes), o problema é formulado como **Classificação Multi-Label de Imagens** (e não classificação multiclass tradicional).

```
   Fotografia do Prato
           │
           ▼
   Pré-processamento (224x224, [0, 1])
           │
           ▼
   Rede Neuronal Convolucional (CNN)
           │
           ▼
   102 Probabilidades Independentes (Sigmoid)
           │
           ▼
   Limiar de Decisão (Threshold)
           │
           ▼
   Alimentos Detetados (ex: arroz ✓, frango ✓, cenoura ✓)
```

---

## 🎯 Metas do Trabalho
- **Meta I (Concluída):** Análise do problema, preparação e engenharia de dados, desenvolvimento e avaliação da rede neuronal inicial (CNN Baseline).
- **Meta II (Próxima):** Trabalho de investigação, otimização de limiares (*threshold tuning*), *data augmentation* e *transfer learning*.
- **Meta III (Final):** Modelo final congelado, script de inferência para novas imagens e relatório final.

---

## 📊 O Dataset: FoodSeg103

O dataset original foi concebido para segmentação semântica de alimentos (*Wu et al., 2021*). Neste projeto, foi adaptado para classificação multi-label extraindo a lista de ingredientes presentes em cada prato (`classes_on_image`).

### Factos Reais Verificados:
- **Total de Imagens:** 7.118 fotografias RGB
- **Classes de Alimentos:** 102 classes puras (IDs 1 a 102). O ID 0 (`background`) e o ID 103 (`other ingredients`) foram excluídos metodologicamente.
- **Média de Alimentos por Prato:** 3.68 ingredientes (mínimo de 1, máximo de 11).
- **Divisão Metodológica (Sem Data Leakage):**
  - **Treino (*Train*):** 4.983 imagens (split de treino oficial).
  - **Validação (*Validation*):** 1.067 imagens (50% do split de validação oficial, semente `seed=42`).
  - **Teste (*Test*):** 1.068 imagens (50% restantes, conjunto independente e intocável).

---

## 🧠 Arquitetura da Rede Neuronal (Baseline)

Desenvolvemos uma **Convolutional Neural Network (CNN)** compacta e transparente, projetada para evitar *overfitting* mantendo custo computacional moderado:

| Bloco | Camada | Saída | Parâmetros | Função |
|---|---|---|---|---|
| **Entrada** | Input | `(None, 224, 224, 3)` | 0 | Imagem redimensionada e normalizada em $[0, 1]$ |
| **Bloco 1** | Conv2D (32 filtros 3x3) + MaxPool | `(None, 112, 112, 32)` | 896 | Deteção de arestas e contrastes básicos |
| **Bloco 2** | Conv2D (64 filtros 3x3) + MaxPool | `(None, 56, 56, 64)` | 18.496 | Deteção de texturas de alimentos |
| **Bloco 3** | Conv2D (128 filtros 3x3) + MaxPool | `(None, 28, 28, 128)` | 73.856 | Deteção de partes e formatos de ingredientes |
| **Pooling** | GlobalAveragePooling2D | `(None, 128)` | 0 | Resume mapas espaciais sem explosão de parâmetros |
| **Denso** | Dense (128, ReLU) + Dropout(0.3) | `(None, 128)` | 16.512 | Combinação de características com regularização |
| **Saída** | Dense (102, **Sigmoid**) | `(None, 102)` | 13.158 | **102 previsões binárias independentes** |

- **Total de Parâmetros:** **122.918** (~480 KB).
- **Função de Perda:** `Binary Cross-Entropy` (essencial para multi-label).
- **Otimizador:** `Adam` (`learning_rate = 0.001`).

---

## 📈 Resultados da Meta I (Conjunto de Teste Independente)

Avaliação realizada nas **1.068 imagens do conjunto de teste** ($108.936$ decisões binárias):

| Métrica Exigida pelo Enunciado | Threshold = 0.5 | Threshold = 0.2 | Conceito de Machine Learning |
|---|---|---|---|
| **Binary / Hamming Accuracy** | **96.52%** | **95.84%** | Proporção de acertos globais (0s e 1s). |
| **Subset Accuracy** | **0.00%** | **0.00%** | Acerto perfeito simultâneo de todos os ingredientes. |
| **Sensibilidade (Recall Micro)** | **0.0000** | **0.0351** | Capacidade de encontrar alimentos presentes. |
| **Especificidade (Micro)** | **1.0000 (100%)** | **0.9917** | Capacidade de rejeitar alimentos ausentes. |
| **Precisão (Precision Micro)** | **0.0000** | **0.2163** | Proporção de deteções corretas. |
| **F1-Score (F-measure Micro)** | **0.0000** | **0.0603** | Média harmónica entre Precisão e Recall. |
| **AUC (Micro ROC-AUC)** | **0.8220** | **0.8220** | Capacidade global de discriminação da rede. |

### Discussão Crítica: A "Falácia da Acurácia"
- Com o limiar $0.5$, a rede tem **96.52% de acurácia**, mas não deteta nenhum alimento ($Recall = 0$). Isso acontece porque 96.5% das posições na matriz de teste são ZEROS (alimentos ausentes). Prever sempre "ausente" dá quase 97% de acurácia sem aprender nada!
- O valor de **AUC = 0.8220** prova que a rede aprendeu representações válidas, mas as probabilidades situam-se em níveis baixos devido à raridade natural dos alimentos.
- Ao baixar o limiar para $0.2$, a precisão sobe para **21.6%** e o F1 para **6.0%**, fundamentando a necessidade de **Otimização de Limiares na Meta II**.

---

## 📁 Estrutura do Projeto

```
FoodRecognition/
├── data/                       # Dados e caches
├── models/                     # Modelos treinados (.keras)
│   └── baseline_best.keras     # Melhor modelo da baseline
├── notebooks/                  # Notebooks para exploração
├── report/                     # Relatórios académicos
│   └── relatorio_meta1.md      # Relatório completo da Meta I
├── results/
│   ├── figures/                # Gráficos analíticos gerados
│   │   ├── class_distribution.png
│   │   ├── foods_per_image.png
│   │   ├── image_dimensions.png
│   │   ├── batch_preview.png
│   │   ├── baseline_training_curves.png
│   │   └── baseline_confusion_matrix.png
│   └── metrics/                # Registos numéricos JSON
│       ├── baseline_history.json
│       └── baseline_test_metrics.json
├── src/                        # Código modular e reprodutível
│   ├── dataset_analysis.py     # Análise exploratória inicial
│   ├── preprocessing.py        # Codificador Multi-Hot (FoodLabelEncoder)
│   ├── validate_labels.py      # Testes e validação dos rótulos
│   ├── data_loader.py          # Redimensionamento, normalização e tf.data
│   ├── validate_data_loader.py # Teste do pipeline de dados
│   ├── model.py                # Arquitetura da CNN Baseline
│   ├── train.py                # Treino da baseline com callbacks
│   └── evaluate.py             # Avaliação rigorosa no conjunto de teste
├── requirements.txt            # Dependências do projeto
├── .gitignore                  # Ficheiros ignorados pelo Git
└── README.md                   # Documentação do projeto
```

---

## 🚀 Como Executar o Projeto

### 1. Clonar e Instalar Dependências
```bash
git clone <URL_DO_REPOSITORIO>
cd Trabalho_Pratico
pip install -r requirements.txt
```

### 2. Análise do Dataset
```bash
python src/dataset_analysis.py
```

### 3. Validação dos Rótulos e Pipeline
```bash
python src/validate_labels.py
python src/validate_data_loader.py
```

### 4. Treinar a CNN Baseline
```bash
python src/train.py
```

### 5. Avaliar no Conjunto de Teste
```bash
python src/evaluate.py
```
