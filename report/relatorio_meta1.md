# Relatório de Progresso — Meta I
## Análise do Problema e Desenvolvimento de um Modelo Baseado em Redes Neuronais

**Unidade Curricular:** Inteligência Computacional  
**Projeto:** Reconhecimento Multi-Label de Alimentos em Imagens de Pratos  
**Dataset:** FoodSeg103 (Adaptado de Wu et al., 2021)  
**Referência Teórica:** Müller, A. C., & Guido, S. (2016). *Introduction to Machine Learning with Python*. O'Reilly Media.

---

## 1. Introdução e Enquadramento do Problema

### 1.1 Natureza do Problema
O objetivo principal do projeto consiste em identificar que ingredientes ou alimentos estão presentes numa fotografia de um prato de comida. Diferente de tarefas clássicas de visão computacional em que cada imagem retrata um único objeto ou conceito (problemas *multiclass*, ex: "isto é um gato ou um cão?"), uma fotografia culinária retrata tipicamente **múltiplos alimentos em simultâneo** (ex: arroz, bife de frango e salada de tomate).

Deste modo, o problema formula-se formalmente como **Classificação Multi-Label de Imagens**:
- **Entrada ($X$):** Fotografia de um prato com formato $H \times W \times 3$ em espaço de cor RGB.
- **Saída ($Y$):** Vetor booleano $Y \in \{0, 1\}^C$, onde cada dimensão $c \in \{1, \dots, C\}$ assinala de forma independente a presença ($1$) ou ausência ($0$) da classe alimentar correspondente.

Não estamos perante uma tarefa de:
- *Classificação multiclass tradicional*, visto que a restrição de classe única e mutuamente exclusiva não se verifica;
- *Regressão*, pois não se pretendem quantificar pesos, volumes calóricos ou percentagens;
- *Segmentação semântica*, uma vez que a delimitação espacial pixel a pixel não é o alvo da aplicação.

### 1.2 Enquadramento no Livro de Referência
De acordo com os princípios metodológicos descritos por Müller & Guido (Capítulo 1 — *"Knowing Your Task and Knowing Your Data"* e Capítulo 2 — *"Supervised Learning"*), o ciclo de vida de qualquer projeto de aprendizagem automática supervisionada divide-se em três etapas imperativas:
1. Compreensão exaustiva das características estatísticas e distribuições dos dados disponíveis;
2. Construção de uma hipótese ou modelo de referência inicial (*baseline*);
3. Avaliação rigorosa da generalização do modelo num conjunto de teste estritamente isolado, para salvaguardar contra *overfitting* ou *data leakage*.

---

## 2. Análise Exploratória do Dataset FoodSeg103

O dataset adotado é o **FoodSeg103**, disponibilizado através do Hugging Face (`json9473/FoodSeg103`). Embora concebido originalmente para segmentação de ingredientes (*Wu et al., 2021*), o dataset possui anotações estruturadas na coluna `classes_on_image`, permitindo mapear os alimentos de cada imagem sem necessidade de treinar redes de segmentação computacionalmente pesadas.

### 2.1 Factos Reais Verificados Experimentalmente
A execução do módulo `src/dataset_analysis.py` permitiu apurar os seguintes factos estatísticos sem assumir estimativas infundadas:
- **Total de Amostras:** 7.118 imagens.
- **Divisões Fornecidas:** 4.983 imagens na partição `train` e 2.135 imagens na partição `validation`. O repositório oficial não inclui conjunto de `test` pré-definido.
- **Formato e Canais:** Imagens com 3 canais de cor (RGB), com resoluções bastante díspares (desde $113 \times 122$ até $5312 \times 4208$ píxeis).
- **Densidade Multi-Label:** Em média, cada imagem contém **3.68 alimentos** (mínimo de 1 e máximo de 11 ingredientes).

### 2.2 Seleção de Classes e Desbalanceamento
O dicionário oficial `id2label.json` contém 104 categorias (IDs de 0 a 103):
- **ID 0 (`background`):** Representa o prato, talheres e mesa na segmentação. Como não é alimento, foi **descartado** de todas as previsões.
- **ID 103 (`other ingredients`):** Categoria residual heterogénea (abrangendo desde especiarias e vinagres a sementes diversas). Metodologicamente, optou-se por **excluir** esta classe, mantendo apenas as **102 classes alimentares puras e bem definidas** (IDs 1 a 102).
- **Distribuição Long-Tail (Assimetria Extrema):** As classes mais frequentes possuem mais de um milhar de instâncias (`bread`: 1.405 imagens; `carrot`: 1.279 imagens; `chicken duck`: 1.242 imagens), ao passo que as classes mais raras contam com menos de dez registos (`kelp`: 3 imagens; `egg tart`: 4 imagens; `pudding`: 5 imagens). Esta assimetria terá forte impacto na avaliação de métricas macro.

---

## 3. Preparação e Engenharia dos Dados

### 3.1 Codificação Multi-Hot (`FoodLabelEncoder`)
Para alimentar a rede neuronal, as listas numéricas de classes (ex: `[0, 48, 66, 90]`) foram transformadas em vetores binários de ponto flutuante $y \in \mathbb{R}^{102}$:
- Cada posição $i \in \{0, \dots, 101\}$ corresponde a um alimento único ($\text{índice} = \text{ID} - 1$);
- $y_i = 1.0$ se o alimento estiver presente;
- $y_i = 0.0$ caso contrário.
Esta conversão foi validada e testada no script `src/validate_labels.py` em 10 amostras reais, confirmando reversibilidade perfeita (`decode(encode(y)) == y`).

### 3.2 Partição dos Dados sem Fuga de Informação (*Data Leakage*)
Para cumprir o requisito de avaliação independente prescrito no enunciado e no livro de Müller & Guido (Cap. 1 & 5), os dados foram repartidos da seguinte forma:
- **Treino (*Train*):** 4.983 imagens (100% da partição de treino original, maximizando os exemplos disponíveis para a rede aprender classes raras);
- **Validação (*Validation*):** 1.067 imagens (50% da partição de validação original, separada deterministicamente com semente `seed=42`);
- **Teste (*Test*):** 1.068 imagens (50% restantes, completamente isoladas de qualquer decisão de treino ou afinação).

### 3.3 Redimensionamento e Normalização de Píxeis
1. **Redimensionamento:** As imagens são uniformizadas para $224 \times 224$ píxeis com interpolação bilinear.
2. **Normalização (Müller & Guido, Cap. 3 — *Preprocessing and Scaling*):** Os valores inteiros de píxel $[0, 255]$ são escalados para o intervalo contínuo $[0.0, 1.0]$ dividindo por $255.0$. Esta operação impede que gradientes gigantescos desestabilizem os pesos da rede no início do treino.
3. **Pipeline Eficiente (`tf.data.Dataset`):** O carregamento é realizado em lotes (*batches* de 32 imagens) sob demanda, evitando o esgotamento da memória RAM.

---

## 4. Arquitetura da Rede Neuronal (Baseline)

### 4.1 Justificação da Escolha: CNN vs. MLP
O enunciado refere que a primeira fase deve contemplar uma rede MLP ou CNN. A escolha recaiu sobre uma **Rede Neuronal Convolucional (CNN)** devido às propriedades de visão computacional:
- Um MLP tradicional destruiria a geometria bidimensional da imagem ($224 \times 224 \times 3 = 150.528$ entradas unidimensionais), exigindo dezenas de milhões de pesos e provocando *overfitting* imediato;
- A CNN introduz **campos receptivos locais** (filtros de $3 \times 3$) e **partilha de pesos** (*weight sharing*), permitindo detetar uma textura alimentar em qualquer região do prato com uma fração reduzida de parâmetros.

### 4.2 Camada de Saída e Função de Ativação: `Sigmoid` vs. `Softmax`
Esta é a distinção técnica nuclear para a defesa do projeto:
- A função **Softmax** obriga a soma de todas as classes a ser $1.0$ ($\sum p_i = 1$). Esta função é matematicamente adequada a problemas mutuamente exclusivos (*multiclass*), mas errónea para pratos mistos;
- A função **Sigmoid** $\sigma(z_i) = \frac{1}{1 + e^{-z_i}}$ opera de modo **independente em cada um dos 102 neurónios**, gerando uma probabilidade isolada para cada alimento.

A função de perda correspondente é a **Binary Cross-Entropy (BCE)**, que penaliza o erro de cada uma das 102 previsões de forma dissociada.

### 4.3 Especificação das Camadas
A arquitetura implementada em `src/model.py` possui **122.918 parâmetros** (cerca de 480 KB), sendo estruturada em:
1. `Conv2D(32, 3x3, ReLU)` + `MaxPooling2D(2x2)` $\rightarrow$ extração de arestas e contrastes;
2. `Conv2D(64, 3x3, ReLU)` + `MaxPooling2D(2x2)` $\rightarrow$ extração de texturas;
3. `Conv2D(128, 3x3, ReLU)` + `MaxPooling2D(2x2)` $\rightarrow$ padrões espaciais complexos;
4. `GlobalAveragePooling2D()` $\rightarrow$ redução da matriz $28 \times 28 \times 128$ para um vetor de 128 valores médios (técnica superior ao Flatten por prevenir *overfitting*);
5. `Dense(128, ReLU)` + `Dropout(0.3)` $\rightarrow$ regularização estocástica;
6. `Dense(102, Sigmoid)` $\rightarrow$ camada de decisão multi-label.

---

## 5. Treino e Validação

O treino foi orquestrado pelo módulo `src/train.py` durante 5 épocas, com otimizador **Adam** ($\text{learning rate} = 0.001$) e salvaguardado por três mecanismos automáticos (*Callbacks*):
- `ModelCheckpoint`: Guardou o melhor modelo em `models/baseline_best.keras` (na época 4, quando a perda de validação atingiu o mínimo de $0.1265$);
- `EarlyStopping`: Proteção contra divergência e *overfitting*;
- `ReduceLROnPlateau`: Redução adaptativa da taxa de aprendizagem quando a perda de validação atinge um patamar.

---

## 6. Avaliação e Análise Crítica dos Resultados (Conjunto de Teste)

O modelo foi submetido à avaliação final nas **1.068 imagens do conjunto de teste independente** ($108.936$ decisões binárias avaliadas), através do script `src/evaluate.py`.

### 6.1 Tabela de Métricas Obtidas no Teste

| Métrica Oficial Exigida | Limiar Convencional ($0.5$) | Limiar Otimizado ($0.2$) | Conceito Teórico / Significado |
|---|---|---|---|
| **Hamming / Binary Accuracy** | **96.52%** | **95.84%** | Percentagem global de decisões (0s e 1s) acertadas. |
| **Subset Accuracy** | **0.00%** | **0.00%** | Fração de imagens em que todos os alimentos foram previstos perfeitamente. |
| **Sensibilidade (Recall Micro)**| **0.0000** | **0.0351** (3.5%) | Proporção de alimentos reais que o modelo conseguiu detetar. |
| **Especificidade (Micro)** | **1.0000 (100%)** | **0.9917** (99.2%) | Proporção de alimentos ausentes que foram corretamente rejeitados. |
| **Precisão (Precision Micro)** | **0.0000** | **0.2163** (21.6%) | Fração das previsões afirmativas do modelo que estavam corretas. |
| **F1-Score (F-measure Micro)** | **0.0000** | **0.0603** (6.0%) | Média harmónica de compromisso entre Precisão e Recall. |
| **AUC (Area Under ROC Micro)** | **0.8220** | **0.8220** | Capacidade de ordenação probabilística global da rede. |

### 6.2 Matriz de Confusão Agregada (Threshold = 0.5)
- **Verdadeiros Negativos (TN):** $105.143$
- **Falsos Positivos (FP):** $0$
- **Falsos Negativos (FN):** $3.793$
- **Verdadeiros Positivos (TP):** $0$

### 6.3 Discussão Crítica: A Falácia da Acurácia (*The Accuracy Fallacy*)
O resultado expõe uma das lições centrais discutidas no livro de Müller & Guido (Capítulo 5 — *"Evaluation Metrics and Scoring"*):
- Com o limiar padrão de $0.5$, a rede atinge **96.52% de Accuracy**, mas não detetou um único alimento ($TP = 0$). Isto acontece porque 96.5% das posições na matriz de teste são zeros (alimentos ausentes). Prever sempre "ausente" gera quase 97% de acerto sem aprender rigorosamente nada.
- É por esta razão específica que o enunciado do trabalho rejeita a avaliação baseada apenas em acurácia e exige **Recall, Especificidade, F1-Score e AUC**.
- O facto de a métrica **AUC atingir 0.8220** comprova que a rede aprendeu representações úteis (os alimentos presentes recebem pontuações consistentemente superiores aos ausentes), contudo a distribuição probabilística é comprimida em patamares baixos ($0.1$ a $0.3$) devido à raridade relativa dos ingredientes.
- Ao calibrar o limiar para $0.2$, a precisão dispara para **21.6%** e o Recall para **3.5%**, demonstrando que a rede é funcional mas necessita de otimização de limiares e transferência de aprendizagem.

---

## 7. Conclusão da Meta I e Planeamento da Meta II

### Conquistas da Meta I:
- [x] Formulação e delimitação rigorosa do problema de classificação multi-label;
- [x] Análise exploratória quantitativa do dataset FoodSeg103 com factos confirmados;
- [x] Implementação de pipeline reproduzível e modular com divisão estrita de dados;
- [x] Desenvolvimento de arquitetura CNN compacta de 122k parâmetros com saídas Sigmoid;
- [x] Treino monitorizado e avaliação formal com todas as métricas exigidas pelo enunciado;
- [x] Demonstração empírica da falácia da acurácia e das propriedades multi-label.

### Plano de Ação para a Meta II (Otimização e Investigação):
1. **Otimização de Limiares (*Threshold Tuning*):** Varrer limiares entre $0.05$ e $0.40$ para encontrar o ponto de equilíbrio ótimo que maximize o Micro F1-score;
2. **Aumento de Dados (*Data Augmentation*):** Introduzir rotações ligeiras, espelhamentos horizontais e variações subtis de iluminação para enriquecer a base de treino;
3. **Investigação de Transfer Learning:** Integrar uma arquitetura pré-treinada comprovada (ex: MobileNetV2 ou ResNet50) para extrair características visuais profundas e comparar formalmente o ganho de desempenho em relação a esta Baseline.

---

## 8. Referências Bibliográficas

1. **Müller, A. C., & Guido, S. (2016).** *Introduction to Machine Learning with Python: A Guide for Data Scientists*. O'Reilly Media.
2. **Wu, X., Fu, X., Liu, Y., Lim, E. P., Hoi, S. C., & Sun, Q. (2021).** *FoodSeg103: A Large-Scale Benchmark for Food Image Segmentation*. Proceedings of the ACM International Conference on Multimedia (MM '21), pp. 506-515.
3. **Hugging Face Datasets:** *FoodSeg103 Dataset Card* (`json9473/FoodSeg103`). Disponível em: https://huggingface.co/datasets/json9473/FoodSeg103.
