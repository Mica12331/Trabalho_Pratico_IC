"""
===============================================================================
Gerador do Relatório Académico em Microsoft Word (.docx) — Meta I
===============================================================================

OBJETIVO:
    Gerar um documento Word (.docx) formatado segundo as normas universitárias:
    - Capa formal e folha de rosto
    - Resumo e Palavras-chave
    - Índice automático (TOC - Table of Contents nativo do Word)
    - Margens académicas (Esquerda: 3cm, Direita/Cima/Baixo: 2.5cm)
    - Tipografia académica (Calibri, texto justificado, espaçamento 1.15, 6pt pós-parágrafo)
    - Cabeçalhos e rodapés com numeração automática de página
    - Títulos estruturados (Heading 1, Heading 2, Heading 3)
    - Tabelas formatadas com estilo profissional
    - Figuras reais de resultados embutidas com legendas formais
    - Citações e referências bibliográficas (Müller & Guido, Wu et al.)

===============================================================================
"""

import os
import sys

# Garantir UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

OUTPUT_DOCX = os.path.join("report", "Relatorio_Meta1_Inteligencia_Computacional.docx")
FIGURES_DIR = os.path.join("results", "figures")


def set_cell_background(cell, fill_hex):
    """Define a cor de fundo de uma célula de tabela em hexadecimal (ex: '1F497D')."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Define margens internas de uma célula."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)


def add_toc(doc):
    """
    Insere o campo XML do Word para o Índice Automático (Table of Contents).
    O Microsoft Word reconhece este campo e gera a listagem automática
    de todos os títulos Heading 1, Heading 2 e Heading 3 com as respetivas páginas.
    """
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run()
    fldChar1 = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls('w'))
    instrText = parse_xml(r'<w:instrText %s xml:space="preserve"> TOC \o "1-3" \h \z \u </w:instrText>' % nsdecls('w'))
    fldChar2 = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls('w'))
    fldChar3 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls('w'))
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)


def add_page_number_to_footer(footer):
    """Insere o número de página dinâmico no rodapé."""
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run("Página ")
    run.font.name = "Calibri"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(120, 120, 120)

    fldSimple = parse_xml(r'<w:fldSimple %s w:instr="PAGE"/>' % nsdecls('w'))
    p._p.append(fldSimple)

    run2 = p.add_run(" de ")
    run2.font.name = "Calibri"
    run2.font.size = Pt(9)
    run2.font.color.rgb = RGBColor(120, 120, 120)

    fldNumpages = parse_xml(r'<w:fldSimple %s w:instr="NUMPAGES"/>' % nsdecls('w'))
    p._p.append(fldNumpages)


def create_university_report():
    print("=" * 70)
    print("A CRIAR RELATÓRIO UNIVERSITÁRIO EM MICROSOFT WORD (.DOCX)")
    print("=" * 70)

    doc = Document()

    # 1. Configurar Margens Universitárias (3.0cm esquerda para encadernação, 2.5cm restantes)
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(0.98)     # ~2.5 cm
        s.bottom_margin = Inches(0.98)  # ~2.5 cm
        s.left_margin = Inches(1.18)    # ~3.0 cm
        s.right_margin = Inches(0.98)   # ~2.5 cm
        s.header_distance = Inches(0.5)
        s.footer_distance = Inches(0.5)

    # 2. Configurar Estilos Globais
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(40, 40, 40)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)
    normal_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Cores Académicas: Azul Escuro Institucional (RGB 31, 73, 125)
    COLOR_PRIMARY = RGBColor(31, 73, 125)
    HEX_PRIMARY = "1F497D"
    HEX_LIGHT_BG = "F2F5F8"

    # =========================================================================
    # CAPA UNIVERSITÁRIA
    # =========================================================================
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_before = Pt(36)
    p_inst.paragraph_format.space_after = Pt(4)
    r_inst = p_inst.add_run("INSTITUIÇÃO DE ENSINO SUPERIOR")
    r_inst.font.size = Pt(13)
    r_inst.font.bold = True
    r_inst.font.color.rgb = COLOR_PRIMARY

    p_dept = doc.add_paragraph()
    p_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dept.paragraph_format.space_after = Pt(120)
    r_dept = p_dept.add_run("Departamento de Engenharia Informática e Sistemas de Informação\nLicenciatura / Mestrado em Engenharia Informática")
    r_dept.font.size = Pt(10)
    r_dept.font.color.rgb = RGBColor(100, 100, 100)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(12)
    r_title = p_title.add_run("Classificação Multi-Label de Alimentos\nem Fotografias de Pratos")
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_PRIMARY

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(140)
    r_sub = p_sub.add_run("Meta I — Análise do Problema e Desenvolvimento de um Modelo Baseado em Redes Neuronais")
    r_sub.font.size = Pt(13)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(70, 70, 70)

    p_author = doc.add_paragraph()
    p_author.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_author.paragraph_format.space_after = Pt(4)
    r_aut = p_author.add_run("Autor: ")
    r_aut.font.bold = True
    p_author.add_run("Alcides Santos\n")
    r_uc = p_author.add_run("Unidade Curricular: ")
    r_uc.font.bold = True
    p_author.add_run("Inteligência Computacional\n")
    r_ref = p_author.add_run("Referência Teórica: ")
    r_ref.font.bold = True
    p_author.add_run("Müller & Guido (2016)")

    p_date = doc.add_paragraph()
    p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_date.paragraph_format.space_before = Pt(80)
    r_date = p_date.add_run("Ano Letivo 2025/2026")
    r_date.font.size = Pt(10)
    r_date.font.color.rgb = RGBColor(120, 120, 120)

    doc.add_page_break()

    # =========================================================================
    # RESUMO & PALAVRAS-CHAVE
    # =========================================================================
    h_res = doc.add_heading("Resumo", level=1)
    h_res.runs[0].font.color.rgb = COLOR_PRIMARY

    doc.add_paragraph(
        "O presente relatório descreve o trabalho realizado no âmbito da Meta I do projeto da unidade curricular "
        "de Inteligência Computacional, cujo propósito é o desenvolvimento de uma rede neuronal capaz de analisar "
        "fotografias de pratos culinários e identificar quais os alimentos ou ingredientes neles presentes. "
        "O problema foi formulado rigorosamente como uma tarefa de Classificação de Imagens Multi-Label, adaptando "
        "o benchmark de segmentação FoodSeg103 para extrair as 102 categorias alimentares válidas presentes em cada prato. "
        "Construiu-se um pipeline de pré-processamento eficiente em TensorFlow/Keras com redimensionamento uniforme "
        "(224x224 píxeis), normalização de escala [0, 1] e codificação multi-hot rastreável. "
        "Foi concebida e treinada uma Convolutional Neural Network (CNN) compacta com 122.918 parâmetros, utilizando "
        "função de ativação Sigmoid na camada de saída e função de perda Binary Cross-Entropy. "
        "A avaliação num conjunto de teste independente composto por 1.068 imagens permitiu apurar e fundamentar "
        "criticamente as métricas de Accuracy, Sensibilidade (Recall), Especificidade, F1-Score e ROC-AUC (0.8220), "
        "demonstrando de forma empírica o fenómeno da 'Falácia da Acurácia' provocado pelo desbalanceamento natural "
        "dos ingredientes e delineando o plano estratégico de otimização para a Meta II."
    )

    p_kw = doc.add_paragraph()
    r_kw_title = p_kw.add_run("Palavras-chave: ")
    r_kw_title.font.bold = True
    p_kw.add_run("Inteligência Computacional, Classificação Multi-Label, Visão Computacional, Redes Neuronais Convolucionais, FoodSeg103, Falácia da Acurácia.")

    doc.add_paragraph().paragraph_format.space_after = Pt(18)

    h_abs = doc.add_heading("Abstract", level=1)
    h_abs.runs[0].font.color.rgb = COLOR_PRIMARY

    doc.add_paragraph(
        "This report details the work accomplished in Milestone I of the Computational Intelligence course project, "
        "focused on developing a neural network capable of analyzing food photographs to identify all ingredients "
        "present on a plate. The task is formulated as Multi-Label Image Classification, adapting the FoodSeg103 "
        "benchmark to detect 102 distinct food categories. A complete data pipeline was implemented featuring 224x224 "
        "image resizing, pixel normalization into [0, 1], and bidirectional multi-hot encoding. "
        "A lightweight CNN baseline architecture (122,918 parameters) was trained using Sigmoid output units and Binary "
        "Cross-Entropy loss. Evaluation on an independent 1,068-image test set produced verified metrics across Accuracy, "
        "Recall, Specificity, F1-score, and ROC-AUC (0.8220), providing an empirical demonstration of the Accuracy Fallacy "
        "in imbalanced food data and establishing a solid baseline for Milestone II optimization."
    )

    p_akw = doc.add_paragraph()
    r_akw_title = p_akw.add_run("Keywords: ")
    r_akw_title.font.bold = True
    p_akw.add_run("Computational Intelligence, Multi-Label Classification, Convolutional Neural Networks, Food Recognition, FoodSeg103, Accuracy Fallacy.")

    doc.add_page_break()

    # =========================================================================
    # ÍNDICE GERAL AUTOMÁTICO (TOC)
    # =========================================================================
    h_toc = doc.add_heading("Índice Geral", level=1)
    h_toc.runs[0].font.color.rgb = COLOR_PRIMARY

    p_toc_note = doc.add_paragraph()
    r_tn = p_toc_note.add_run("[Nota: Este índice é gerado automaticamente pelo Microsoft Word a partir dos títulos do documento. Para atualizar números de página, clique com o botão direito no índice e selecione 'Atualizar Campo'.]")
    r_tn.font.size = Pt(9)
    r_tn.font.italic = True
    r_tn.font.color.rgb = RGBColor(120, 120, 120)

    # Inserir o campo XML TOC do Word
    add_toc(doc)

    doc.add_page_break()

    # Configurar cabeçalho e rodapé para as secções seguintes
    body_section = doc.sections[-1]
    header = body_section.header
    p_head = header.paragraphs[0]
    p_head.text = "Inteligência Computacional — Relatório da Meta I | Classificação Multi-Label de Alimentos"
    p_head.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_head.runs[0].font.name = "Calibri"
    p_head.runs[0].font.size = Pt(8.5)
    p_head.runs[0].font.color.rgb = RGBColor(120, 120, 120)

    footer = body_section.footer
    add_page_number_to_footer(footer)

    # Helper para adicionar títulos com numeração e cor
    def add_sec_heading(title, level):
        h = doc.add_heading(title, level=level)
        h.runs[0].font.name = "Calibri"
        h.runs[0].font.color.rgb = COLOR_PRIMARY
        if level == 1:
            h.paragraph_format.space_before = Pt(18)
            h.paragraph_format.space_after = Pt(8)
            h.runs[0].font.size = Pt(15)
        elif level == 2:
            h.paragraph_format.space_before = Pt(12)
            h.paragraph_format.space_after = Pt(4)
            h.runs[0].font.size = Pt(12.5)
        elif level == 3:
            h.paragraph_format.space_before = Pt(8)
            h.paragraph_format.space_after = Pt(2)
            h.runs[0].font.size = Pt(11)
        return h

    # Helper para adicionar figuras com legenda formal
    def insert_figure(filename, caption, width_in=5.5):
        img_path = os.path.join(FIGURES_DIR, filename)
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(8)
            p_img.paragraph_format.space_after = Pt(4)
            p_img.add_run().add_picture(img_path, width=Inches(width_in))

            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(12)
            r_cap_label = p_cap.add_run("Figura: ")
            r_cap_label.font.bold = True
            r_cap_label.font.size = Pt(9.5)
            r_cap_text = p_cap.add_run(caption)
            r_cap_text.font.size = Pt(9.5)
            r_cap_text.font.italic = True
        else:
            p_warn = doc.add_paragraph(f"[Figura não encontrada: {filename}]")
            p_warn.runs[0].font.italic = True

    # =========================================================================
    # CORPO DO RELATÓRIO
    # =========================================================================

    # -------------------------------------------------------------------------
    # 1. INTRODUÇÃO E ENQUADRAMENTO DO PROBLEMA
    # -------------------------------------------------------------------------
    add_sec_heading("1. Introdução e Enquadramento do Problema", level=1)

    add_sec_heading("1.1 Contexto e Motivação", level=2)
    doc.add_paragraph(
        "A análise automática de imagens de refeições assume crescente relevância na interseção entre "
        "Inteligência Artificial, ciências da nutrição e saúde preventiva. No entanto, o reconhecimento visual "
        "de pratos de comida constitui um dos desafios mais complexos em Visão Computacional devido à elevada "
        "deformabilidade dos ingredientes, à variação extrema nas formas de confeção e empratamento e, "
        "fundamentalmente, à copresença simultânea de múltiplos alimentos num mesmo prato."
    )

    add_sec_heading("1.2 Formulação do Problema: Multi-Label vs. Multiclass", level=2)
    doc.add_paragraph(
        "Em problemas canónicos de classificação supervisionada de imagens (como a distinção de dígitos no MNIST "
        "ou espécies animais no CIFAR-10), assume-se a premissa de que cada instância pertence a uma e apenas "
        "uma categoria mutuamente exclusiva (classificação multiclass). Na análise culinária, contudo, um prato típico "
        "apresenta arroz, carne e legumes em simultâneo. Trata-se, por conseguinte, de um problema de Classificação "
        "Multi-Label, onde a rede neuronal deve tomar 102 decisões binárias independentes em simultâneo."
    )
    doc.add_paragraph(
        "Não se pretende abordar tarefas de regressão (não se estimam gramagens, percentagens calóricas ou volumes) "
        "nem tarefas de segmentação semântica pixel a pixel, mantendo o foco exclusivo na questão: "
        "'Quais os alimentos/ingredientes presentes nesta imagem?'."
    )

    add_sec_heading("1.3 Objetivos da Meta I", level=2)
    doc.add_paragraph(
        "Em estrita concordância com o enunciado do trabalho prático, os objetivos definidos para a Meta I foram:\n"
        "1. Estudo analítico e verificação experimental dos factos do dataset FoodSeg103;\n"
        "2. Desenho e implementação da engenharia de dados (redimensionamento, normalização e codificação multi-hot);\n"
        "3. Conceção e implementação de uma arquitetura inicial baseada em redes neuronais convolucionais (CNN Baseline);\n"
        "4. Treino monitorizado e avaliação formal num conjunto de teste independente, reportando as métricas "
        "exigidas (Accuracy, Sensibilidade, Especificidade, F-measure, AUC e Matriz de Confusão)."
    )

    add_sec_heading("1.4 Enquadramento no Livro de Referência", level=2)
    doc.add_paragraph(
        "A metodologia de investigação adotada fundamenta-se nos princípios preconizados por Müller & Guido (2016). "
        "Nos Capítulos 1 e 2, os autores salientam a necessidade prioritária de inspecionar detalhadamente a tarefa "
        "e os dados antes de qualquer escolha algorítmica, alertando para os perigos do sobreajuste (overfitting). "
        "No Capítulo 3, fundamenta-se a indispensabilidade do escalamento de variáveis de entrada em redes neuronais. "
        "Finalmente, no Capítulo 5, explora-se a problemática do desbalanceamento severo de classes e a falácia "
        "do uso acrítico da acurácia como critério único de avaliação."
    )

    # -------------------------------------------------------------------------
    # 2. CARACTERIZAÇÃO E ANÁLISE DO DATASET FOODSEG103
    # -------------------------------------------------------------------------
    add_sec_heading("2. Caracterização e Análise do Dataset FoodSeg103", level=1)

    add_sec_heading("2.1 Origem e Descrição Geral", level=2)
    doc.add_paragraph(
        "O FoodSeg103 (Wu et al., 2021) é um benchmark construído a partir de receitas com fotografia do repositório "
        "Recipe1M. O dataset é composto por 7.118 imagens no total, distribuídas originalmente em 4.983 imagens no "
        "split de treino e 2.135 imagens no split de validação. Cada registo contém a imagem original, a máscara "
        "de segmentação e a lista 'classes_on_image' com os IDs das classes identificadas no prato."
    )

    add_sec_heading("2.2 Seleção e Filtragem de Classes", level=2)
    doc.add_paragraph(
        "O mapeamento oficial id2label.json contém 104 categorias (IDs de 0 a 103). Procedeu-se à filtragem rigorosa:\n"
        "• ID 0 (background): Representa pratos, talheres e mesa na anotação de píxeis. Não sendo um alimento, foi excluído;\n"
        "• ID 103 (other ingredients): Categoria residual que agrega dezenas de ingredientes não padronizados. "
        "De forma a garantir que a rede aprende conceitos visuais consistentes e unívocos, foi também excluída;\n"
        "• Restam exatamente 102 classes alimentares puras (IDs de 1 a 102), que constituem o espaço de classes do modelo."
    )

    insert_figure("batch_preview.png", "Amostra de pratos do dataset pré-processados a 224x224 com rótulos multi-hot descodificados.")

    add_sec_heading("2.3 Análise do Desbalanceamento Extremo (Long-Tail)", level=2)
    doc.add_paragraph(
        "A análise estatística revelou uma assimetria severa de representação. Enquanto ingredientes basilares "
        "como pão (1.405 imagens), cenoura (1.279), aves (1.242) e molhos (1.145) contam com elevada representação, "
        "ingredientes como kelp (3 imagens), tarte de ovo (4) e pudim (5) encontram-se raramente registados. "
        "Este desbalanceamento dita que a acurácia global será dominada pela ausência das classes raras, exigindo "
        "métricas insensíveis a este viés (como o ROC-AUC e F1-score)."
    )

    insert_figure("class_distribution.png", "Distribuição da frequência de ocorrência das classes alimentares no dataset.")
    insert_figure("foods_per_image.png", "Histograma da quantidade de alimentos simultâneos por fotografia culinária.")

    # -------------------------------------------------------------------------
    # 3. METODOLOGIA E ENGENHARIA DE DADOS
    # -------------------------------------------------------------------------
    add_sec_heading("3. Metodologia e Engenharia de Dados", level=1)

    add_sec_heading("3.1 Codificação Multi-Hot (FoodLabelEncoder)", level=2)
    doc.add_paragraph(
        "Implementou-se o módulo src/preprocessing.py através da classe FoodLabelEncoder. "
        "Para cada imagem, a lista de IDs brutos [0, 48, 66, 90] é convertida num vetor booleano y de dimensão 102, "
        "onde cada índice ativo corresponde a ID - 1. O módulo assegura rastreabilidade biunívoca total entre "
        "índice do tensor, ID oficial e nome do ingrediente, tendo sido validado por testes unitários exaustivos."
    )

    add_sec_heading("3.2 Partição dos Dados sem Fuga de Informação", level=2)
    doc.add_paragraph(
        "Como o FoodSeg103 não disponibiliza um split de teste isolado e o enunciado exige uma avaliação formal "
        "em dados nunca vistos, definiu-se a seguinte estratégia metodológica:\n"
        "• Treino (Train): 4.983 imagens (100% da partição de treino original, preservando o máximo de exemplos para o gradiente);\n"
        "• Validação (Validation): 1.067 imagens (50% da partição original de validação, com seed=42);\n"
        "• Teste Independente (Test): 1.068 imagens (os restantes 50%, guardados intactos até à fase de avaliação)."
    )

    add_sec_heading("3.3 Redimensionamento e Normalização de Píxeis", level=2)
    doc.add_paragraph(
        "As imagens originais possuem resoluções díspares que vão de 113x122 até 5312x4208 píxeis. "
        "Seguindo as boas práticas de Visão Computacional, cada imagem foi convertida para o espaço RGB e "
        "redimensionada para 224x224 píxeis através de interpolação bilinear. Em consonância com o Capítulo 3 de "
        "Müller & Guido, os valores inteiros dos píxeis [0, 255] foram normalizados para o intervalo contínuo [0.0, 1.0] "
        "dividindo por 255.0, evitando a saturação precoce dos gradientes."
    )

    # -------------------------------------------------------------------------
    # 4. ARQUITETURA DA REDE NEURONAL (BASELINE)
    # -------------------------------------------------------------------------
    add_sec_heading("4. Arquitetura da Rede Neuronal (Baseline)", level=1)

    add_sec_heading("4.1 Justificação da Escolha: CNN vs. MLP", level=2)
    doc.add_paragraph(
        "O enunciado admite redes MLP ou CNN. A opção recaiu categoricamente sobre uma Convolutional Neural Network (CNN). "
        "Um MLP tradicional alimentado por imagens de 224x224x3 píxeis exigiria achatar a matriz num vetor de 150.528 entradas. "
        "Uma primeira camada oculta com 512 unidades exigiria mais de 77 milhões de pesos ajustáveis, provocando "
        "sobreajuste imediato e destruindo as correlações geométricas bidimensionais da imagem. "
        "A CNN explora campos receptivos locais (filtros 3x3) e partilha de parâmetros, detetando texturas e formas "
        "com elevada eficiência computacional."
    )

    add_sec_heading("4.2 Função de Ativação de Saída: Sigmoid vs. Softmax", level=2)
    doc.add_paragraph(
        "Em problemas multiclass utiliza-se a função Softmax, que normaliza as saídas de modo a que somem 1.0, "
        "gerando competição mútua entre categorias. No contexto multi-label de pratos mistos, onde a presença de "
        "arroz não anula a presença de carne, a função de saída mandatória é a Sigmoid:"
    )
    p_eq = doc.add_paragraph()
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_eq = p_eq.add_run("σ(z_i) = 1 / (1 + e^(-z_i))   para cada classe i ∈ {1, ..., 102}")
    r_eq.font.italic = True
    r_eq.font.bold = True
    doc.add_paragraph(
        "Esta formulação permite que cada um dos 102 neurónios atue como um classificador binário independente. "
        "A função de perda é a Binary Cross-Entropy (BCE), que quantifica a divergência probabilística de cada classe."
    )

    add_sec_heading("4.3 Especificação das Camadas e Contagem de Parâmetros", level=2)
    doc.add_paragraph(
        "A arquitetura (implementada em src/model.py) foi dimensionada para conter 122.918 parâmetros (~480 KB), "
        "permitindo treino expedito e minimizando o risco de memorização (overfitting):"
    )

    # Tabela de Arquitetura
    table_arch = doc.add_table(rows=1, cols=5)
    table_arch.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Camada", "Tipo", "Formato de Saída", "Parâmetros", "Função no Modelo"]
    hdr_cells = table_arch.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        hdr_cells[i].paragraphs[0].runs[0].font.bold = True
        hdr_cells[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(hdr_cells[i], HEX_PRIMARY)
        set_cell_margins(hdr_cells[i], top=80, bottom=80, left=100, right=100)

    arch_data = [
        ("input_image", "InputLayer", "(None, 224, 224, 3)", "0", "Entrada RGB normalizada [0, 1]"),
        ("conv1", "Conv2D (32, 3x3)", "(None, 224, 224, 32)", "896", "Deteção de arestas e contrastes"),
        ("pool1", "MaxPooling2D (2x2)", "(None, 112, 112, 32)", "0", "Redução espacial e invariância local"),
        ("conv2", "Conv2D (64, 3x3)", "(None, 112, 112, 64)", "18.496", "Extração de texturas de alimentos"),
        ("pool2", "MaxPooling2D (2x2)", "(None, 56, 56, 64)", "0", "Redução espacial 2x"),
        ("conv3", "Conv2D (128, 3x3)", "(None, 56, 56, 128)", "73.856", "Extração de formas e padrões complexos"),
        ("pool3", "MaxPooling2D (2x2)", "(None, 28, 28, 128)", "0", "Redução espacial 2x"),
        ("gap", "GlobalAvgPool2D", "(None, 128)", "0", "Média espacial por canal (evita Flatten)"),
        ("dense_feat", "Dense (128, ReLU)", "(None, 128)", "16.512", "Integração semântica de features"),
        ("dropout", "Dropout (0.3)", "(None, 128)", "0", "Regularização (combate a overfitting)"),
        ("output", "Dense (102, Sigmoid)", "(None, 102)", "13.158", "102 probabilidades independentes [0, 1]")
    ]

    for row_idx, row_values in enumerate(arch_data):
        row_cells = table_arch.add_row().cells
        bg_col = HEX_LIGHT_BG if row_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_values):
            row_cells[c_idx].text = val
            row_cells[c_idx].paragraphs[0].runs[0].font.size = Pt(9.5)
            set_cell_background(row_cells[c_idx], bg_col)
            set_cell_margins(row_cells[c_idx], top=60, bottom=60, left=80, right=80)

    p_tcap = doc.add_paragraph()
    p_tcap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_tcap = p_tcap.add_run("Tabela 1 — Especificação da arquitetura e contagem de parâmetros da CNN Baseline.")
    r_tcap.font.size = Pt(9)
    r_tcap.font.italic = True

    # -------------------------------------------------------------------------
    # 5. TREINO E VALIDAÇÃO EXPERIMENTAL
    # -------------------------------------------------------------------------
    add_sec_heading("5. Treino e Validação Experimental", level=1)

    add_sec_heading("5.1 Protocolo Experimental e Hiperparâmetros", level=2)
    doc.add_paragraph(
        "O treino foi realizado durante 5 épocas com otimizador Adam (taxa de aprendizagem inicial de 0.001) "
        "e tamanho de lote (batch size) de 32 imagens. Cada época processou 156 lotes sobre as 4.983 imagens de treino."
    )

    add_sec_heading("5.2 Mecanismos de Salvaguarda (Callbacks)", level=2)
    doc.add_paragraph(
        "Implementaram-se três mecanismos de controlo do Keras (Müller & Guido, Cap. 2):\n"
        "1. ModelCheckpoint: Monitorizou a perda de validação (val_loss) e salvaguardou automaticamente o melhor "
        "modelo em models/baseline_best.keras (atingido na época 4);\n"
        "2. EarlyStopping: Configurado com paciência de 2 épocas para interromper o treino em caso de sobreajuste manifesto;\n"
        "3. ReduceLROnPlateau: Reduziu a taxa de aprendizagem para metade quando a descida do erro estagnou."
    )

    insert_figure("baseline_training_curves.png", "Curvas de aprendizagem da perda (Loss) e da área sob a curva ROC (AUC) ao longo das épocas.")

    # -------------------------------------------------------------------------
    # 6. AVALIAÇÃO E ANÁLISE CRÍTICA DOS RESULTADOS
    # -------------------------------------------------------------------------
    add_sec_heading("6. Avaliação e Análise Crítica dos Resultados", level=1)

    add_sec_heading("6.1 Desempenho no Conjunto de Teste Independente", level=2)
    doc.add_paragraph(
        "O melhor modelo foi submetido à avaliação final sobre as 1.068 imagens do conjunto de teste independente, "
        "totalizando 108.936 previsões binárias individuais (1.068 x 102). A Tabela 2 sintetiza os resultados "
        "obtidos para todas as métricas exigidas no enunciado do trabalho prático."
    )

    # Tabela de Métricas
    table_metrics = doc.add_table(rows=1, cols=4)
    table_metrics.alignment = WD_TABLE_ALIGNMENT.CENTER
    m_headers = ["Métrica Exigida pelo Enunciado", "Limiar Padrão (0.5)", "Limiar Ajustado (0.2)", "Interpretação no Contexto Multi-Label"]
    m_hdr_cells = table_metrics.rows[0].cells
    for i, title in enumerate(m_headers):
        m_hdr_cells[i].text = title
        m_hdr_cells[i].paragraphs[0].runs[0].font.bold = True
        m_hdr_cells[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(m_hdr_cells[i], HEX_PRIMARY)
        set_cell_margins(m_hdr_cells[i], top=80, bottom=80, left=100, right=100)

    metrics_data = [
        ("Binary / Hamming Accuracy", "96.52%", "95.84%", "Fração de decisões binárias acertadas (0s e 1s)"),
        ("Subset Accuracy", "0.00%", "0.00%", "Exige acerto simultâneo perfeito de todos os 102 rótulos"),
        ("Sensibilidade (Recall Micro)", "0.0000", "0.0351 (3.5%)", "Proporção de alimentos reais que a rede detetou"),
        ("Especificidade (Micro)", "1.0000 (100%)", "0.9917 (99.2%)", "Proporção de alimentos ausentes corretamente rejeitados"),
        ("Precisão (Precision Micro)", "0.0000", "0.2163 (21.6%)", "Fração de previsões afirmativas que estavam certas"),
        ("F1-Score (F-measure Micro)", "0.0000", "0.0603 (6.0%)", "Média harmónica entre Precisão e Recall"),
        ("AUC (Area Under ROC Micro)", "0.8220", "0.8220", "Capacidade global de ordenação probabilística da rede")
    ]

    for row_idx, row_values in enumerate(metrics_data):
        row_cells = table_metrics.add_row().cells
        bg_col = HEX_LIGHT_BG if row_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_values):
            row_cells[c_idx].text = val
            row_cells[c_idx].paragraphs[0].runs[0].font.size = Pt(9.5)
            set_cell_background(row_cells[c_idx], bg_col)
            set_cell_margins(row_cells[c_idx], top=60, bottom=60, left=80, right=80)

    p_mtcap = doc.add_paragraph()
    p_mtcap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_mtcap = p_mtcap.add_run("Tabela 2 — Resultados comparativos de desempenho no conjunto de teste independente (1.068 imagens).")
    r_mtcap.font.size = Pt(9)
    r_mtcap.font.italic = True

    add_sec_heading("6.2 Matriz de Confusão Agregada", level=2)
    doc.add_paragraph(
        "A agregação das 102 matrizes de confusão binárias no conjunto de teste para o limiar convencional de 0.5 revelou:\n"
        "• Verdadeiros Negativos (TN): 105.143 decisões corretas de ausência;\n"
        "• Falsos Positivos (FP): 0 falsos alarmes;\n"
        "• Falsos Negativos (FN): 3.793 ingredientes não detetados;\n"
        "• Verdadeiros Positivos (TP): 0 ingredientes confirmados."
    )

    insert_figure("baseline_confusion_matrix.png", "Matriz de Confusão Agregada calculada sobre as 108.936 previsões do conjunto de teste.")

    add_sec_heading("6.3 A Falácia da Acurácia em Dados Desbalanceados", level=2)
    doc.add_paragraph(
        "Os resultados obtidos ilustram de forma inequívoca o fenómeno que Müller & Guido (2016, Cap. 5) designam "
        "por 'A Falácia da Acurácia' (The Accuracy Fallacy). A rede obteve uma acurácia aparente de 96.52% apesar de "
        "não ter detetado nenhum ingrediente ativo (TP = 0). Isso sucede porque 96.5% de todas as posições na matriz "
        "de teste são ZEROS (ingredientes ausentes). Um preditor estático que responda sempre 'não há comida' alcança "
        "uma taxa de acerto de 96.5% sem aprender rigorosamente nenhuma representação visual."
    )
    doc.add_paragraph(
        "A relevância desta constatação é capital para a defesa do projeto: justifica perfeitamente o motivo pelo qual "
        "o enunciado do trabalho proíbe o recurso exclusivo à acurácia e exige métricas robustas ao desbalanceamento "
        "(como o Recall, F1 e AUC). O facto de o ROC-AUC atingir 0.8220 comprova matematicamente que a rede atribui "
        "probabilidades sistematicamente superiores aos alimentos presentes face aos ausentes; no entanto, devido à "
        "raridade relativa das classes, essas probabilidades situam-se naturalmente entre 0.10 e 0.35, ficando abaixo "
        "do patamar convencional de 0.50."
    )

    add_sec_heading("6.4 Impacto da Variação do Limiar (Threshold)", level=2)
    doc.add_paragraph(
        "Ao recalibrar o limiar de decisão para 0.20 (Tabela 2), a Sensibilidade salta de 0.00% para 3.51%, "
        "com a Precisão a fixar-se em 21.63% e o F1-score a subir para 6.03%. Esta evidência empírica estabelece a "
        "otimização sistemática do limiar de decisão como o primeiro vetor prioritário de investigação na Meta II."
    )

    # -------------------------------------------------------------------------
    # 7. CONCLUSÕES DA META I E PLANEAMENTO DA META II
    # -------------------------------------------------------------------------
    add_sec_heading("7. Conclusões da Meta I e Planeamento da Meta II", level=1)

    add_sec_heading("7.1 Balanço das Conquistas", level=2)
    doc.add_paragraph(
        "A Meta I foi concluída com sucesso pleno, cumprindo todos os requisitos estabelecidos:\n"
        "• Estruturação de um repositório modular, limpo e versionado em Git;\n"
        "• Inspeção aprofundada dos 7.118 registos do FoodSeg103 com factos empíricos documentados;\n"
        "• Desenho de um pipeline de engenharia de dados estritamente isolado contra data leakage;\n"
        "• Implementação e treino de uma CNN compacta de 122k parâmetros adaptada a classificação multi-label;\n"
        "• Avaliação rigorosa no teste independente, com discussão teórica detalhada sobre desbalanceamento e métricas."
    )

    add_sec_heading("7.2 Roteiro Estratégico para a Meta II", level=2)
    doc.add_paragraph(
        "Na Meta II ('Trabalho de investigação e otimização do modelo desenvolvido na fase anterior'), o projeto avançará "
        "através de três experiências controladas e fundamentadas:\n"
        "1. Otimização Global e por Classe do Limiar de Decisão: Varrer sistematicamente limiares no intervalo [0.05, 0.45] "
        "no conjunto de validação para maximizar o F1-score Micro e Macro;\n"
        "2. Aumento de Dados (Data Augmentation): Incorporar transformações geométricas realistas (espelhamento horizontal, "
        "rotações ligeiras de ±15° e ajustes de iluminação) no conjunto de treino para aumentar a capacidade de generalização;\n"
        "3. Transfer Learning com Redes Pré-treinadas: Investigar arquiteturas consagradas na literatura (como MobileNetV2 ou "
        "ResNet50 pré-treinadas no ImageNet), comparando formalmente os ganhos de sensibilidade e F1 face à Baseline atual."
    )

    # -------------------------------------------------------------------------
    # 8. REFERÊNCIAS BIBLIOGRÁFICAS
    # -------------------------------------------------------------------------
    add_sec_heading("8. Referências Bibliográficas", level=1)

    refs = [
        "Müller, A. C., & Guido, S. (2016). Introduction to Machine Learning with Python: A Guide for Data Scientists. O'Reilly Media.",
        "Wu, X., Fu, X., Liu, Y., Lim, E. P., Hoi, S. C., & Sun, Q. (2021). FoodSeg103: A Large-Scale Benchmark for Food Image Segmentation. In Proceedings of the 29th ACM International Conference on Multimedia (MM '21), pp. 506–515.",
        "Hugging Face. (2024). FoodSeg103 Dataset Card. Repositório oficial disponível em: https://huggingface.co/datasets/json9473/FoodSeg103.",
        "Chollet, F. (2021). Deep Learning with Python (2nd ed.). Manning Publications.",
        "Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press."
    ]

    for ref in refs:
        p_ref = doc.add_paragraph(ref)
        p_ref.paragraph_format.left_indent = Inches(0.5)
        p_ref.paragraph_format.first_line_indent = Inches(-0.5)
        p_ref.paragraph_format.space_after = Pt(6)

    # Guardar documento
    doc.save(OUTPUT_DOCX)
    print(f"\n[SUCESSO] Relatório universitário gerado e guardado em: {OUTPUT_DOCX}")


if __name__ == "__main__":
    create_university_report()
