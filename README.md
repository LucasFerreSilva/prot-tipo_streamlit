# **README – Analisador de Planilhas (Excel) com Streamlit e Plotly**

## **Descrição Geral**

Este projeto é uma aplicação interativa desenvolvida com **Streamlit**, **Pandas** e **Plotly**, projetada para realizar análise exploratória de dados diretamente a partir de arquivos Excel (.xlsx ou .xls).

A ferramenta permite que o usuário carregue uma planilha, selecione a aba desejada, defina a linha de cabeçalho, escolha colunas para análise e visualize estatísticas e gráficos gerados automaticamente. As funcionalidades foram construídas para oferecer uma análise flexível, mesmo quando o arquivo Excel contém formatações inconsistentes ou cabeçalhos desalinhados.

O programa foi idealizado para cenários como experimentos científicos, análise laboratorial, dados de produção, planilhas administrativas, entre outros.

---

## **Funcionalidades Principais**

### 1. **Upload de Arquivo Excel**

* Suporte para arquivos .xlsx e .xls.
* Exibição das primeiras 20 linhas da aba selecionada, sem cabeçalho.
* Leitura eficiente com cache para evitar recarregamentos desnecessários.

### 2. **Seleção da Aba e Configuração do Cabeçalho**

* Seleção de qualquer aba contida no Excel.
* Possibilidade de escolher qual linha representa o cabeçalho.
* Criação automática de nomes únicos para colunas duplicadas ou vazias.

### 3. **Limpeza e Padronização dos Dados**

* Remoção de linhas e colunas completamente vazias.
* Conversão automática de colunas para formato numérico quando apropriado.
* Detecção e registro de valores inválidos convertidos para NaN.
* Visualização completa dos tipos de dados após conversão.

### 4. **Exclusão Opcional de Colunas**

* Permite remover colunas da análise sem afetar a visualização da tabela original.
* Evita duplicidade de nomes de colunas.

### 5. **Filtros Dinâmicos**

* Filtragem por valores presentes na coluna de tratamento.
* Filtro por faixa de valores para a variável numérica selecionada.
* Atualização automática das tabelas e gráficos conforme os filtros são alterados.

### 6. **Análise Estatística**

O programa gera automaticamente um resumo estatístico por grupo (tratamento), contendo:

* número de observações
* média
* desvio padrão
* valor mínimo
* valor máximo

Além disso, apresenta indicadores gerais, como:

* total de amostras filtradas
* média geral
* desvio padrão geral

### 7. **Geração de Gráficos Interativos (Plotly)**

O tipo de gráfico oferecido depende do tipo da variável numérica:

#### Se a variável for contínua:

* Barras – média por tratamento
* Boxplot por tratamento
* Histograma com número de bins calculado pela regra de Freedman–Diaconis
* Barras empilhadas por faixas (quantis)

#### Se a variável tiver poucos valores únicos (variável discreta):

* Barras – média por tratamento
* Barras – frequência por valor
* Barras empilhadas por faixas (quantis)

### 8. **Abas de Organização**

A interface utiliza três abas principais:

1. **Gráficos & Resumo**: KPIs, resumo estatístico e gráfico interativo.
2. **Dados Filtrados**: visualização da tabela após filtros.
3. **Detalhes da Conversão**: relatório de conversões numéricas e tipos de dados.

### 9. **Exportação de Dados**

* Download dos dados limpos (antes dos filtros) em formato CSV.
* Download dos dados filtrados em formato CSV.

---

## **Como Utilizar**

### Pré-requisitos:

* Python instalado
* Dependências instaladas:

  ```
  pip install streamlit pandas numpy plotly openpyxl
  ```

### Executar o programa:

Dentro da pasta do projeto:

```
streamlit run nome_do_arquivo.py
```

O navegador abrirá a interface automaticamente.

---

## **Estrutura do Código**

O código é organizado nas seguintes seções:

1. **Configuração da página**
2. **Funções auxiliares** (tratamento de dados, conversão numérica, cálculo de bins, criação de gráficos)
3. **Upload do arquivo**
4. **Carregamento das abas e pré-visualização**
5. **Processamento do cabeçalho**
6. **Limpeza e tratamentos das colunas**
7. **Filtros dinâmicos**
8. **Resumo estatístico**
9. **Geração dos gráficos**
10. **Exibição final e exportação**
