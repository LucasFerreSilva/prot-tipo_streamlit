import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==============================
# Configuração da página
# ==============================
st.set_page_config(
    page_title="Análise de Planilhas – Streamlit + Plotly",
    layout="wide"
)

st.title("📊 Analisador de planilhas (Excel + Plotly)")

st.markdown(
    '''
    Carregue uma planilha Excel, escolha a aba, ajuste a linha do cabeçalho
    e selecione a coluna de tratamento e uma ou mais colunas numéricas para gerar
    tabelas e gráficos interativos automaticamente.
    As opções de gráficos se adaptam ao tipo de dado e você pode aplicar filtros dinâmicos.
    '''
)

# ==============================
# Funções auxiliares
# ==============================

@st.cache_data
def get_sheet_names(file_bytes: bytes):
    """Retorna os nomes das abas do Excel, com cache."""
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    return xls.sheet_names


@st.cache_data
def load_sheet(file_bytes: bytes, sheet_name: str, nrows: int | None = None) -> pd.DataFrame:
    """Carrega uma aba do Excel (sem cabeçalho), opcionalmente limitado a nrows, com cache."""
    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet_name,
        header=None,
        nrows=nrows
    )


def build_clean_dataframe(df_raw_full: pd.DataFrame, header_row_idx_zero: int) -> pd.DataFrame:
    """Monta o DataFrame com cabeçalho correto, removendo linhas/colunas completamente vazias."""
    header_series = df_raw_full.iloc[header_row_idx_zero]

    # Criar nomes de colunas únicos
    new_cols = []
    usados = set()

    for i, val in enumerate(header_series):
        if pd.isna(val) or str(val).strip() == "":
            col_name = f"col_{i+1}"
        else:
            col_name = str(val).strip()

        base = col_name
        cont = 1
        while col_name in usados:
            col_name = f"{base}_{cont}"
            cont += 1

        usados.add(col_name)
        new_cols.append(col_name)

    # Dados começam na linha após o cabeçalho
    df_temp = df_raw_full.iloc[header_row_idx_zero + 1 :].copy()
    df_temp.columns = new_cols

    # Remover colunas e linhas totalmente vazias
    df_temp = df_temp.dropna(axis=1, how="all")
    df_temp = df_temp.dropna(axis=0, how="all")

    df_clean = df_temp.reset_index(drop=True)
    return df_clean


def convert_numeric_columns(df: pd.DataFrame):
    """
    Converte colunas que parecem numéricas (muitas entradas válidas numéricas)
    usando vírgula como separador decimal se houver.
    Retorna o DataFrame convertido e um dicionário com estatísticas de conversão.
    """
    df_conv = df.copy()
    info_invalid = {}

    for col in df_conv.columns:
        if df_conv[col].dtype == "object":
            original = df_conv[col]
            s = original.astype(str).str.replace(",", ".")
            converted = pd.to_numeric(s, errors="coerce")

            non_null_before = original.notna()
            total_non_null = int(non_null_before.sum())
            if total_non_null == 0:
                continue

            invalid_mask = non_null_before & converted.isna()
            num_invalid = int(invalid_mask.sum())
            frac_invalid = num_invalid / total_non_null

            # Só converte se a maioria for válida (evita tentar converter colunas categóricas)
            if frac_invalid < 0.5:
                df_conv[col] = converted
                if num_invalid > 0:
                    info_invalid[col] = {
                        "valores_na_coluna": total_non_null,
                        "invalidos_convertidos_para_NaN": num_invalid,
                        "percentual_invalidos": round(frac_invalid * 100, 1),
                    }

    return df_conv, info_invalid


def freedman_diaconis_bins(series: pd.Series):
    """Calcula número de bins via regra de Freedman–Diaconis; retorna None se não for possível."""
    data = series.dropna().values
    n = len(data)
    if n < 2:
        return None

    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return None

    bin_width = 2 * iqr / (n ** (1 / 3))
    if bin_width <= 0:
        return None

    n_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    if n_bins < 5:
        n_bins = 5
    return n_bins


def compute_summary(df: pd.DataFrame, col_treat: str, col_y: str) -> pd.DataFrame:
    """Resumo estatístico por tratamento para uma coluna."""
    resumo = (
        df.groupby(col_treat)[col_y]
        .agg(["count", "mean", "std", "min", "max"])
        .rename(
            columns={
                "count": "n",
                "mean": "média",
                "std": "desvio padrão",
                "min": "mínimo",
                "max": "máximo",
            }
        )
    )
    return resumo


def create_plot(
    df_filtered: pd.DataFrame,
    resumo: pd.DataFrame,
    col_treat: str,
    col_y: str,
    cols_y: list,
    tipo_grafico: str,
    is_discrete_small: bool,
):
    """Cria um gráfico Plotly de acordo com o tipo de gráfico escolhido."""
    if df_filtered.empty:
        return None

    # Barras – média por tratamento (suporta 1 ou várias colunas numéricas)
    if tipo_grafico == "Barras – média por tratamento":
        if len(cols_y) == 1:
            resumo_plot = resumo.reset_index()
            fig = px.bar(
                resumo_plot,
                x=col_treat,
                y="média",
                text="média",
                title=f"Média de {col_y} por {col_treat}",
                labels={col_treat: col_treat, "média": col_y},
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(yaxis_title=col_y)
            return fig
        else:
            resumo_multi = df_filtered.groupby(col_treat)[cols_y].mean().reset_index()
            tidy = resumo_multi.melt(id_vars=col_treat, var_name="Variável", value_name="Média")
            fig = px.bar(
                tidy,
                x=col_treat,
                y="Média",
                color="Variável",
                barmode="group",
                text="Média",
                title=f"Média das variáveis selecionadas por {col_treat}",
                labels={"Média": "Valor médio"},
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            return fig

    # Boxplot por tratamento (apenas contínuo, usa variável principal)
    if tipo_grafico == "Boxplot por tratamento" and not is_discrete_small:
        fig = px.box(
            df_filtered,
            x=col_treat,
            y=col_y,
            title=f"Distribuição de {col_y} por {col_treat}",
            points="all",
        )
        return fig

    # Histograma (faixas automáticas, usa variável principal)
    if tipo_grafico == "Histograma (faixas automáticas)":
        n_bins = freedman_diaconis_bins(df_filtered[col_y])
        fig = px.histogram(
            df_filtered,
            x=col_y,
            nbins=n_bins,
            title=f"Histograma de {col_y}" + (f" (bins={n_bins})" if n_bins else ""),
        )
        fig.update_yaxes(title="Frequência")
        return fig

    # Barras – frequência por valor (para escala discreta, usa variável principal)
    if tipo_grafico == "Barras – frequência por valor":
        freq = pd.crosstab(df_filtered[col_treat], df_filtered[col_y])
        if freq.empty:
            return None
        freq_tidy = (
            freq.reset_index()
            .melt(id_vars=col_treat, var_name=col_y, value_name="Frequência")
        )
        fig = px.bar(
            freq_tidy,
            x=col_treat,
            y="Frequência",
            color=col_y,
            barmode="group",
            title=f"Frequência de {col_y} por {col_treat}",
        )
        return fig

    # Barras empilhadas – faixas (quantis) (usa variável principal)
    if tipo_grafico == "Barras empilhadas – faixas (quantis)":
        series = df_filtered[col_y].dropna()
        if len(series) < 3:
            return None

        labels = ["Baixo", "Médio", "Alto"]
        try:
            faixa = pd.qcut(series, q=3, labels=labels, duplicates="drop")
        except Exception:
            # Fallback: se qcut falhar (muitos valores repetidos), usa cut em 3 intervalos fixos
            try:
                faixa = pd.cut(series, bins=3, labels=labels)
            except Exception:
                return None

        df_faixa = df_filtered.loc[series.index, [col_treat]].copy()
        df_faixa["Faixa"] = faixa

        freq = (
            pd.crosstab(df_faixa[col_treat], df_faixa["Faixa"], normalize="index")
            .reset_index()
            .melt(id_vars=col_treat, var_name="Faixa", value_name="Proporção")
        )

        fig = px.bar(
            freq,
            x=col_treat,
            y="Proporção",
            color="Faixa",
            barmode="stack",
            title=f"Distribuição de {col_y} em faixas por {col_treat}",
        )
        fig.update_yaxes(tickformat=".0%")
        return fig

    return None


# ==============================
# 1. Upload do arquivo
# ==============================
uploaded_file = st.file_uploader(
    "Selecione o arquivo Excel (.xlsx ou .xls)",
    type=["xlsx", "xls"],
)

if uploaded_file is None:
    st.info("👆 Carregue um arquivo para começar.")
    st.stop()

file_bytes = uploaded_file.getvalue()

# ==============================
# 2. Abas do arquivo (com cache)
# ==============================
try:
    sheet_names = get_sheet_names(file_bytes)
except Exception as e:
    st.error(f"Erro ao ler o arquivo: {e}")
    st.stop()

st.sidebar.header("Configurações")

sheet_name = st.sidebar.selectbox(
    "Escolha a aba da planilha",
    sheet_names,
)

# Carrega apenas as primeiras linhas para preview (performance)
df_raw_preview = load_sheet(file_bytes, sheet_name, nrows=20)
# Carrega a aba completa para o processamento
df_raw_full = load_sheet(file_bytes, sheet_name, nrows=None)

with st.expander("Ver dados brutos da aba (primeiras 20 linhas, sem cabeçalho)"):
    st.dataframe(df_raw_preview)

# ==============================
# 3. Escolha da linha de cabeçalho
# ==============================
header_row_display = st.sidebar.number_input(
    "Linha do cabeçalho (1 = primeira linha da aba)",
    min_value=1,
    max_value=len(df_raw_full),
    value=1,
    step=1,
)
header_row_idx_zero = header_row_display - 1

df_clean = build_clean_dataframe(df_raw_full, header_row_idx_zero)

st.subheader(f"📄 Dados após aplicar cabeçalho (linha {header_row_display})")
st.dataframe(df_clean.head())

# ==============================
# 4. Conversão numérica com validação
# ==============================
df_conv, invalid_info = convert_numeric_columns(df_clean)

# ==============================
# 5. Exclusão de colunas da análise
# ==============================
st.sidebar.markdown("### Limpeza de colunas")

cols_exclude = st.sidebar.multiselect(
    "Excluir colunas da análise (não afeta visualização da tabela original)",
    options=df_conv.columns.tolist(),
    default=[],
)

if cols_exclude:
    df_analysis = df_conv.drop(columns=cols_exclude)
else:
    df_analysis = df_conv.copy()

# Garantir que não existam nomes de colunas duplicados
if df_analysis.columns.duplicated().any():
    st.warning(
        "Foram encontradas colunas com nomes repetidos. "
        "Apenas a primeira ocorrência de cada nome será mantida na análise."
    )
    df_analysis = df_analysis.loc[:, ~df_analysis.columns.duplicated()]

st.subheader("📄 Dados prontos para análise (após exclusões opcionais)")
st.dataframe(df_analysis.head())

# Botão de download do CSV limpo (df_analysis)
csv_bytes = df_analysis.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="📥 Baixar dados LIMPOS em CSV",
    data=csv_bytes,
    file_name=f"dados_tratados_{sheet_name}.csv",
    mime="text/csv",
)

# ==============================
# 6. Escolha de colunas para análise
# ==============================
st.sidebar.markdown("### Colunas para análise")

if df_analysis.empty:
    st.warning("A tabela de análise está vazia após as exclusões. Ajuste as colunas removidas.")
    st.stop()

col_treat = st.sidebar.selectbox(
    "Coluna que identifica o tratamento / grupo",
    df_analysis.columns,
)

# Colunas numéricas disponíveis (removendo a de tratamento, se for numérica)
numeric_cols_all = df_analysis.select_dtypes(include="number").columns.tolist()
numeric_cols = [c for c in numeric_cols_all if c != col_treat]

if not numeric_cols:
    st.warning(
        "⚠️ Não há colunas numéricas disponíveis (exceto a de tratamento) após a conversão e exclusões.\n"
        "Verifique a linha do cabeçalho e o que foi excluído."
    )
    st.stop()

# Coluna principal para análise (usada em filtros, KPIs, alguns gráficos)
col_y = st.sidebar.selectbox(
    "Coluna numérica principal para analisar",
    numeric_cols,
)

# Colunas adicionais opcionais para gráficos (apenas usadas em "Barras – média por tratamento")
extra_numeric_options = [c for c in numeric_cols if c != col_y]
cols_y_extra = st.sidebar.multiselect(
    "Colunas numéricas adicionais para gráficos (opcional)",
    options=extra_numeric_options,
    default=[],
)

cols_y = [col_y] + cols_y_extra

# Segurança extra: garantir que a coluna de tratamento não caiu aqui por algum motivo
if col_treat in cols_y:
    st.error(
        "A coluna de tratamento não pode ser usada como métrica numérica. "
        "Ajuste a seleção de colunas."
    )
    st.stop()

# impedir que usuário escolha a mesma coluna de tratamento e métrica principal
if col_treat == col_y:
    st.error(
        "A coluna de tratamento e a coluna numérica principal não podem ser a mesma. "
        "Por favor, selecione colunas diferentes."
    )
    st.stop()

# ==============================
# 7. Filtros dinâmicos
# ==============================
df_filtered = df_analysis[[col_treat] + cols_y].copy()
df_filtered = df_filtered.dropna(subset=[col_treat, col_y])

# Filtro por categorias de tratamento
col_treat_data = df_filtered[col_treat]
if isinstance(col_treat_data, pd.DataFrame):
    col_treat_data = col_treat_data.iloc[:, 0]

unique_treats = sorted(col_treat_data.dropna().unique().tolist())

selected_treats = st.sidebar.multiselect(
    "Filtrar tratamentos",
    options=unique_treats,
    default=unique_treats,
)

if selected_treats:
    df_filtered = df_filtered[df_filtered[col_treat].isin(selected_treats)]
else:
    st.warning("Nenhum tratamento selecionado nos filtros.")
    st.stop()

# Filtro por faixa de valores da variável numérica principal
min_val = float(df_filtered[col_y].min())
max_val = float(df_filtered[col_y].max())

if min_val == max_val:
    range_vals = (min_val, max_val)
else:
    step_val = (max_val - min_val) / 100 if max_val > min_val else 1.0
    range_vals = st.sidebar.slider(
        f"Filtrar faixa de valores de {col_y}",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val),
        step=step_val,
    )

df_filtered = df_filtered[
    (df_filtered[col_y] >= range_vals[0]) & (df_filtered[col_y] <= range_vals[1])
]

if df_filtered.empty:
    st.warning("Nenhum dado após aplicar os filtros. Ajuste os filtros na barra lateral.")
    st.stop()

# ==============================
# 8. Resumo estatístico e tipo de dado (para variável principal)
# ==============================
data_col = df_filtered[col_y].dropna()
n_unique = data_col.nunique()
is_discrete_small = n_unique <= 15

try:
    resumo = compute_summary(df_filtered, col_treat, col_y)
except Exception as e:
    st.error(f"Erro ao calcular o resumo estatístico: {e}")
    st.stop()

# ==============================
# 9. Tipos de gráfico (adaptativos)
# ==============================
if is_discrete_small:
    st.sidebar.markdown("### Tipo de gráfico (dado discreto/escala)")
    opcoes_grafico = [
        "Barras – média por tratamento",
        "Barras – frequência por valor",
        "Barras empilhadas – faixas (quantis)",
    ]
else:
    st.sidebar.markdown("### Tipo de gráfico (dado contínuo)")
    opcoes_grafico = [
        "Barras – média por tratamento",
        "Boxplot por tratamento",
        "Histograma (faixas automáticas)",
        "Barras empilhadas – faixas (quantis)",
    ]

tipo_grafico = st.sidebar.radio(
    "Escolha o tipo de gráfico",
    opcoes_grafico,
)

# ==============================
# 10. Abas para organização
# ==============================
tab1, tab2, tab3 = st.tabs(
    ["📈 Gráficos & Resumo", "📄 Dados Filtrados", "🔍 Detalhes da Conversão"]
)

# --------- Aba 1: KPIs, resumo e gráfico ----------
with tab1:
    st.subheader("📊 Indicadores principais (após filtros)")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total de Amostras", len(df_filtered))
    kpi2.metric(f"Média Geral ({col_y})", f"{df_filtered[col_y].mean():.2f}")
    kpi3.metric("Desvio Padrão", f"{df_filtered[col_y].std():.2f}")

    st.subheader(f"📈 Resumo de {col_y} por {col_treat}")
    st.dataframe(resumo)

    st.subheader("📉 Gráfico interativo")
    fig = None
    try:
        fig = create_plot(
            df_filtered=df_filtered,
            resumo=resumo,
            col_treat=col_treat,
            col_y=col_y,
            cols_y=cols_y,
            tipo_grafico=tipo_grafico,
            is_discrete_small=is_discrete_small,
        )
    except Exception as e:
        st.error(f"Erro ao gerar o gráfico: {e}")

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Não foi possível gerar o gráfico para essa combinação de opções/dados.")

    # 📥 Download dos dados FILTRADOS
    csv_filt_bytes = df_filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 Baixar dados FILTRADOS em CSV",
        data=csv_filt_bytes,
        file_name=f"dados_filtrados_{sheet_name}.csv",
        mime="text/csv",
    )

# --------- Aba 2: Dados filtrados ----------
with tab2:
    st.subheader("📄 Dados filtrados (após filtros e seleção de colunas)")
    st.dataframe(df_filtered)

# --------- Aba 3: Detalhes da conversão ----------
with tab3:
    st.subheader("🔍 Detalhes da conversão numérica")
    if invalid_info:
        st.markdown(
            "Algumas colunas foram convertidas para numérico com valores inválidos "
            "transformados em **NaN**:"
        )
        st.write(pd.DataFrame(invalid_info).T)
    else:
        st.write("Nenhuma conversão numérica problemática foi identificada.")

    st.subheader("Tipos de dados após conversão (df_conv)")
    st.write(df_conv.dtypes)
