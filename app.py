import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import Holt

st.set_page_config(page_title="Dashboard de Jogos", layout="wide")

st.title("🕹️ Análise de Vendas de Jogos")
st.markdown("Dados de vendas de jogos com análises descritivas e previsão para o próximo ano (1980-2016).")


@st.cache_data
def carregar_dados():
    df = pd.read_csv("vgsales.csv")
    df = df.dropna()
    df = df[(df["Year"] >= 1980) & (df["Year"] <= 2016)]
    df["Year"] = df["Year"].astype(int)
    return df


df = carregar_dados()

def prever_top_categoria_proximo_ano(df_filtrado, coluna_categoria, ano_max, top_n=5):
    ano_previsao = int(ano_max) + 1

    categorias_top = (
        df_filtrado.groupby(coluna_categoria)["Global_Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    previsoes = []
    ano_min_local = int(df_filtrado["Year"].min())

    for categoria in categorias_top:
        df_cat = df_filtrado[df_filtrado[coluna_categoria] == categoria]

        serie = (
            df_cat.groupby("Year")["Global_Sales"]
            .sum()
            .sort_index()
        )

        anos_completos = pd.Index(range(ano_min_local, int(ano_max) + 1), name="Year")
        serie = serie.reindex(anos_completos, fill_value=0)

        if len(serie) < 3 or serie.sum() == 0:
            previsao = float(serie.iloc[-1]) if len(serie) > 0 else 0.0
        else:
            try:
                modelo = Holt(serie).fit(optimized=True)
                previsao = float(modelo.forecast(1)[0])
                previsao = max(previsao, 0)
            except:
                previsao = float(serie.iloc[-1])

        previsoes.append({
            "Categoria": categoria,
            "Vendas_Previstas": previsao
        })

    previsoes_df = pd.DataFrame(previsoes).sort_values("Vendas_Previstas", ascending=False)

    top_categoria = previsoes_df.iloc[0]["Categoria"]
    top_valor = previsoes_df.iloc[0]["Vendas_Previstas"]

    return ano_previsao, previsoes_df, top_categoria, top_valor


def prever_top_regiao_proximo_ano(df_filtrado, ano_max):
    """
    Prevê qual região terá maior volume de vendas no ano seguinte ao ano_max,
    usando Holt nas séries anuais das regiões.
    """
    ano_previsao = int(ano_max) + 1
    ano_min_local = int(df_filtrado["Year"].min())

    colunas_regiao = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
    previsoes = []

    for regiao in colunas_regiao:
        serie = (
            df_filtrado.groupby("Year")[regiao]
            .sum()
            .sort_index()
        )

        anos_completos = pd.Index(range(ano_min_local, int(ano_max) + 1), name="Year")
        serie = serie.reindex(anos_completos, fill_value=0)

        if len(serie) < 3 or serie.sum() == 0:
            previsao = float(serie.iloc[-1]) if len(serie) > 0 else 0.0
        else:
            try:
                modelo = Holt(serie).fit(optimized=True)
                previsao = float(modelo.forecast(1)[0])
                previsao = max(previsao, 0)
            except:
                previsao = float(serie.iloc[-1])

        previsoes.append({
            "Regiao": regiao.replace("_Sales", ""),
            "Vendas_Previstas": previsao
        })

    previsoes_df = pd.DataFrame(previsoes).sort_values("Vendas_Previstas", ascending=False)

    top_regiao = previsoes_df.iloc[0]["Regiao"]
    top_valor = previsoes_df.iloc[0]["Vendas_Previstas"]

    return ano_previsao, previsoes_df, top_regiao, top_valor


# =========================================================
# FILTROS
# =========================================================
st.sidebar.header("🔎 Filtros")

plataformas = st.sidebar.multiselect(
    "Plataforma:",
    sorted(df["Platform"].unique()),
    default=sorted(df["Platform"].unique())
)

generos = st.sidebar.multiselect(
    "Gênero:",
    sorted(df["Genre"].unique()),
    default=sorted(df["Genre"].unique())
)

anos = st.sidebar.slider(
    "Período:",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (int(df["Year"].min()), int(df["Year"].max()))
)

metrica = st.sidebar.selectbox(
    "Tipo de vendas:",
    ["Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales"]
)

# separa os anos do slider
ano_min, ano_max = anos

# aplica filtros
df_filtrado = df[
    (df["Platform"].isin(plataformas)) &
    (df["Genre"].isin(generos)) &
    (df["Year"].between(ano_min, ano_max))
]

# tratamento de exceção
if df_filtrado.empty:
    st.warning("Nenhum dado encontrado com esses filtros.")
    st.stop()


# =========================================================
# PREVISÃO DO PRÓXIMO ANO
# =========================================================
ano_previsao = int(ano_max) + 1

# gênero previsto
ano_prev, previsoes_genero, genero_previsto_top, valor_genero_previsto = prever_top_categoria_proximo_ano(
    df_filtrado, "Genre", ano_max
)

# plataforma prevista
_, previsoes_plataforma, plataforma_prevista_top, valor_plataforma_prevista = prever_top_categoria_proximo_ano(
    df_filtrado, "Platform", ano_max
)

# região prevista
_, previsoes_regiao, regiao_prevista_top, valor_regiao_prevista = prever_top_regiao_proximo_ano(
    df_filtrado, ano_max
)


# =========================================================
# DADOS AGREGADOS PARA ANÁLISE DESCRITIVA
# =========================================================
vendas_genero = df_filtrado.groupby("Genre")[metrica].sum().reset_index()
vendas_genero = vendas_genero.sort_values(by=metrica, ascending=False)

top_plataformas = df_filtrado.groupby("Platform")[metrica].sum().reset_index()
top_plataformas = top_plataformas.sort_values(by=metrica, ascending=False)

vendas_ano = df_filtrado.groupby("Year")[metrica].sum().reset_index()

comparacao = df_filtrado.groupby("Year")[["NA_Sales", "EU_Sales", "JP_Sales"]].sum().reset_index()

# insights descritivos atuais
genero_top_atual = vendas_genero.iloc[0]["Genre"]
plataforma_top_atual = top_plataformas.iloc[0]["Platform"]
ano_pico = int(vendas_ano.loc[vendas_ano[metrica].idxmax()]["Year"])
regiao_top_atual = comparacao[["NA_Sales", "EU_Sales", "JP_Sales"]].sum().idxmax().replace("_Sales", "")


# =========================================================
# RESUMO GERAL
# =========================================================
st.subheader("📊 Resumo Geral")

col1, col2, col3 = st.columns(3)
col1.metric("Total de Jogos", len(df_filtrado))
col2.metric("Vendas Totais (mi)", round(df_filtrado[metrica].sum(), 2))
col3.metric("Média por Jogo (mi)", round(df_filtrado[metrica].mean(), 2))


# =========================================================
# INSIGHTS DESCRITIVOS
# =========================================================
st.markdown("### 📌 Principais Insights do período filtrado")

st.markdown(f"""
- **Gênero mais vendido no período:** {genero_top_atual}  
- **Plataforma mais vendida no período:** {plataforma_top_atual}  
- **Ano de pico de vendas:** {ano_pico}  
- **Região com mais vendas no período:** {regiao_top_atual}
""")


# =========================================================
# PREVISÃO DO PRÓXIMO ANO
# =========================================================
st.divider()
st.header("🔮 Expectativa para o próximo ano")

st.write(
    f"Com base no intervalo filtrado de **{ano_min} a {ano_max}**, "
    f"a expectativa para **{ano_prev}** é:"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Gênero mais vendido (previsto)",
        genero_previsto_top,
        f"{valor_genero_previsto:.2f} mi"
    )

with col2:
    st.metric(
        "Plataforma mais vendida (prevista)",
        plataforma_prevista_top,
        f"{valor_plataforma_prevista:.2f} mi"
    )

with col3:
    st.metric(
        "Região com mais vendas (prevista)",
        regiao_prevista_top,
        f"{valor_regiao_prevista:.2f} mi"
    )

with st.expander(f"Ver detalhes das previsões para {ano_prev}"):
    st.subheader("Previsão por gênero")
    st.dataframe(previsoes_genero)

    st.subheader("Previsão por plataforma")
    st.dataframe(previsoes_plataforma)

    st.subheader("Previsão por região")
    st.dataframe(previsoes_regiao)


# =========================================================
# GRÁFICOS
# =========================================================
st.divider()
st.subheader("🏆 Top 10 Jogos Mais Vendidos")

top10 = df_filtrado.sort_values(by=metrica, ascending=False).head(10)

fig1 = px.bar(
    top10,
    x=metrica,
    y="Name",
    orientation="h",
    color="Platform",
    title="Jogos com maiores vendas"
)
fig1.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig1, use_container_width=True)

st.divider()
st.subheader("📦 Vendas por Gênero")

fig2 = px.bar(
    vendas_genero,
    x="Genre",
    y=metrica,
    color="Genre",
    title="Total de vendas por gênero"
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("📅 Vendas ao Longo do Tempo")

fig3 = px.line(
    vendas_ano,
    x="Year",
    y=metrica,
    title="Evolução das vendas ao longo dos anos"
)
fig3.update_xaxes(dtick=1)
st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.subheader("🌎 Comparativo de Vendas por Região")

fig_regiao = px.line(
    comparacao,
    x="Year",
    y=["NA_Sales", "EU_Sales", "JP_Sales"],
    title="Vendas por Região ao Longo do Tempo"
)
st.plotly_chart(fig_regiao, use_container_width=True)

st.divider()
st.subheader("🎮 Top Plataformas")

top10_plataformas = top_plataformas.head(10)

fig4 = px.bar(
    top10_plataformas,
    x="Platform",
    y=metrica,
    color="Platform",
    title="Plataformas com maiores vendas"
)
st.plotly_chart(fig4, use_container_width=True)


# =========================================================
# TABELA
# =========================================================
st.subheader("📋 Tabela por Plataforma")

tabela = df_filtrado.groupby("Platform")[metrica].agg(["mean", "sum", "count"])
tabela = tabela.rename(columns={
    "mean": "Média",
    "sum": "Total",
    "count": "Quantidade"
})

st.dataframe(tabela)

st.divider()

with st.expander("🔍 Ver dados completos"):
    st.dataframe(df_filtrado)
