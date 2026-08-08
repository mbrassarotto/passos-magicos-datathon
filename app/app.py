"""
Datathon Passos Mágicos — Dashboard de Storytelling + Previsão de Risco

Como rodar localmente:
    pip install -r requirements.txt
    streamlit run app.py

Arquivos esperados na mesma pasta:
    - dados_pede.csv                       (base limpa, gerada pelo notebook 01)
    - modelo_gb_risco_temporal.joblib      (bundle do modelo, gerado por treinar_modelo_gb.py)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from treinar_modelo_gb import treinar_modelo

# --------------------------------------------------------------------------
# Configuração geral
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Datathon Passos Mágicos",
    page_icon="✨",
    layout="wide",
)

# Caminhos relativos à pasta deste arquivo (app.py), não ao diretório de
# trabalho do processo — no Streamlit Community Cloud, o app roda com o cwd
# na raiz do repositório, não necessariamente na pasta do script.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dados_pede.csv"

CORES_PEDRA = {
    "Quartzo": "#8d99ae",
    "Ágata": "#e07a5f",
    "Ametista": "#7b2cbf",
    "Topázio": "#f4a261",
}


@st.cache_data
def carregar_dados() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    colunas_numericas = [
        "ano", "idade", "ano_ingresso", "inde", "iaa", "ieg", "ips", "ipp",
        "ida", "nota_matematica", "nota_portugues", "nota_ingles", "ipv",
        "ian", "defasagem", "fase_num", "fase_ideal_num",
    ]
    for c in colunas_numericas:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def classificar_defasagem(v):
        if pd.isna(v):
            return "Não informado"
        if v > 0:
            return "Acima do esperado"
        if v == 0:
            return "Sem defasagem"
        if v == -1:
            return "Defasagem moderada"
        return "Defasagem severa"

    df["categoria_defasagem"] = df["defasagem"].apply(classificar_defasagem)
    return df


@st.cache_resource(show_spinner="Treinando o modelo de risco (leva alguns segundos)...")
def carregar_modelo():
    try:
        return treinar_modelo(DATA_PATH)
    except FileNotFoundError:
        return None


df = carregar_dados()
bundle = carregar_modelo()

ORDEM_DEFASAGEM = ["Acima do esperado", "Sem defasagem", "Defasagem moderada", "Defasagem severa", "Não informado"]
CORES_DEFASAGEM = {
    "Acima do esperado": "#2a9d8f",
    "Sem defasagem": "#457b9d",
    "Defasagem moderada": "#f4a261",
    "Defasagem severa": "#e63946",
    "Não informado": "#adb5bd",
}

st.title("✨ Datathon Passos Mágicos")
st.caption("Storytelling com dados (PEDE 2022–2024) e previsão de risco de defasagem")

abas = st.tabs([
    "Visão geral",
    "1. IAN",
    "2. IDA",
    "3. IEG",
    "4. IAA",
    "5. IPS",
    "6. IPP",
    "7. IPV",
    "8. Multidimensional",
    "9. Previsão de risco (ML)",
    "10. Efetividade",
    "11. Insights",
])

# --------------------------------------------------------------------------
# 0. Visão geral
# --------------------------------------------------------------------------
with abas[0]:
    st.subheader("Panorama geral da base")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros (aluno x ano)", f"{len(df):,}".replace(",", "."))
    c2.metric("Alunos únicos", df["ra"].nunique())
    c3.metric("Período", f"{int(df['ano'].min())}–{int(df['ano'].max())}")
    c4.metric("INDE médio (2024)", round(df.loc[df["ano"] == 2024, "inde"].mean(), 2))

    col1, col2 = st.columns(2)
    with col1:
        registros_ano = df.groupby("ano").size().reset_index(name="alunos")
        fig = px.bar(registros_ano, x="ano", y="alunos", text="alunos",
                     title="Alunos avaliados por ano")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        indicadores = ["inde", "ian", "ida", "ieg", "iaa", "ips"]
        medias = df.groupby("ano")[indicadores].mean().round(2).reset_index()
        medias_long = medias.melt(id_vars="ano", var_name="indicador", value_name="media")
        fig = px.line(medias_long, x="ano", y="media", color="indicador", markers=True,
                      title="Evolução das médias dos principais indicadores")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "Use as abas acima para navegar pelas 11 perguntas de negócio do desafio. "
        "A aba **9. Previsão de risco (ML)** permite simular o risco de um aluno "
        "entrar em defasagem no ano seguinte."
    )

# --------------------------------------------------------------------------
# 1. IAN — adequação de nível
# --------------------------------------------------------------------------
with abas[1]:
    st.subheader("Pergunta 1 — Adequação de nível (IAN) e defasagem")
    st.markdown(
        "Classificação operacional: **Acima do esperado** (fase acima da ideal), "
        "**Sem defasagem** (fase = ideal), **Defasagem moderada** (1 fase abaixo), "
        "**Defasagem severa** (2+ fases abaixo)."
    )

    dist = (
        df.groupby(["ano", "categoria_defasagem"]).size().reset_index(name="qtd")
    )
    dist["percentual"] = dist.groupby("ano")["qtd"].transform(lambda s: s / s.sum() * 100)

    fig = px.bar(
        dist, x="ano", y="percentual", color="categoria_defasagem",
        category_orders={"categoria_defasagem": ORDEM_DEFASAGEM},
        color_discrete_map=CORES_DEFASAGEM,
        barmode="stack", text=dist["percentual"].round(1),
        title="Distribuição percentual da defasagem por ano",
        labels={"percentual": "% de alunos", "ano": "Ano", "categoria_defasagem": "Categoria"},
    )
    fig.update_traces(texttemplate="%{text}%", textposition="inside")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    p2022 = dist[dist["ano"] == 2022].set_index("categoria_defasagem")["percentual"]
    p2024 = dist[dist["ano"] == 2024].set_index("categoria_defasagem")["percentual"]
    col1.metric("Defasagem moderada+severa (2022)",
                f"{p2022.get('Defasagem moderada', 0) + p2022.get('Defasagem severa', 0):.1f}%")
    col2.metric("Defasagem moderada+severa (2024)",
                f"{p2024.get('Defasagem moderada', 0) + p2024.get('Defasagem severa', 0):.1f}%",
                delta=f"{(p2024.get('Defasagem moderada', 0) + p2024.get('Defasagem severa', 0)) - (p2022.get('Defasagem moderada', 0) + p2022.get('Defasagem severa', 0)):.1f} p.p.")

    st.markdown(
        "**Leitura:** a defasagem severa caiu de forma expressiva entre 2022 e 2024, "
        "e a fatia de alunos sem defasagem (ou acima do esperado) cresceu, indicando "
        "melhora geral na adequação de nível ao longo do ciclo."
    )

# --------------------------------------------------------------------------
# 2. IDA — desempenho acadêmico
# --------------------------------------------------------------------------
with abas[2]:
    st.subheader("Pergunta 2 — Desempenho acadêmico (IDA)")

    col1, col2 = st.columns(2)
    with col1:
        resumo = df.groupby("ano")["ida"].agg(["mean", "median", "std"]).round(2).reset_index()
        resumo.columns = ["ano", "média", "mediana", "desvio padrão"]
        fig = px.line(resumo, x="ano", y=["média", "mediana"], markers=True,
                      title="IDA médio e mediano por ano")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(resumo, hide_index=True, use_container_width=True)
    with col2:
        fig = px.box(df.dropna(subset=["ida"]), x="ano", y="ida",
                     title="Distribuição do IDA por ano")
        st.plotly_chart(fig, use_container_width=True)

    ida_fase = (
        df.dropna(subset=["ida", "fase_num"])
        .groupby(["ano", "fase_num"])["ida"].mean().reset_index()
    )
    fig = px.line(ida_fase, x="fase_num", y="ida", color="ano", markers=True,
                  title="IDA médio por fase e ano")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Leitura:** o IDA médio subiu de 2022 para 2023, mas recuou parcialmente em 2024 "
        "(embora ainda acima do nível de 2022). A dispersão (desvio-padrão) aumentou em 2024, "
        "indicando maior heterogeneidade de desempenho entre os alunos."
    )

# --------------------------------------------------------------------------
# 3. IEG — engajamento
# --------------------------------------------------------------------------
with abas[3]:
    st.subheader("Pergunta 3 — Engajamento (IEG) x Desempenho (IDA) x Ponto de virada (IPV)")

    corr_por_ano = (
        df.dropna(subset=["ieg", "ida", "ipv"])
        .groupby("ano")[["ieg", "ida", "ipv"]]
        .apply(lambda g: pd.Series({
            "corr_IEG_IDA": g["ieg"].corr(g["ida"]),
            "corr_IEG_IPV": g["ieg"].corr(g["ipv"]),
        }))
        .round(2)
        .reset_index()
    )
    st.dataframe(corr_por_ano, hide_index=True, use_container_width=True)

    def faixa_ieg(v):
        if pd.isna(v):
            return "Não informado"
        if v < 5:
            return "Baixo engajamento"
        if v < 7.5:
            return "Médio engajamento"
        return "Alto engajamento"

    tmp = df.copy()
    tmp["faixa_ieg"] = tmp["ieg"].apply(faixa_ieg)
    resumo_faixa = (
        tmp[tmp["faixa_ieg"] != "Não informado"]
        .groupby("faixa_ieg")[["ida", "ipv"]].mean().round(2)
        .reindex(["Baixo engajamento", "Médio engajamento", "Alto engajamento"])
        .reset_index()
    )
    fig = px.bar(resumo_faixa.melt(id_vars="faixa_ieg", var_name="indicador", value_name="média"),
                 x="faixa_ieg", y="média", color="indicador", barmode="group",
                 title="IDA e IPV médios por faixa de engajamento (IEG)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Leitura:** existe associação positiva e moderada entre engajamento (IEG) e os "
        "indicadores de desempenho (IDA) e ponto de virada (IPV) nos três anos — quanto maior "
        "o engajamento, maiores tendem a ser IDA e IPV, mas a relação não é fortíssima "
        "(correlações na faixa de 0,4 a 0,6), sugerindo que engajamento é um fator relevante, "
        "mas não o único."
    )

# --------------------------------------------------------------------------
# 4. IAA — autoavaliação
# --------------------------------------------------------------------------
with abas[4]:
    st.subheader("Pergunta 4 — Autoavaliação (IAA) x Desempenho (IDA) e Engajamento (IEG)")

    validos = df.dropna(subset=["iaa", "ida", "ieg"])
    corr_iaa_ida = validos["iaa"].corr(validos["ida"])
    corr_iaa_ieg = validos["iaa"].corr(validos["ieg"])

    col1, col2 = st.columns(2)
    col1.metric("Correlação IAA x IDA", round(corr_iaa_ida, 2))
    col2.metric("Correlação IAA x IEG", round(corr_iaa_ieg, 2))

    fig = px.scatter(validos, x="iaa", y="ida", color="ano", opacity=0.5, trendline="ols",
                      title="IAA (autoavaliação) x IDA (desempenho real)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Leitura:** as correlações são positivas, porém fracas (em torno de 0,1). "
        "Isso indica baixa consistência entre a percepção que o aluno tem de si mesmo e "
        "seu desempenho real — muitos alunos com autoavaliação alta não necessariamente "
        "têm desempenho ou engajamento proporcionalmente altos, e vice-versa."
    )

# --------------------------------------------------------------------------
# 5. IPS — psicossocial
# --------------------------------------------------------------------------
with abas[5]:
    st.subheader("Pergunta 5 — Aspectos psicossociais (IPS) antecedendo quedas de desempenho")

    pares = []
    for ano_ini, ano_fim in [(2022, 2023), (2023, 2024)]:
        base_ini = df[df["ano"] == ano_ini][["ra", "ips", "ida", "ieg"]].rename(
            columns={"ips": "ips_ini", "ida": "ida_ini", "ieg": "ieg_ini"})
        base_fim = df[df["ano"] == ano_fim][["ra", "ida", "ieg"]].rename(
            columns={"ida": "ida_fim", "ieg": "ieg_fim"})
        par = base_ini.merge(base_fim, on="ra", how="inner")
        par["queda_ida"] = par["ida_fim"] < par["ida_ini"]
        par["periodo"] = f"{ano_ini}→{ano_fim}"
        pares.append(par)
    pares_df = pd.concat(pares, ignore_index=True).dropna(subset=["ips_ini"])

    resumo = pares_df.groupby(["periodo", "queda_ida"])["ips_ini"].mean().round(2).reset_index()
    resumo["queda_ida"] = resumo["queda_ida"].map({True: "Com queda no IDA", False: "Sem queda no IDA"})

    fig = px.bar(resumo, x="periodo", y="ips_ini", color="queda_ida", barmode="group",
                 title="IPS inicial médio: alunos com e sem queda posterior no IDA")
    st.plotly_chart(fig, use_container_width=True)

    def faixa_ips(v):
        if pd.isna(v):
            return "Não informado"
        if v < 5:
            return "Baixo"
        if v < 7.5:
            return "Médio"
        return "Alto"

    pares_df["faixa_ips"] = pares_df["ips_ini"].apply(faixa_ips)
    prop_queda = (
        pares_df[pares_df["faixa_ips"] != "Não informado"]
        .groupby(["periodo", "faixa_ips"])["queda_ida"].mean().mul(100).round(1)
        .reset_index()
    )
    fig2 = px.bar(prop_queda, x="faixa_ips", y="queda_ida", color="periodo", barmode="group",
                  category_orders={"faixa_ips": ["Baixo", "Médio", "Alto"]},
                  labels={"queda_ida": "% de alunos com queda no IDA"},
                  title="Percentual de alunos com queda no IDA, por faixa de IPS inicial")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "**Leitura:** as médias de IPS inicial são próximas entre quem teve e quem não teve "
        "queda posterior de IDA, e a proporção de quedas não muda de forma consistente entre "
        "as faixas de IPS. Ou seja, **não há um padrão psicossocial claro que antecipe quedas** "
        "de desempenho ou engajamento nesta base."
    )

# --------------------------------------------------------------------------
# 6. IPP — psicopedagógico
# --------------------------------------------------------------------------
with abas[6]:
    st.subheader("Pergunta 6 — Avaliação psicopedagógica (IPP) x Adequação de nível (IAN)")
    st.caption("O IPP só está disponível em 2023 e 2024.")

    validos = df.dropna(subset=["ipp", "ian"])
    corr_pearson = validos["ipp"].corr(validos["ian"])
    corr_spearman = validos["ipp"].corr(validos["ian"], method="spearman")

    col1, col2 = st.columns(2)
    col1.metric("Correlação de Pearson (IPP x IAN)", round(corr_pearson, 2))
    col2.metric("Correlação de Spearman (IPP x IAN)", round(corr_spearman, 2))

    ipp_por_ian = validos.groupby("ian")["ipp"].mean().round(2).reset_index()
    fig = px.bar(ipp_por_ian, x="ian", y="ipp", title="IPP médio por nível de IAN")
    st.plotly_chart(fig, use_container_width=True)

    ipp_por_categoria = (
        df.dropna(subset=["ipp"])
        .groupby("categoria_defasagem")["ipp"].mean().round(2)
        .reindex(ORDEM_DEFASAGEM[:-1])
        .dropna()
        .reset_index()
    )
    fig2 = px.bar(ipp_por_categoria, x="categoria_defasagem", y="ipp",
                  color="categoria_defasagem", color_discrete_map=CORES_DEFASAGEM,
                  title="IPP médio por categoria de defasagem")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "**Leitura:** a relação entre IPP e IAN é positiva, porém fraca (correlação em torno "
        "de 0,12–0,13). O IPP médio cresce à medida que o IAN aumenta e cai à medida que a "
        "defasagem se agrava — direção coerente com a expectativa, mas sem força suficiente "
        "para dizer que o IPP **confirma** a defasagem medida pelo IAN de forma robusta."
    )

# --------------------------------------------------------------------------
# 7. IPV — ponto de virada
# --------------------------------------------------------------------------
with abas[7]:
    st.subheader("Pergunta 7 — Fatores associados ao ponto de virada (IPV)")

    indicadores_ipv = ["ida", "ieg", "iaa", "ips", "ipp"]
    correlacoes = (
        df[["ipv"] + indicadores_ipv].corr()["ipv"].drop("ipv").sort_values(ascending=False).round(2)
    )
    fig = px.bar(correlacoes.reset_index().rename(columns={"index": "indicador", "ipv": "correlação"}),
                 x="indicador", y="correlação",
                 title="Correlação de cada indicador com o IPV (mesmo ano)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Leitura:** os indicadores mais associados ao IPV no mesmo ano são o **IPP**, o "
        "**IEG** e o **IDA**. O IPP tem a correlação mais alta, mas só está disponível em "
        "2023–2024. Olhando a evolução ano a ano dos mesmos alunos, as variações de **IDA** e "
        "**IEG** são as que mais acompanham as variações do IPV — ou seja, melhorar desempenho "
        "e engajamento tende a vir acompanhado de avanço no ponto de virada."
    )

# --------------------------------------------------------------------------
# 8. Multidimensionalidade — INDE
# --------------------------------------------------------------------------
with abas[8]:
    st.subheader("Pergunta 8 — Combinações de indicadores que elevam o INDE")

    base = df.dropna(subset=["ida", "ieg", "ips", "ipp", "inde"]).copy()
    for col in ["ida", "ieg", "ips", "ipp"]:
        limite = base[col].quantile(0.75)
        base[f"{col}_alto"] = base[col] >= limite

    base["qtd_indicadores_altos"] = base[["ida_alto", "ieg_alto", "ips_alto", "ipp_alto"]].sum(axis=1)
    resumo = base.groupby("qtd_indicadores_altos")["inde"].agg(["mean", "count"]).round(2).reset_index()
    resumo.columns = ["qtd_indicadores_altos", "inde_medio", "n_alunos"]

    fig = px.bar(resumo, x="qtd_indicadores_altos", y="inde_medio", text="n_alunos",
                 title="INDE médio conforme a quantidade de indicadores 'altos' (IDA, IEG, IPS, IPP)",
                 labels={"qtd_indicadores_altos": "Quantidade de indicadores no topo (≥ 3º quartil)",
                         "inde_medio": "INDE médio"})
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    combinacoes = (
        base.groupby(["ida_alto", "ieg_alto"])["inde"].mean().round(2).reset_index()
    )
    combinacoes["combinação"] = combinacoes.apply(
        lambda r: f"IDA {'alto' if r['ida_alto'] else 'baixo'} / IEG {'alto' if r['ieg_alto'] else 'baixo'}", axis=1
    )
    fig2 = px.bar(combinacoes, x="combinação", y="inde", title="INDE médio por combinação IDA x IEG")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "**Leitura:** o INDE cresce de forma praticamente monotônica conforme aumenta a "
        "quantidade de indicadores no topo da distribuição. A combinação **IDA alto + IEG "
        "alto** se destaca — ter esses dois indicadores elevados ao mesmo tempo puxa o INDE "
        "para cima de forma mais consistente do que qualquer indicador isolado."
    )

# --------------------------------------------------------------------------
# 9. Previsão de risco (ML)
# --------------------------------------------------------------------------
with abas[9]:
    st.subheader("Pergunta 9 — Previsão de risco de nova defasagem")

    if bundle is None:
        st.error(
            f"Não encontrei o arquivo `{DATA_PATH.name}` nesta pasta para treinar o modelo. "
            "Confirme que ele está na mesma pasta do app.py."
        )
    else:
        st.markdown(bundle["target_definition"])
        st.caption(
            f"Modelo: **{bundle['modelo']}** • Limiar de alerta: **{bundle['limiar']:.2f}** • "
            f"{bundle['validacao']}"
        )

        with st.expander("Métricas do modelo no teste temporal (2023 → 2024)"):
            met = bundle["metrics_teste_temporal"]
            cols = st.columns(len(met))
            nomes = {
                "acuracia": "Acurácia", "precisao": "Precisão", "recall": "Recall",
                "f1": "F1-score", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier",
            }
            for c, (k, v) in zip(cols, met.items()):
                c.metric(nomes.get(k, k), f"{v:.2f}")

        modo = st.radio("Como você quer prever?", ["Um aluno (formulário)", "Vários alunos (upload de CSV)"], horizontal=True)

        pipeline = bundle["pipeline"]
        num_features = bundle["num_features"]
        cat_features = bundle["cat_features"]
        limiar = bundle["limiar"]

        if modo == "Um aluno (formulário)":
            st.markdown("##### Dados do aluno")
            entrada = {}
            col1, col2, col3 = st.columns(3)
            colunas_ui = [col1, col2, col3]
            for i, feat in enumerate(num_features):
                lo, hi = bundle["ranges_numeric"].get(feat, (0.0, 10.0))
                default = bundle["defaults_numeric"].get(feat, (lo + hi) / 2)
                with colunas_ui[i % 3]:
                    entrada[feat] = st.number_input(
                        feat.replace("_", " ").capitalize(), value=float(round(default, 1)),
                        min_value=float(np.floor(lo)),
                        step=0.1,
                    )

            st.markdown("##### Perfil do aluno")
            col1, col2, col3 = st.columns(3)
            colunas_ui = [col1, col2, col3]
            for i, feat in enumerate(cat_features):
                opcoes = bundle["categories_cat"].get(feat, [])
                default = bundle["defaults_cat"].get(feat, opcoes[0] if opcoes else "")
                with colunas_ui[i % 3]:
                    idx = opcoes.index(default) if default in opcoes else 0
                    entrada[feat] = st.selectbox(feat.replace("_", " ").capitalize(), opcoes, index=idx)

            if st.button("Calcular risco", type="primary"):
                X_novo = pd.DataFrame([entrada])[num_features + cat_features]
                proba = pipeline.predict_proba(X_novo)[0, 1]
                alerta = proba >= limiar

                st.markdown("---")
                c1, c2 = st.columns([1, 2])
                with c1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=proba * 100,
                        number={"suffix": "%"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#e63946" if alerta else "#2a9d8f"},
                            "steps": [
                                {"range": [0, limiar * 100], "color": "#d8f3dc"},
                                {"range": [limiar * 100, 100], "color": "#ffccd5"},
                            ],
                            "threshold": {"line": {"color": "black", "width": 3}, "value": limiar * 100},
                        },
                        title={"text": "Probabilidade de nova defasagem"},
                    ))
                    fig.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    if alerta:
                        st.error(
                            f"**Alerta de risco.** Probabilidade estimada de {proba:.1%} de o "
                            f"aluno entrar em defasagem no próximo ano (limiar de decisão: {limiar:.0%})."
                        )
                    else:
                        st.success(
                            f"**Sem alerta no momento.** Probabilidade estimada de {proba:.1%}, "
                            f"abaixo do limiar de decisão ({limiar:.0%})."
                        )
                    st.caption(
                        "Este resultado é um score de priorização preventiva — não é uma "
                        "determinação de que o aluno necessariamente ficará defasado."
                    )
        else:
            st.markdown(
                "Envie um CSV com as colunas: `" + "`, `".join(num_features + cat_features) + "`"
            )
            arquivo = st.file_uploader("Arquivo CSV", type=["csv"])
            if arquivo is not None:
                novo = pd.read_csv(arquivo)
                faltando = [c for c in num_features + cat_features if c not in novo.columns]
                if faltando:
                    st.error(f"Faltam colunas no arquivo: {faltando}")
                else:
                    X_novo = novo[num_features + cat_features].copy()
                    for c in cat_features:
                        X_novo[c] = X_novo[c].astype(str)
                    probas = pipeline.predict_proba(X_novo)[:, 1]
                    novo["probabilidade_risco"] = probas
                    novo["alerta_risco"] = (probas >= limiar).astype(int)
                    novo = novo.sort_values("probabilidade_risco", ascending=False)

                    st.success(f"{int(novo['alerta_risco'].sum())} de {len(novo)} alunos com alerta de risco.")
                    st.dataframe(
                        novo[["probabilidade_risco", "alerta_risco"] + [c for c in novo.columns if c not in ("probabilidade_risco", "alerta_risco")]],
                        use_container_width=True,
                    )
                    st.download_button(
                        "Baixar resultado (CSV)",
                        novo.to_csv(index=False).encode("utf-8-sig"),
                        file_name="previsoes_risco.csv",
                        mime="text/csv",
                    )

# --------------------------------------------------------------------------
# 10. Efetividade do programa
# --------------------------------------------------------------------------
with abas[10]:
    st.subheader("Pergunta 10 — Efetividade do programa por pedra (Quartzo, Ágata, Ametista, Topázio)")

    ordem_pedra = ["Quartzo", "Ágata", "Ametista", "Topázio"]
    dist_pedra = (
        df.dropna(subset=["pedra_padrao"])
        .groupby(["ano", "pedra_padrao"]).size().reset_index(name="qtd")
    )
    dist_pedra["percentual"] = dist_pedra.groupby("ano")["qtd"].transform(lambda s: s / s.sum() * 100)

    fig = px.bar(dist_pedra, x="ano", y="percentual", color="pedra_padrao",
                 category_orders={"pedra_padrao": ordem_pedra}, color_discrete_map=CORES_PEDRA,
                 barmode="stack", title="Distribuição de pedra por ano (transversal)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Matriz de transição — 2023 → 2024 (% por linha)")
    p23 = df[df["ano"] == 2023][["ra", "pedra_padrao"]].rename(columns={"pedra_padrao": "pedra_2023"})
    p24 = df[df["ano"] == 2024][["ra", "pedra_padrao"]].rename(columns={"pedra_padrao": "pedra_2024"})
    transicao = p23.merge(p24, on="ra", how="inner").dropna()
    matriz = pd.crosstab(transicao["pedra_2023"], transicao["pedra_2024"], normalize="index").mul(100).round(1)
    matriz = matriz.reindex(index=[p for p in ordem_pedra if p in matriz.index],
                             columns=[p for p in ordem_pedra if p in matriz.columns])
    fig2 = px.imshow(matriz, text_auto=True, color_continuous_scale="Blues",
                      labels=dict(x="Pedra em 2024", y="Pedra em 2023", color="%"),
                      title="Para onde os alunos foram entre 2023 e 2024")
    st.plotly_chart(fig2, use_container_width=True)

    ind_pedra_ano = (
        df.dropna(subset=["pedra_padrao"])
        .groupby(["pedra_padrao", "ano"])[["inde"]].mean().round(2).reset_index()
    )
    fig3 = px.line(ind_pedra_ano, x="ano", y="inde", color="pedra_padrao",
                   category_orders={"pedra_padrao": ordem_pedra}, color_discrete_map=CORES_PEDRA,
                   markers=True, title="INDE médio por pedra e ano")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        "**Leitura:** a evidência é mista. A favor da efetividade: a fatia de alunos em "
        "Topázio cresce e em Quartzo cai ano a ano, e a defasagem severa também diminui. "
        "Contra/inconclusivo: olhando os mesmos alunos ao longo do tempo, subir e descer de "
        "pedra acontece nos dois sentidos, e a matriz de transição mostra que uma parte "
        "relevante dos alunos permanece na mesma pedra ou até recua — o que sugere melhora "
        "geral do programa, mas não uma trajetória linear garantida para cada aluno."
    )

# --------------------------------------------------------------------------
# 11. Insights adicionais
# --------------------------------------------------------------------------
with abas[11]:
    st.subheader("Pergunta 11 — Insights adicionais")

    st.markdown("##### Mapa de correlação entre indicadores (2024)")
    indicadores_corr = ["inde", "ian", "ida", "ieg", "iaa", "ips", "ipp", "ipv",
                         "nota_matematica", "nota_portugues", "nota_ingles"]
    corr = df[df["ano"] == 2024][indicadores_corr].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Correlação entre indicadores — 2024")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### INDE por indicação de bolsa")
        if "indicado_bolsa" in df.columns:
            resumo = df.dropna(subset=["indicado_bolsa", "inde"]).groupby("indicado_bolsa")["inde"].mean().round(2).reset_index()
            fig = px.bar(resumo, x="indicado_bolsa", y="inde", title="INDE médio por indicação de bolsa")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("##### INDE por gênero")
        resumo_genero = df.dropna(subset=["genero", "inde"]).groupby("genero")["inde"].mean().round(2).reset_index()
        fig = px.bar(resumo_genero, x="genero", y="inde", title="INDE médio por gênero")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Tempo de programa x INDE")
    tmp = df.dropna(subset=["ano_ingresso", "inde"]).copy()
    tmp["anos_no_programa"] = tmp["ano"] - tmp["ano_ingresso"]
    tmp = tmp[tmp["anos_no_programa"] >= 0]
    resumo_tempo = tmp.groupby("anos_no_programa")["inde"].agg(["mean", "count"]).round(2).reset_index()
    resumo_tempo.columns = ["anos_no_programa", "inde_medio", "n_alunos"]
    resumo_tempo = resumo_tempo[resumo_tempo["n_alunos"] >= 5]
    fig = px.bar(resumo_tempo, x="anos_no_programa", y="inde_medio", text="n_alunos",
                 title="INDE médio conforme o tempo de permanência no programa")
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Sugestões para a Passos Mágicos:**\n"
        "- Priorizar acompanhamento próximo para alunos sinalizados pelo modelo de risco "
        "(aba 9), especialmente aqueles com fase avançada e IPV/IDA mais baixos.\n"
        "- Investigar mais a fundo os alunos que oscilam entre pedras (matriz de transição), "
        "pois representam o grupo mais instável ano a ano.\n"
        "- Como IAA tem baixa relação com o desempenho real, vale revisar se o instrumento "
        "de autoavaliação está captando bem a percepção dos alunos, ou complementar com "
        "outras formas de escuta.\n"
        "- Reforçar o registro do IPP em todos os anos (hoje ausente em 2022), já que ele "
        "aparece como um dos indicadores mais associados ao ponto de virada."
    )
