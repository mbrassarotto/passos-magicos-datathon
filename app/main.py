"""
Passos Mágicos — Aplicação de Previsão de Risco de Nova Defasagem
====================================================================

Aplicação Streamlit que disponibiliza o modelo preditivo treinado para a
Passos Mágicos, permitindo estimar o risco de um aluno sem defasagem
entrar em situação de defasagem no ano seguinte.

Datathon — Fase 5 — Pós Tech FIAP
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuração geral da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Passos Mágicos — Previsão de Risco",
    page_icon="🧭",
    layout="wide",
)

CAMINHO_BASE = Path(__file__).resolve().parent
CAMINHO_MODELO = CAMINHO_BASE / "data" / "modelo_risco_nova_defasagem.joblib"
CAMINHO_PREVISOES_2025 = CAMINHO_BASE / "data" / "previsoes_risco_2025.csv"

COLUNAS_ENTRADA_LOTE = [
    "ra",
    "idade",
    "fase_num",
    "ano",
    "ano_ingresso",
    "iaa",
    "ieg",
    "ips",
    "ida",
    "ipv",
    "nota_matematica",
    "nota_portugues",
]

ROTULOS_CAMPOS = {
    "idade": "Idade",
    "fase_num": "Fase atual (número)",
    "anos_no_programa": "Anos no programa",
    "ida": "IDA — Indicador de Aprendizagem",
    "ieg": "IEG — Indicador de Engajamento",
    "ipv": "IPV — Indicador do Ponto de Virada",
    "ips": "IPS — Indicador Psicossocial",
    "iaa": "IAA — Indicador de Autoavaliação",
    "nota_matematica": "Nota de Matemática",
    "nota_portugues": "Nota de Português",
}


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando modelo treinado...")
def carregar_artefato_modelo():
    """Carrega o pipeline treinado e os metadados salvos em disco."""
    if not CAMINHO_MODELO.exists():
        st.error(
            "Arquivo do modelo não encontrado em "
            f"`{CAMINHO_MODELO}`. Verifique se o arquivo "
            "`modelo_risco_nova_defasagem.joblib` está na pasta `data/`."
        )
        st.stop()
    return joblib.load(CAMINHO_MODELO)


@st.cache_data(show_spinner="Carregando estimativas de 2025...")
def carregar_previsoes_2025():
    """Carrega as previsões de risco de 2025 já geradas no notebook."""
    if not CAMINHO_PREVISOES_2025.exists():
        return None
    return pd.read_csv(CAMINHO_PREVISOES_2025)


def calcular_anos_no_programa(ano: pd.Series, ano_ingresso: pd.Series) -> pd.Series:
    """Calcula há quantos anos o aluno está no programa.

    Segue a mesma regra usada no treinamento do modelo: valores
    negativos (inconsistentes) são convertidos para ausente (NaN).
    """
    anos = ano - ano_ingresso
    anos = anos.where(anos >= 0, np.nan)
    return anos


def classificar_risco(probabilidade: float, limiar: float) -> str:
    """Classifica a probabilidade estimada em alerta ou sem alerta."""
    return "Alerta de risco" if probabilidade >= limiar else "Sem alerta"


def gerar_csv_modelo(df: pd.DataFrame) -> bytes:
    """Converte um DataFrame em bytes CSV para uso em botão de download."""
    return df.to_csv(index=False).encode("utf-8-sig")


# ---------------------------------------------------------------------------
# Carregamento do modelo (uma única vez, com cache)
# ---------------------------------------------------------------------------

artefato = carregar_artefato_modelo()
pipeline_modelo = artefato["pipeline"]
features_modelo = artefato["features_modelo"]
limiar_modelo = artefato["limiar"]
nome_modelo = artefato.get("nome_modelo", "Modelo preditivo")


# ---------------------------------------------------------------------------
# Barra lateral — navegação
# ---------------------------------------------------------------------------

st.sidebar.title("🧭 Passos Mágicos")
st.sidebar.caption("Previsão de risco de nova defasagem escolar")

pagina = st.sidebar.radio(
    "Navegação",
    options=[
        "Previsão individual",
        "Previsão em lote (CSV)",
        "Painel — Estimativas 2025",
        "Sobre o modelo",
    ],
)

st.sidebar.divider()
st.sidebar.markdown(
    f"**Modelo em uso:** {nome_modelo}  \n"
    f"**Limiar de decisão:** {limiar_modelo:.2f}"
)


# ---------------------------------------------------------------------------
# Página 1 — Previsão individual
# ---------------------------------------------------------------------------

if pagina == "Previsão individual":

    st.title("Previsão individual de risco")
    st.markdown(
        "Preencha os indicadores do aluno para estimar a probabilidade de "
        "ele **entrar em situação de defasagem no ano seguinte**, caso "
        "atualmente não esteja defasado."
    )

    with st.form("formulario_previsao_individual"):

        col_esquerda, col_direita = st.columns(2)

        with col_esquerda:
            idade = st.number_input(
                ROTULOS_CAMPOS["idade"], min_value=5, max_value=30, value=12, step=1
            )
            fase_num = st.number_input(
                ROTULOS_CAMPOS["fase_num"], min_value=0, max_value=9, value=3, step=1
            )
            ano_atual = st.number_input(
                "Ano de referência dos dados", min_value=2015, max_value=2035, value=2024, step=1
            )
            ano_ingresso = st.number_input(
                "Ano de ingresso no programa", min_value=2010, max_value=2035, value=2022, step=1
            )
            nota_matematica = st.slider(
                ROTULOS_CAMPOS["nota_matematica"], min_value=0.0, max_value=10.0, value=6.5, step=0.1
            )
            nota_portugues = st.slider(
                ROTULOS_CAMPOS["nota_portugues"], min_value=0.0, max_value=10.0, value=6.5, step=0.1
            )

        with col_direita:
            iaa = st.slider(
                ROTULOS_CAMPOS["iaa"], min_value=0.0, max_value=10.0, value=8.0, step=0.1
            )
            ieg = st.slider(
                ROTULOS_CAMPOS["ieg"], min_value=0.0, max_value=10.0, value=8.0, step=0.1
            )
            ips = st.slider(
                ROTULOS_CAMPOS["ips"], min_value=0.0, max_value=10.0, value=7.0, step=0.1
            )
            ida = st.slider(
                ROTULOS_CAMPOS["ida"], min_value=0.0, max_value=10.0, value=7.0, step=0.1
            )
            ipv = st.slider(
                ROTULOS_CAMPOS["ipv"], min_value=0.0, max_value=10.0, value=7.5, step=0.1
            )

        botao_calcular = st.form_submit_button("Calcular risco", type="primary")

    if botao_calcular:

        anos_no_programa = ano_atual - ano_ingresso

        if anos_no_programa < 0:
            st.warning(
                "O ano de ingresso é posterior ao ano de referência. "
                "Verifique os valores informados."
            )
        else:
            entrada = pd.DataFrame(
                [
                    {
                        "idade": idade,
                        "fase_num": fase_num,
                        "anos_no_programa": anos_no_programa,
                        "ida": ida,
                        "ieg": ieg,
                        "ipv": ipv,
                        "ips": ips,
                        "iaa": iaa,
                        "nota_matematica": nota_matematica,
                        "nota_portugues": nota_portugues,
                    }
                ]
            )[features_modelo]

            probabilidade = float(
                pipeline_modelo.predict_proba(entrada)[:, 1][0]
            )
            classificacao = classificar_risco(probabilidade, limiar_modelo)

            st.divider()
            st.subheader("Resultado da previsão")

            col_prob, col_classe = st.columns(2)

            with col_prob:
                st.metric(
                    "Probabilidade estimada de nova defasagem",
                    f"{probabilidade * 100:.1f}%",
                )
                st.progress(min(max(probabilidade, 0.0), 1.0))

            with col_classe:
                if classificacao == "Alerta de risco":
                    st.error(f"⚠️ {classificacao}")
                else:
                    st.success(f"✅ {classificacao}")
                st.caption(
                    f"Classificação baseada no limiar operacional de "
                    f"{limiar_modelo:.2f}, definido para priorizar recall "
                    "na identificação de alunos em risco."
                )

            st.info(
                "Esta estimativa deve ser utilizada como instrumento de "
                "**priorização preventiva**, apoiando o acompanhamento "
                "pedagógico, e não como uma determinação de que o aluno "
                "necessariamente entrará em defasagem."
            )


# ---------------------------------------------------------------------------
# Página 2 — Previsão em lote
# ---------------------------------------------------------------------------

elif pagina == "Previsão em lote (CSV)":

    st.title("Previsão em lote a partir de um arquivo CSV")
    st.markdown(
        "Envie um arquivo CSV contendo os indicadores de vários alunos para "
        "calcular o risco de nova defasagem de toda a turma de uma só vez."
    )

    st.markdown("**Colunas obrigatórias no arquivo:**")
    st.code(", ".join(COLUNAS_ENTRADA_LOTE), language="text")

    modelo_exemplo = pd.DataFrame(
        [
            {
                "ra": "RA-EXEMPLO",
                "idade": 12,
                "fase_num": 3,
                "ano": 2024,
                "ano_ingresso": 2022,
                "iaa": 8.5,
                "ieg": 8.0,
                "ips": 7.0,
                "ida": 7.0,
                "ipv": 7.5,
                "nota_matematica": 6.5,
                "nota_portugues": 6.5,
            }
        ]
    )

    st.download_button(
        "Baixar modelo de planilha (CSV)",
        data=gerar_csv_modelo(modelo_exemplo),
        file_name="modelo_previsao_em_lote.csv",
        mime="text/csv",
    )

    arquivo_enviado = st.file_uploader(
        "Selecione o arquivo CSV com os dados dos alunos", type=["csv"]
    )

    if arquivo_enviado is not None:

        try:
            dados_lote = pd.read_csv(arquivo_enviado)
        except Exception as erro:
            st.error(f"Não foi possível ler o arquivo enviado: {erro}")
            st.stop()

        colunas_ausentes = [
            coluna for coluna in COLUNAS_ENTRADA_LOTE if coluna not in dados_lote.columns
        ]

        if colunas_ausentes:
            st.error(
                "As seguintes colunas obrigatórias não foram encontradas no "
                f"arquivo: {colunas_ausentes}"
            )
        else:
            dados_lote = dados_lote.copy()

            for coluna in COLUNAS_ENTRADA_LOTE:
                if coluna != "ra":
                    dados_lote[coluna] = pd.to_numeric(
                        dados_lote[coluna], errors="coerce"
                    )

            dados_lote["anos_no_programa"] = calcular_anos_no_programa(
                dados_lote["ano"], dados_lote["ano_ingresso"]
            )

            linhas_validas = dados_lote[features_modelo].notna().all(axis=1)

            if not linhas_validas.all():
                st.warning(
                    f"{(~linhas_validas).sum()} linha(s) possuem valores "
                    "ausentes ou inválidos nas colunas exigidas pelo modelo "
                    "e não terão previsão calculada."
                )

            entrada_lote = dados_lote.loc[linhas_validas, features_modelo]

            probabilidades = pipeline_modelo.predict_proba(entrada_lote)[:, 1]

            resultado_lote = dados_lote.loc[linhas_validas].copy()
            resultado_lote["probabilidade_novo_risco"] = probabilidades
            resultado_lote["probabilidade_risco_percentual"] = (
                probabilidades * 100
            ).round(2)
            resultado_lote["alerta_risco"] = (
                probabilidades >= limiar_modelo
            ).astype(int)
            resultado_lote["classificacao"] = resultado_lote["alerta_risco"].map(
                {1: "Alerta de risco", 0: "Sem alerta"}
            )

            resultado_lote = resultado_lote.sort_values(
                "probabilidade_novo_risco", ascending=False
            ).reset_index(drop=True)

            st.divider()
            st.subheader("Resultado da previsão em lote")

            col1, col2, col3 = st.columns(3)
            col1.metric("Alunos avaliados", len(resultado_lote))
            col2.metric(
                "Alertas de risco", int(resultado_lote["alerta_risco"].sum())
            )
            col3.metric(
                "Percentual de alertas",
                f"{resultado_lote['alerta_risco'].mean() * 100:.1f}%",
            )

            st.dataframe(resultado_lote, width='stretch')

            st.download_button(
                "Baixar resultado completo (CSV)",
                data=gerar_csv_modelo(resultado_lote),
                file_name="previsao_risco_lote.csv",
                mime="text/csv",
                type="primary",
            )


# ---------------------------------------------------------------------------
# Página 3 — Painel com estimativas de 2025 já calculadas
# ---------------------------------------------------------------------------

elif pagina == "Painel — Estimativas 2025":

    st.title("Painel de risco — Estimativas para 2025")
    st.markdown(
        "Estimativas geradas a partir dos dados de 2024 para os alunos "
        "ainda sem defasagem, indicando o risco de entrarem em defasagem "
        "em 2025."
    )

    previsoes_2025 = carregar_previsoes_2025()

    if previsoes_2025 is None:
        st.warning(
            "Arquivo de estimativas de 2025 não encontrado em "
            f"`{CAMINHO_PREVISOES_2025}`."
        )
    else:
        with st.sidebar:
            st.divider()
            st.markdown("**Filtros do painel**")

            instituicoes = sorted(
                previsoes_2025["instituicao_padrao"].dropna().unique().tolist()
            )
            instituicoes_selecionadas = st.multiselect(
                "Instituição", options=instituicoes, default=instituicoes
            )

            generos = sorted(previsoes_2025["genero"].dropna().unique().tolist())
            generos_selecionados = st.multiselect(
                "Gênero", options=generos, default=generos
            )

            apenas_alertas = st.checkbox("Mostrar somente alunos em alerta", value=False)

        previsoes_filtradas = previsoes_2025[
            previsoes_2025["instituicao_padrao"].isin(instituicoes_selecionadas)
            & previsoes_2025["genero"].isin(generos_selecionados)
        ]

        if apenas_alertas:
            previsoes_filtradas = previsoes_filtradas[
                previsoes_filtradas["alerta_risco"] == 1
            ]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Alunos avaliados", len(previsoes_filtradas))
        col2.metric(
            "Alertas de risco", int(previsoes_filtradas["alerta_risco"].sum())
        )
        percentual_alertas = (
            previsoes_filtradas["alerta_risco"].mean() * 100
            if len(previsoes_filtradas) > 0
            else 0
        )
        col3.metric("Percentual de alertas", f"{percentual_alertas:.1f}%")
        col4.metric("Limiar utilizado", f"{limiar_modelo:.2f}")

        st.divider()

        col_grafico1, col_grafico2 = st.columns(2)

        with col_grafico1:
            st.markdown("**Alertas de risco por fase**")
            alertas_por_fase = (
                previsoes_filtradas.groupby("fase_num")["alerta_risco"]
                .sum()
                .sort_index()
            )
            st.bar_chart(alertas_por_fase)

        with col_grafico2:
            st.markdown("**Distribuição da probabilidade de risco**")
            faixas = pd.cut(
                previsoes_filtradas["probabilidade_risco_percentual"],
                bins=[0, 20, 40, 48, 60, 80, 100],
                include_lowest=True,
            )
            distribuicao = previsoes_filtradas.groupby(faixas.astype(str)).size()
            st.bar_chart(distribuicao)

        st.divider()
        st.markdown("**Tabela de estimativas**")
        st.dataframe(
            previsoes_filtradas.sort_values(
                "probabilidade_novo_risco", ascending=False
            ),
            width='stretch',
        )

        st.download_button(
            "Baixar tabela filtrada (CSV)",
            data=gerar_csv_modelo(previsoes_filtradas),
            file_name="estimativas_risco_2025_filtrado.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Página 4 — Sobre o modelo
# ---------------------------------------------------------------------------

elif pagina == "Sobre o modelo":

    st.title("Sobre o modelo preditivo")

    st.markdown(
        f"""
### Objetivo

Estimar se um aluno **atualmente sem defasagem** entrará em situação de
defasagem no ano seguinte, a partir dos indicadores educacionais coletados
no ano atual.

### Ficha técnica

- **Algoritmo selecionado:** {nome_modelo}
- **Limiar de decisão:** {limiar_modelo:.2f}
- **Definição do alvo:** {artefato.get("definicao_alvo", "—")}
- **População elegível:** {artefato.get("populacao_elegivel", "—")}
- **Estratégia de validação:** {artefato.get("validacao", "—")}
- **Critério de escolha do limiar:** {artefato.get("criterio_limiar", "—")}

### Variáveis utilizadas pelo modelo

| Variável | Descrição |
|---|---|
| Idade | Idade do aluno |
| Fase atual | Fase em que o aluno está matriculado |
| Anos no programa | Ano de referência menos o ano de ingresso |
| IDA | Indicador de Aprendizagem |
| IEG | Indicador de Engajamento |
| IPV | Indicador do Ponto de Virada |
| IPS | Indicador Psicossocial |
| IAA | Indicador de Autoavaliação |
| Nota de Matemática | Nota na disciplina de Matemática |
| Nota de Português | Nota na disciplina de Português |

### Desempenho observado no teste temporal (2023 → 2024)

- Acurácia: **74,60%**
- ROC-AUC: **0,846**
- PR-AUC: **0,684**
- Recall: **67,86%**
- Precisão: **52,29%**
- F1-score: **59,07%**
- Brier Score: **0,157**

### Limitações e uso recomendado

O modelo deve ser utilizado como uma ferramenta de **priorização
preventiva**, auxiliando na identificação de alunos que podem demandar
acompanhamento mais próximo — e não como uma determinação de que um aluno
necessariamente ficará defasado. A base histórica utilizada para
treinamento e validação possui apenas duas transições anuais completas
disponíveis (2022→2023 e 2023→2024).
"""
    )
