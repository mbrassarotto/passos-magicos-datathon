"""
Treina o modelo de RISCO FUTURO de defasagem (temporal) usando Gradient Boosting,
com variáveis numéricas e categóricas.

Pergunta de negócio (Datathon, item 9):
"Construa um modelo preditivo que mostre a probabilidade do aluno entrar em
risco de defasagem."

Definição do alvo (igual ao notebook 03_modelagem_temporal_risco_defasagem):
- Usa-se o ano N para prever a situação do aluno no ano N+1.
- Só entram no treino/aplicação alunos SEM defasagem no ano N (defasagem >= 0)
  e em fase < 8 (fases finais não são comparáveis).
- Alvo = 1 se o aluno, sem defasagem em N, PASSOU a ter defasagem em N+1.

Diferenças em relação ao notebook 03 (que usava só variáveis numéricas + Extra Trees):
- Aqui usamos Gradient Boosting.
- Incluímos variáveis categóricas (gênero, instituição, flags de inconsistência).
- Mantemos fase_num e anos_no_programa como preditores porque, no contexto
  TEMPORAL (prever o ano seguinte), eles são legítimos: não vazam o alvo,
  pois são conhecidos no momento da previsão (diferente de usar a defasagem
  atual/IAN/pedra, que são excluídos por vazamento).

Uso:
    python treinar_modelo_gb.py --data dados_pede.csv --out modelo_gb_risco_temporal.joblib
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

NUM_FEATURES = [
    "idade", "fase_num", "anos_no_programa", "ida", "ieg", "ipv", "ips",
    "ipp", "iaa", "nota_matematica", "nota_portugues", "nota_ingles",
]
CAT_FEATURES = [
    "genero", "instituicao_padrao",
    "genero_inconsistente", "ano_ingresso_inconsistente", "idade_inconsistente",
]

TARGET_DEFINITION = (
    "Alvo binário **alvo_novo_risco**: entre alunos SEM defasagem no ano atual "
    "(defasagem >= 0, fase < 8), sinaliza 1 quando o aluno PASSA a ter defasagem "
    "(defasagem < 0) no ano seguinte. Colunas que vazam informação do alvo (ian, "
    "inde, pedra, fase_ideal, defasagem*) não são usadas como preditoras."
)


def build_temporal_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    colunas_numericas_base = [
        "ano", "idade", "ano_ingresso", "fase_num", "iaa", "ieg", "ips",
        "ipp", "ida", "nota_matematica", "nota_portugues", "nota_ingles",
        "ipv", "defasagem",
    ]
    for c in colunas_numericas_base:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["anos_no_programa"] = df["ano"] - df["ano_ingresso"]
    df.loc[df["anos_no_programa"] < 0, "anos_no_programa"] = np.nan

    base_futura = df[["ra", "ano", "defasagem"]].copy()
    base_futura["ano"] = base_futura["ano"] - 1
    base_futura = base_futura.rename(columns={"defasagem": "defasagem_proximo_ano"})

    base = df.merge(base_futura, on=["ra", "ano"], how="inner", validate="one_to_one")
    base = base.dropna(subset=["defasagem", "defasagem_proximo_ano"]).copy()

    base = base[(base["defasagem"] >= 0) & (base["fase_num"] < 8)].copy()
    base["alvo_novo_risco"] = (base["defasagem_proximo_ano"] < 0).astype(int)

    for c in CAT_FEATURES:
        base[c] = base[c].astype(str).replace({"nan": "Não informado", "None": "Não informado"})

    return base, df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUM_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT_FEATURES),
    ])
    clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def compute_form_helpers(X: pd.DataFrame) -> dict:
    defaults_numeric = {c: float(X[c].median()) for c in NUM_FEATURES}
    ranges_numeric = {
        c: (float(X[c].min()), float(X[c].max())) for c in NUM_FEATURES if X[c].notna().any()
    }
    defaults_cat = {c: X[c].mode().iloc[0] for c in CAT_FEATURES}
    categories_cat = {c: sorted(X[c].dropna().unique().tolist()) for c in CAT_FEATURES}
    return {
        "defaults_numeric": defaults_numeric,
        "ranges_numeric": ranges_numeric,
        "defaults_cat": defaults_cat,
        "categories_cat": categories_cat,
    }


def treinar_modelo(csv_path: Path) -> dict:
    """Executa todo o pipeline de treino e retorna o bundle em memória
    (sem passar por serialização joblib — evita incompatibilidade de
    versões de numpy/scikit-learn entre ambientes)."""
    print(f"Lendo dados de: {csv_path}")
    base, df_completo = build_temporal_dataset(Path(csv_path))
    print(f"Registros elegíveis (transições ano->ano+1): {len(base)}")

    features = NUM_FEATURES + CAT_FEATURES
    target = "alvo_novo_risco"

    # Split temporal, igual ao notebook 03: treino 2022->2023, teste 2023->2024
    treino = base[base["ano"] == 2022].copy()
    teste = base[base["ano"] == 2023].copy()

    X_treino, y_treino = treino[features].copy(), treino[target].copy()
    X_teste, y_teste = teste[features].copy(), teste[target].copy()

    print(f"Treino: {len(treino)} alunos | {int(y_treino.sum())} casos de novo risco")
    print(f"Teste:  {len(teste)} alunos | {int(y_teste.sum())} casos de novo risco")

    pipeline = build_pipeline()
    print("Treinando Gradient Boosting (2022 -> 2023)...")
    pipeline.fit(X_treino, y_treino)

    # Limiar via OOF no treino, exigindo recall mínimo de 80% (mesma lógica do notebook 03)
    n_splits = min(5, int(y_treino.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    proba_oof = cross_val_predict(pipeline, X_treino, y_treino, cv=cv, method="predict_proba")[:, 1]

    limiares = np.linspace(0.10, 0.90, 81)
    linhas = []
    for lim in limiares:
        pred = (proba_oof >= lim).astype(int)
        linhas.append({
            "limiar": lim,
            "recall": recall_score(y_treino, pred, zero_division=0),
            "precision": precision_score(y_treino, pred, zero_division=0),
        })
    tabela = pd.DataFrame(linhas)
    candidatos = tabela[tabela["recall"] >= 0.80].sort_values("precision", ascending=False)
    LIMIAR = float(candidatos.iloc[0]["limiar"]) if not candidatos.empty else 0.5
    print(f"Limiar selecionado: {LIMIAR:.2f}")

    # Avaliação no teste temporal
    proba_teste = pipeline.predict_proba(X_teste)[:, 1]
    pred_teste = (proba_teste >= LIMIAR).astype(int)

    metrics = {
        "acuracia": accuracy_score(y_teste, pred_teste),
        "precisao": precision_score(y_teste, pred_teste, zero_division=0),
        "recall": recall_score(y_teste, pred_teste, zero_division=0),
        "f1": f1_score(y_teste, pred_teste, zero_division=0),
        "roc_auc": roc_auc_score(y_teste, proba_teste),
        "pr_auc": average_precision_score(y_teste, proba_teste),
        "brier": brier_score_loss(y_teste, proba_teste),
    }
    print("Métricas no teste temporal (2023 -> 2024):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Retreino final com todo o histórico de transições
    X_hist, y_hist = base[features].copy(), base[target].copy()
    pipeline_final = build_pipeline()
    pipeline_final.fit(X_hist, y_hist)
    print("Modelo final treinado com todo o histórico (2022->2023 e 2023->2024).")

    helpers = compute_form_helpers(X_hist)

    return {
        "pipeline": pipeline_final,
        "modelo": "Gradient Boosting",
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "features": features,
        "limiar": LIMIAR,
        "metrics_teste_temporal": metrics,
        "target_definition": TARGET_DEFINITION,
        "populacao_elegivel": "Alunos com defasagem atual >= 0 e fase_num < 8.",
        "validacao": "Treino 2022->2023, teste temporal 2023->2024, modelo final com histórico completo.",
        **helpers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dados_pede.csv")
    parser.add_argument("--out", default="modelo_gb_risco_temporal.joblib")
    args = parser.parse_args()

    bundle = treinar_modelo(Path(args.data))
    joblib.dump(bundle, args.out)
    print(f"\nBundle salvo em: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
