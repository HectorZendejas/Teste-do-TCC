import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

st.set_page_config(
    page_title="TCC — Previsão de Preços de Passagens",
    page_icon="✈️",
    layout="wide",
)

# ── Arquitetura LSTM ─────────────────────────────────────────────────────────

class LSTMRegressor(nn.Module):
    """Trata cada amostra tabular como sequência de 1 passo temporal."""
    def __init__(self, input_size, hidden_size=64, dropout=0.2):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_size, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc2(self.relu(self.fc1(out))).squeeze(-1)


class LSTMWrapper:
    """Wrapper sklearn-compatível para o LSTMRegressor PyTorch."""
    def __init__(self, model: LSTMRegressor):
        self.model = model

    def predict(self, X):
        X_t = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_t).numpy()


def _fit_lstm(X_tr, y_tr, X_vl, y_vl,
              epochs=60, batch_size=64, lr=1e-3,
              hidden_size=64, dropout=0.2, patience=8):
    """Treina o LSTM com early stopping e retorna um LSTMWrapper."""
    Xtr = torch.tensor(np.array(X_tr), dtype=torch.float32).unsqueeze(1)
    ytr = torch.tensor(np.array(y_tr), dtype=torch.float32)
    Xvl = torch.tensor(np.array(X_vl), dtype=torch.float32).unsqueeze(1)
    yvl = torch.tensor(np.array(y_vl), dtype=torch.float32)

    loader    = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    model     = LSTMRegressor(X_tr.shape[1], hidden_size=hidden_size, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xvl), yvl).item()
        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return LSTMWrapper(model)


# ── Helpers de pré-processamento ────────────────────────────────────────────

def _duration_to_minutes(s):
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return np.nan
    h = int(re.search(r"(\d+)\s*h", s).group(1)) if re.search(r"(\d+)\s*h", s) else 0
    m = int(re.search(r"(\d+)\s*m", s).group(1)) if re.search(r"(\d+)\s*m", s) else 0
    return h * 60 + m

def _stops_to_int(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower()
    if "non" in s:
        return 0
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan

def _time_col_to_minutes(series):
    mins = []
    for val in series.fillna("").astype(str):
        m = re.match(r"(\d{1,2}):(\d{2})", val.strip())
        mins.append(int(m.group(1)) * 60 + int(m.group(2)) if m else np.nan)
    return mins

# ── Carregamento e pré-processamento ────────────────────────────────────────

@st.cache_data
def load_raw(path):
    return pd.read_excel(path)

@st.cache_data
def preprocess(df_raw, remove_outliers=True):
    df = df_raw.copy()

    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True, errors="coerce")
    df["journey_day"]   = df["Date_of_Journey"].dt.day
    df["journey_month"] = df["Date_of_Journey"].dt.month
    df.drop(columns=["Date_of_Journey"], inplace=True)

    df["duration_mins"] = df["Duration"].apply(_duration_to_minutes)
    df.drop(columns=["Duration"], inplace=True)

    df["total_stops"] = df["Total_Stops"].apply(_stops_to_int)
    df.drop(columns=["Total_Stops"], inplace=True)

    df["dep_time_mins"]     = _time_col_to_minutes(df["Dep_Time"])
    df["arrival_time_mins"] = _time_col_to_minutes(df["Arrival_Time"])
    df.drop(columns=["Dep_Time", "Arrival_Time"], inplace=True)

    if remove_outliers and "Price" in df.columns:
        cap = df["Price"].quantile(0.99)
        df = df[df["Price"] <= cap].copy()

    features = [
        "Airline", "Source", "Destination", "Additional_Info",
        "journey_day", "journey_month", "duration_mins", "total_stops",
        "dep_time_mins", "arrival_time_mins",
    ]
    categorical_cols = ["Airline", "Source", "Destination", "Additional_Info"]
    numeric_cols     = [
        "journey_day", "journey_month", "duration_mins", "total_stops",
        "dep_time_mins", "arrival_time_mins",
    ]

    X = df[features]
    y = pd.to_numeric(df["Price"], errors="coerce") if "Price" in df.columns else None

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", enc),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    X_proc = preprocessor.fit_transform(X)

    try:
        cat_names = preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(categorical_cols).tolist()
    except AttributeError:
        cat_names = preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names(categorical_cols).tolist()
    feature_names = numeric_cols + cat_names

    return X_proc, y, preprocessor, features, feature_names, df

@st.cache_data
def preprocess_test(df_test_raw, _preprocessor, features):
    df = df_test_raw.copy()
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True, errors="coerce")
    df["journey_day"]   = df["Date_of_Journey"].dt.day
    df["journey_month"] = df["Date_of_Journey"].dt.month
    df.drop(columns=["Date_of_Journey"], inplace=True)
    df["duration_mins"]     = df["Duration"].apply(_duration_to_minutes)
    df.drop(columns=["Duration"], inplace=True)
    df["total_stops"]       = df["Total_Stops"].apply(_stops_to_int)
    df.drop(columns=["Total_Stops"], inplace=True)
    df["dep_time_mins"]     = _time_col_to_minutes(df["Dep_Time"])
    df["arrival_time_mins"] = _time_col_to_minutes(df["Arrival_Time"])
    df.drop(columns=["Dep_Time", "Arrival_Time"], inplace=True)
    return _preprocessor.transform(df[features])

# PyTorch models can't be pickled by cache_data — use cache_resource
@st.cache_resource
def train_one(model_name, _params_key, n_estimators, max_depth, learning_rate, subsample,
              lstm_epochs, lstm_hidden, _X_train, _y_train, _X_valid, _y_valid):
    """Treina um único modelo com os hiperparâmetros fornecidos."""
    if model_name == "Random Forest":
        m = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth != 0 else None,
            random_state=42, n_jobs=-1,
        )
        m.fit(_X_train, _y_train)
        return m
    elif model_name == "XGBoost":
        m = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        m.fit(_X_train, _y_train)
        return m
    else:  # LSTM
        return _fit_lstm(
            _X_train, _y_train, _X_valid, _y_valid,
            epochs=lstm_epochs, hidden_size=lstm_hidden,
        )

@st.cache_resource
def train_all_baseline(_X_train, _y_train, _X_valid, _y_valid):
    """Treina os três modelos com parâmetros padrão para comparação baseline."""
    results = {}

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(_X_train, _y_train)
    results["Random Forest"] = rf

    xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(_X_train, _y_train)
    results["XGBoost"] = xgb

    results["LSTM"] = _fit_lstm(_X_train, _y_train, _X_valid, _y_valid,
                                epochs=60, hidden_size=64)
    return results

# ════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════

st.title("✈️ Previsão de Preços de Passagens Aéreas")
st.caption("TCC — Comparação de Modelos de Machine Learning  |  Dataset: voos domésticos na Índia")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações")

    train_path = st.text_input("Arquivo de treino", "Data_Train.xlsx")
    test_path  = st.text_input("Arquivo de teste",  "Test_set.xlsx")

    st.divider()
    st.subheader("Divisão treino / validação")
    test_size = st.slider("Tamanho da validação (%)", 10, 40, 20, step=5)

    st.divider()
    st.subheader("Modelo para explorar")
    model_choice = st.selectbox("Algoritmo", ["Random Forest", "XGBoost", "LSTM"])

    st.markdown("**Hiperparâmetros**")
    # defaults so all vars are always defined
    n_estimators  = 100
    max_depth_val = 0       # 0 → None for RF
    learning_rate = 0.1
    subsample     = 0.9
    lstm_epochs   = 60
    lstm_hidden   = 64

    if model_choice == "Random Forest":
        n_estimators  = st.slider("n_estimators", 50, 300, 100, step=50)
        rf_depth_sel  = st.select_slider("max_depth", options=["None", 5, 10, 20, 30])
        max_depth_val = 0 if rf_depth_sel == "None" else int(rf_depth_sel)
    elif model_choice == "XGBoost":
        n_estimators  = st.slider("n_estimators", 50, 400, 100, step=50)
        max_depth_val = st.select_slider("max_depth",     options=[3, 5, 7, 9])
        learning_rate = st.select_slider("learning_rate", options=[0.01, 0.05, 0.1, 0.2])
        subsample     = st.select_slider("subsample",     options=[0.7, 0.8, 0.9, 1.0])
    else:  # LSTM
        lstm_epochs = st.slider("Épocas (máx)", 20, 120, 60, step=10)
        lstm_hidden = st.select_slider("Neurônios LSTM", options=[32, 64, 128])

    # Unique key for cache_resource (represents all hyperparams as a tuple)
    params_key = (model_choice, n_estimators, max_depth_val, learning_rate,
                  subsample, lstm_epochs, lstm_hidden)

# ── Carregar dados ────────────────────────────────────────────────────────────
try:
    raw_df = load_raw(train_path)
except Exception as e:
    st.error(f"Erro ao carregar {train_path}: {e}")
    st.stop()

X, y, preprocessor, features, feature_names, df_proc = preprocess(raw_df)
mask  = y.notna()
X, y  = X[mask], y[mask]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=test_size / 100, random_state=42
)

# ════════════════════════════════════════════════════════════════════════════
# ABAS
# ════════════════════════════════════════════════════════════════════════════
tab_eda, tab_models, tab_explore, tab_importance, tab_test = st.tabs([
    "📊 EDA",
    "🏆 Comparação de Modelos",
    "🔬 Explorar Modelo",
    "📌 Importância de Features",
    "🎯 Previsões no Test Set",
])

# ── ABA 1: EDA ───────────────────────────────────────────────────────────────
with tab_eda:
    st.header("Análise Exploratória de Dados")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros (treino)", f"{len(raw_df):,}")
    c2.metric("Após remoção de outliers", f"{len(df_proc):,}")
    c3.metric("Preço médio (INR)", f"{df_proc['Price'].mean():,.0f}")
    c4.metric("Preço mediano (INR)", f"{df_proc['Price'].median():,.0f}")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribuição de Price")
        cap = raw_df["Price"].quantile(0.99)
        fig_hist = px.histogram(
            raw_df[raw_df["Price"] <= cap], x="Price", nbins=60,
            color_discrete_sequence=["steelblue"],
            labels={"Price": "Preço (INR)"},
        )
        fig_hist.update_layout(showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(f"Outliers removidos: {(raw_df['Price'] > cap).sum()} registros com Price > {cap:,.0f} INR (top 1%)")

    with col_b:
        st.subheader("Preço por Número de Paradas")
        stop_order  = ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"]
        valid_stops = [s for s in stop_order if s in raw_df["Total_Stops"].dropna().unique()]
        df_eda = raw_df[raw_df["Price"] <= cap].copy()
        fig_stops = px.box(
            df_eda, x="Total_Stops", y="Price",
            category_orders={"Total_Stops": valid_stops},
            color="Total_Stops",
            labels={"Total_Stops": "Paradas", "Price": "Preço (INR)"},
        )
        fig_stops.update_layout(showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig_stops, use_container_width=True)

    st.subheader("Preço por Companhia Aérea")
    airline_order = df_eda.groupby("Airline")["Price"].median().sort_values(ascending=False).index.tolist()
    fig_airline = px.box(
        df_eda, x="Airline", y="Price", category_orders={"Airline": airline_order},
        color="Airline",
        labels={"Airline": "Companhia", "Price": "Preço (INR)"},
    )
    fig_airline.update_layout(showlegend=False, xaxis_tickangle=-35, margin=dict(t=20))
    st.plotly_chart(fig_airline, use_container_width=True)

    st.subheader("Amostra dos dados")
    st.dataframe(raw_df.head(8), use_container_width=True)

# ── ABA 2: Comparação de Modelos ─────────────────────────────────────────────
with tab_models:
    st.header("Comparação de Modelos — Baseline")
    st.caption("Random Forest e XGBoost com n_estimators=100; LSTM com 60 épocas e early stopping. Avaliados no conjunto de validação.")

    with st.spinner("Treinando os três modelos (LSTM pode levar ~30 s)..."):
        baseline_models = train_all_baseline(X_train, y_train, X_valid, y_valid)

    rows = []
    for name, mdl in baseline_models.items():
        preds = mdl.predict(X_valid)
        rmse  = np.sqrt(mean_squared_error(y_valid, preds))
        r2    = r2_score(y_valid, preds)
        rows.append({"Modelo": name, "RMSE (INR)": round(rmse, 2), "R²": round(r2, 4)})

    results_df = pd.DataFrame(rows).sort_values("RMSE (INR)").reset_index(drop=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_rmse = px.bar(
            results_df, x="Modelo", y="RMSE (INR)", color="Modelo",
            title="RMSE por modelo (menor = melhor)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_rmse.update_layout(showlegend=False)
        st.plotly_chart(fig_rmse, use_container_width=True)
    with c2:
        fig_r2 = px.bar(
            results_df, x="Modelo", y="R²", color="Modelo",
            title="R² por modelo (maior = melhor)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_r2.update_layout(showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)

    st.subheader("Tabela de resultados")
    st.dataframe(results_df.style.highlight_min("RMSE (INR)", color="#d4edda")
                                  .highlight_max("R²",         color="#d4edda"),
                 use_container_width=True)

    best = results_df.iloc[0]["Modelo"]
    st.success(f"**Melhor modelo baseline:** {best} — RMSE: {results_df.iloc[0]['RMSE (INR)']:,.2f} INR  |  R²: {results_df.iloc[0]['R²']:.4f}")

# ── ABA 3: Explorar Modelo ───────────────────────────────────────────────────
with tab_explore:
    st.header(f"Explorar: {model_choice}")
    st.caption("Ajuste os hiperparâmetros na barra lateral e veja o impacto nas métricas.")

    with st.spinner(f"Treinando {model_choice}..."):
        model = train_one(
            model_choice, str(params_key),
            n_estimators, max_depth_val, learning_rate, subsample,
            lstm_epochs, lstm_hidden,
            X_train, y_train, X_valid, y_valid,
        )

    preds = model.predict(X_valid)
    rmse  = np.sqrt(mean_squared_error(y_valid, preds))
    r2    = r2_score(y_valid, preds)

    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", f"{rmse:,.2f} INR")
    m2.metric("R²",   f"{r2:.4f}")
    m3.metric("Registros validação", f"{len(y_valid):,}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Real vs. Previsto")
        fig_scatter = px.scatter(
            x=y_valid, y=preds,
            labels={"x": "Preço real (INR)", "y": "Preço previsto (INR)"},
            opacity=0.5, color_discrete_sequence=["steelblue"],
        )
        fig_scatter.add_shape(
            type="line",
            x0=float(y_valid.min()), y0=float(y_valid.min()),
            x1=float(y_valid.max()), y1=float(y_valid.max()),
            line=dict(color="red", dash="dash"),
        )
        fig_scatter.update_layout(margin=dict(t=20))
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Pontos na linha diagonal = previsão perfeita. Dispersão indica o erro do modelo.")

    with col2:
        st.subheader("Distribuição dos Resíduos")
        residuals = np.array(y_valid) - preds
        fig_res = px.histogram(
            residuals, nbins=50,
            labels={"value": "Resíduo (real − previsto)", "count": "Frequência"},
            color_discrete_sequence=["steelblue"],
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="red")
        fig_res.update_layout(showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig_res, use_container_width=True)
        st.caption("Distribuição centrada em zero indica ausência de viés sistemático.")

# ── ABA 4: Importância de Features ───────────────────────────────────────────
with tab_importance:
    st.header("Importância de Features")
    st.caption("Calculada sobre o modelo selecionado na barra lateral (apenas RF e XGBoost fornecem importâncias diretas).")

    if model_choice not in ("Random Forest", "XGBoost"):
        st.info(
            "O modelo **LSTM** não expõe importâncias diretas de features — a rede aprende "
            "representações internas distribuídas entre todos os pesos. Selecione "
            "**Random Forest** ou **XGBoost** na barra lateral para visualizar a importância."
        )
    else:
        with st.spinner("Calculando importâncias..."):
            model_fi = train_one(
                model_choice, str(params_key),
                n_estimators, max_depth_val, learning_rate, subsample,
                lstm_epochs, lstm_hidden,
                X_train, y_train, X_valid, y_valid,
            )
            importances = model_fi.feature_importances_

        top_n = st.slider("Número de features exibidas", 10, 30, 15)
        top_idx   = np.argsort(importances)[-top_n:]
        top_names = [feature_names[i] for i in top_idx]
        top_vals  = importances[top_idx]

        fig_fi = go.Figure(go.Bar(
            x=top_vals, y=top_names, orientation="h",
            marker_color="steelblue",
        ))
        fig_fi.update_layout(
            title=f"Top {top_n} features — {model_choice}",
            xaxis_title="Importância relativa",
            yaxis_title="",
            margin=dict(t=40),
            height=max(400, top_n * 28),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption("Barras maiores = feature com maior influência nas previsões do modelo.")

        with st.expander("Ver tabela completa"):
            fi_df = pd.DataFrame({
                "Feature":    [feature_names[i] for i in np.argsort(importances)[::-1]],
                "Importância": sorted(importances, reverse=True),
            })
            st.dataframe(fi_df, use_container_width=True)

# ── ABA 5: Previsões no Test Set ─────────────────────────────────────────────
with tab_test:
    st.header("Previsões no Conjunto de Teste")
    st.caption(f"Usando o melhor modelo baseline treinado na aba anterior. Arquivo: `{test_path}`")

    try:
        raw_test = load_raw(test_path)
        X_test   = preprocess_test(raw_test, preprocessor, features)

        with st.spinner("Gerando previsões..."):
            baseline_models_test = train_all_baseline(X_train, y_train, X_valid, y_valid)
            best_name  = results_df.iloc[0]["Modelo"] if "results_df" in dir() else "XGBoost"
            best_model = baseline_models_test.get(best_name, baseline_models_test["XGBoost"])
            y_pred     = best_model.predict(X_test)

        df_out = raw_test.copy()
        df_out["Price_Predicted"] = np.round(y_pred).astype(int)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total de previsões", f"{len(df_out):,}")
        m2.metric("Preço médio previsto", f"{df_out['Price_Predicted'].mean():,.0f} INR")
        m3.metric("Preço mediano previsto", f"{df_out['Price_Predicted'].median():,.0f} INR")

        fig_pred = px.histogram(
            df_out, x="Price_Predicted", nbins=60,
            labels={"Price_Predicted": "Preço Previsto (INR)"},
            color_discrete_sequence=["steelblue"],
            title=f"Distribuição dos preços previstos — {best_name}",
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        st.subheader("Amostra das previsões")
        show_cols = ["Airline", "Source", "Destination", "Total_Stops", "Price_Predicted"]
        st.dataframe(df_out[show_cols].head(20), use_container_width=True)

        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar predictions_test_set.csv",
            data=csv,
            file_name="predictions_test_set.csv",
            mime="text/csv",
        )

    except FileNotFoundError:
        st.warning(f"Arquivo `{test_path}` não encontrado. Verifique o caminho na barra lateral.")
    except Exception as e:
        st.error(f"Erro ao gerar previsões: {e}")
