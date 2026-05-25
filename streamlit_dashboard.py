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
    page_title="Previsão de Passagens Aéreas — TCC",
    page_icon="✈️",
    layout="wide",
)

# ────────────────────────────────────────────────────────────────────────────
# LSTM (deep learning)
# ────────────────────────────────────────────────────────────────────────────

class LSTMRegressor(nn.Module):
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


# ────────────────────────────────────────────────────────────────────────────
# PRÉ-PROCESSAMENTO
# ────────────────────────────────────────────────────────────────────────────

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

    features         = ["Airline", "Source", "Destination", "Additional_Info",
                        "journey_day", "journey_month", "duration_mins", "total_stops",
                        "dep_time_mins", "arrival_time_mins"]
    categorical_cols = ["Airline", "Source", "Destination", "Additional_Info"]
    numeric_cols     = ["journey_day", "journey_month", "duration_mins", "total_stops",
                        "dep_time_mins", "arrival_time_mins"]

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

@st.cache_resource
def train_one(model_name, _params_key, n_estimators, max_depth_val, learning_rate,
              subsample, lstm_epochs, lstm_hidden, _X_train, _y_train, _X_valid, _y_valid):
    if model_name == "Random Forest":
        m = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth_val if max_depth_val != 0 else None,
            random_state=42, n_jobs=-1,
        )
        m.fit(_X_train, _y_train)
        return m
    elif model_name == "XGBoost":
        m = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth_val,
            learning_rate=learning_rate, subsample=subsample,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        m.fit(_X_train, _y_train)
        return m
    else:
        return _fit_lstm(_X_train, _y_train, _X_valid, _y_valid,
                         epochs=lstm_epochs, hidden_size=lstm_hidden)

@st.cache_resource
def train_all_baseline(_X_train, _y_train, _X_valid, _y_valid):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(_X_train, _y_train)
    xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(_X_train, _y_train)
    lstm = _fit_lstm(_X_train, _y_train, _X_valid, _y_valid, epochs=60, hidden_size=64)
    return {"Random Forest": rf, "XGBoost": xgb, "Rede Neural (LSTM)": lstm}


# ════════════════════════════════════════════════════════════════════════════
# CABEÇALHO PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

st.title("✈️ Quanto custa uma passagem aérea na Índia?")
st.markdown(
    "Este painel foi desenvolvido como **Trabalho de Conclusão de Curso** e mostra como "
    "um computador pode **aprender** a prever o preço de passagens aéreas a partir de "
    "informações como companhia, horário, duração e número de escalas do voo."
)

with st.expander("📖 Como funciona esse projeto? (clique para entender)"):
    st.markdown("""
    **O que é Machine Learning?**
    É uma técnica onde ensinamos o computador a resolver um problema mostrando muitos exemplos.
    Aqui, mostramos milhares de voos com seus preços reais — e o computador aprende
    os padrões que fazem um voo ser mais caro ou mais barato.

    **O que o computador aprende?**
    Por exemplo: voos com mais escalas tendem a ser mais caros; certas companhias
    cobram mais; voos à noite têm preços diferentes dos diurnos. O modelo aprende
    tudo isso sozinho, olhando os dados.

    **Como sabemos se ele aprendeu bem?**
    Separamos uma parte dos dados que o computador **nunca viu** durante o treino.
    Depois pedimos que ele preveja os preços dessa parte e comparamos com os valores reais.
    Quanto menor o erro, melhor o modelo aprendeu.

    **Os três modelos testados:**
    - 🌳 **Random Forest:** como um júri com centenas de especialistas votando juntos — cada "árvore" dá um palpite e a maioria vence.
    - ⚡ **XGBoost:** aprende com os próprios erros, como um atleta que treina focando nas fraquezas. Campeão em competições de dados.
    - 🧠 **Rede Neural (LSTM):** inspirado no cérebro humano, aprende padrões complexos entre as informações do voo.
    """)

st.divider()

# ── BARRA LATERAL ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações")

    st.markdown("**Arquivos de dados**")
    train_path = st.text_input("Dados de treino", "Data_Train.xlsx")
    test_path  = st.text_input("Dados de teste",  "Test_set.xlsx")

    st.divider()

    st.markdown("**Quanto dos dados reservar para o teste?**")
    st.caption("Esses dados ficam escondidos do modelo durante o treino — servem só para medir o erro no final.")
    test_size = st.slider("Percentual de teste", 10, 40, 20, step=5,
                          help="20% = o modelo aprende com 80% dos voos e é testado nos 20% restantes.")

    st.divider()

    st.markdown("**Escolha um modelo para explorar**")
    model_labels = {
        "Random Forest":      "🌳 Random Forest",
        "XGBoost":            "⚡ XGBoost",
        "Rede Neural (LSTM)": "🧠 Rede Neural (LSTM)",
    }
    model_choice = st.selectbox(
        "Modelo",
        list(model_labels.keys()),
        format_func=lambda x: model_labels[x],
    )

    st.markdown("**Ajustes avançados do modelo**")
    st.caption("Você pode experimentar diferentes configurações e ver como o resultado muda.")

    n_estimators  = 100
    max_depth_val = 0
    learning_rate = 0.1
    subsample     = 0.9
    lstm_epochs   = 60
    lstm_hidden   = 64

    if model_choice == "Random Forest":
        n_estimators = st.slider(
            "Número de 'especialistas' (árvores)",
            50, 300, 100, step=50,
            help="Mais árvores = mais preciso, mas mais lento.",
        )
        rf_depth_sel  = st.select_slider(
            "Profundidade máxima de raciocínio",
            options=["Sem limite", 5, 10, 20, 30],
            help="Quanto mais profundo, mais detalhe cada árvore analisa.",
        )
        max_depth_val = 0 if rf_depth_sel == "Sem limite" else int(rf_depth_sel)

    elif model_choice == "XGBoost":
        n_estimators = st.slider(
            "Número de rodadas de aprendizado",
            50, 400, 100, step=50,
            help="Mais rodadas = aprende mais, mas pode 'decorar' demais.",
        )
        max_depth_val = st.select_slider(
            "Profundidade de análise por rodada",
            options=[3, 5, 7, 9],
            help="Valores menores = aprendizado mais geral.",
        )
        learning_rate = st.select_slider(
            "Velocidade de aprendizado",
            options=[0.01, 0.05, 0.1, 0.2],
            help="Velocidades menores são mais cuidadosas e geralmente melhores.",
        )
        subsample = st.select_slider(
            "Fração dos dados por rodada",
            options=[0.7, 0.8, 0.9, 1.0],
            help="Usar menos dados por rodada ajuda a não decorar os exemplos.",
        )

    else:  # LSTM
        lstm_epochs = st.slider(
            "Quantas vezes a rede estuda os dados",
            20, 120, 60, step=10,
            help="A rede para antes se perceber que parou de melhorar (early stopping).",
        )
        lstm_hidden = st.select_slider(
            "Tamanho da memória da rede",
            options=[32, 64, 128],
            help="Redes maiores capturam padrões mais complexos, mas precisam de mais dados.",
        )

    params_key = (model_choice, n_estimators, max_depth_val, learning_rate,
                  subsample, lstm_epochs, lstm_hidden)

# ── CARREGAR DADOS ────────────────────────────────────────────────────────────
try:
    raw_df = load_raw(train_path)
except Exception as e:
    st.error(f"❌ Não foi possível abrir o arquivo '{train_path}'. Verifique o nome e tente novamente.")
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
    "📊 Conhecendo os Dados",
    "🏆 Qual modelo é melhor?",
    "🔬 Experimentar um Modelo",
    "📌 O que influencia o preço?",
    "🎯 Prever Preços Novos",
])

# ════════════════════════════════════════════════════════════════════════════
# ABA 1 — CONHECENDO OS DADOS
# ════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.header("📊 Conhecendo os Dados")
    st.markdown(
        "Antes de ensinar qualquer coisa ao computador, precisamos entender os dados. "
        "Aqui exploramos os **10.683 voos domésticos na Índia** que usamos no projeto."
    )

    # ── Números rápidos ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✈️ Voos no dataset", f"{len(raw_df):,}")
    c2.metric("✅ Voos usados no treino", f"{len(df_proc):,}",
              delta=f"−{len(raw_df) - len(df_proc)} outliers removidos",
              delta_color="off")
    c3.metric("💰 Preço médio", f"₹ {df_proc['Price'].mean():,.0f}",
              help="Em rúpias indianas (INR). 1 INR ≈ R$ 0,06.")
    c4.metric("📍 Preço mediano", f"₹ {df_proc['Price'].median():,.0f}",
              help="Metade dos voos custa menos que esse valor.")

    st.info(
        "💡 **O que é um outlier?** São voos com preços absurdamente altos (acima de ₹35.000) "
        "que poderiam confundir o modelo — como tentar aprender o preço de um carro popular "
        "incluindo Ferraris na conta. Removemos o 1% mais caro para o modelo aprender melhor."
    )

    st.divider()

    # ── Distribuição de preços ───────────────────────────────────────────────
    cap = raw_df["Price"].quantile(0.99)
    df_eda = raw_df[raw_df["Price"] <= cap].copy()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Como os preços se distribuem?")
        st.caption("Cada barra mostra quantos voos têm aquele faixa de preço.")
        fig_hist = px.histogram(
            df_eda, x="Price", nbins=60,
            color_discrete_sequence=["#4C9BE8"],
            labels={"Price": "Preço (₹)", "count": "Nº de voos"},
        )
        fig_hist.update_layout(showlegend=False, margin=dict(t=10),
                                yaxis_title="Número de voos")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(
            "📌 A maioria dos voos custa entre ₹4.000 e ₹12.000. "
            "Poucos voos muito caros foram removidos da análise."
        )

    with col_b:
        st.subheader("Escalas encarecem o voo?")
        st.caption("Cada caixa mostra a faixa de preços para aquele número de paradas.")
        stop_order  = ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"]
        valid_stops = [s for s in stop_order if s in raw_df["Total_Stops"].dropna().unique()]
        stop_labels = {
            "non-stop": "Direto", "1 stop": "1 escala",
            "2 stops": "2 escalas", "3 stops": "3 escalas", "4 stops": "4 escalas",
        }
        df_eda_stops = df_eda.copy()
        df_eda_stops["Paradas"] = df_eda_stops["Total_Stops"].map(stop_labels).fillna(df_eda_stops["Total_Stops"])
        stop_labels_ordered = [stop_labels.get(s, s) for s in valid_stops]
        fig_stops = px.box(
            df_eda_stops, x="Paradas", y="Price",
            category_orders={"Paradas": stop_labels_ordered},
            color="Paradas",
            labels={"Price": "Preço (₹)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_stops.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig_stops, use_container_width=True)
        st.caption(
            "📌 Sim! Voos com mais escalas tendem a custar mais. "
            "A linha no meio de cada caixa é o preço do 'voo típico'."
        )

    # ── Preço por companhia ──────────────────────────────────────────────────
    st.subheader("Qual companhia é mais cara?")
    st.caption("As companhias estão ordenadas da mais cara para a mais barata (pelo preço mediano).")
    airline_order = df_eda.groupby("Airline")["Price"].median().sort_values(ascending=False).index.tolist()
    fig_airline = px.box(
        df_eda, x="Airline", y="Price",
        category_orders={"Airline": airline_order},
        color="Airline",
        labels={"Airline": "Companhia", "Price": "Preço (₹)"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_airline.update_layout(showlegend=False, xaxis_tickangle=-30, margin=dict(t=10))
    st.plotly_chart(fig_airline, use_container_width=True)
    st.caption(
        "📌 Jet Airways e Air India cobram mais (companhias premium). "
        "IndiGo, SpiceJet e GoAir são mais baratas (modelo low-cost)."
    )

    # ── Amostra dos dados ────────────────────────────────────────────────────
    with st.expander("🔍 Ver os dados brutos (primeiras 8 linhas)"):
        st.caption("Esses são os dados exatamente como vieram — o computador precisou transformar tudo isso em números para aprender.")
        st.dataframe(raw_df.head(8), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# ABA 2 — QUAL MODELO É MELHOR?
# ════════════════════════════════════════════════════════════════════════════
with tab_models:
    st.header("🏆 Qual modelo aprende melhor?")
    st.markdown(
        "Treinamos **três modelos diferentes** com os mesmos dados e comparamos "
        "quem erra menos ao prever preços que nunca viu antes."
    )

    with st.expander("🤔 Como medir se um modelo é bom?"):
        st.markdown("""
        Usamos duas medidas:

        **Erro médio (RMSE):** Em média, quantas rúpias o modelo erra por voo.
        → Se o erro médio é ₹1.500, significa que a previsão fica a ~₹1.500 do preço real.
        → **Quanto menor, melhor.**

        **Precisão (R²):** Que porcentagem da variação de preços o modelo consegue explicar.
        → 0,85 significa que o modelo captura 85% dos fatores que fazem um voo ser caro ou barato.
        → **Quanto mais próximo de 1,0 (100%), melhor.**
        """)

    with st.spinner("⏳ Treinando os três modelos... A rede neural pode levar ~30 segundos."):
        baseline_models = train_all_baseline(X_train, y_train, X_valid, y_valid)

    rows = []
    for name, mdl in baseline_models.items():
        preds = mdl.predict(X_valid)
        rmse  = np.sqrt(mean_squared_error(y_valid, preds))
        r2    = r2_score(y_valid, preds)
        rows.append({
            "Modelo": name,
            "Erro médio por voo": f"₹ {rmse:,.0f}",
            "Precisão (R²)": f"{r2:.1%}",
            "_rmse": rmse,
            "_r2": r2,
        })

    results_df = pd.DataFrame(rows).sort_values("_rmse").reset_index(drop=True)

    # ── Pódio ────────────────────────────────────────────────────────────────
    best_row = results_df.iloc[0]
    st.success(
        f"🥇 **Melhor modelo: {best_row['Modelo']}** — "
        f"erra em média **{best_row['Erro médio por voo']}** por voo "
        f"e tem precisão de **{best_row['Precisão (R²)']}**."
    )

    # ── Gráficos ─────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Erro médio por modelo")
        st.caption("Barra menor = modelo mais preciso ✅")
        fig_rmse = px.bar(
            results_df, x="Modelo", y="_rmse", color="Modelo",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"_rmse": "Erro médio (₹)", "Modelo": ""},
            text=results_df["Erro médio por voo"],
        )
        fig_rmse.update_traces(textposition="outside")
        fig_rmse.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig_rmse, use_container_width=True)

    with c2:
        st.subheader("Precisão de cada modelo")
        st.caption("Barra maior = modelo mais preciso ✅")
        fig_r2 = px.bar(
            results_df, x="Modelo", y="_r2", color="Modelo",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"_r2": "Precisão (R²)", "Modelo": ""},
            text=results_df["Precisão (R²)"],
        )
        fig_r2.update_traces(textposition="outside")
        fig_r2.update_layout(showlegend=False, yaxis_tickformat=".0%", margin=dict(t=10))
        st.plotly_chart(fig_r2, use_container_width=True)

    # ── Tabela resumo ─────────────────────────────────────────────────────────
    st.subheader("Resumo dos resultados")
    st.dataframe(
        results_df[["Modelo", "Erro médio por voo", "Precisão (R²)"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "⚠️ Esses resultados usam configurações padrão (sem ajuste fino). "
        "Na aba '🔬 Experimentar', você pode ajustar os parâmetros e ver se consegue melhorar."
    )


# ════════════════════════════════════════════════════════════════════════════
# ABA 3 — EXPERIMENTAR UM MODELO
# ════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.header(f"🔬 Experimentando: {model_labels[model_choice]}")
    st.markdown(
        "Use os controles na **barra lateral esquerda** para ajustar as configurações do modelo "
        "e veja aqui como o desempenho muda. É como afinar um instrumento musical!"
    )

    with st.spinner(f"⏳ Treinando {model_labels[model_choice]}..."):
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
    m1.metric(
        "💸 Erro médio por voo", f"₹ {rmse:,.0f}",
        help="Em média, o modelo erra esse valor ao prever o preço de um voo.",
    )
    m2.metric(
        "🎯 Precisão do modelo", f"{r2:.1%}",
        help="Quanto das variações de preço o modelo consegue explicar.",
    )
    m3.metric("📋 Voos testados", f"{len(y_valid):,}",
              help="Número de voos usados para medir o erro (nunca vistos pelo modelo).")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Preço previsto × Preço real")
        st.caption(
            "Cada ponto é um voo. Se o ponto estiver na linha vermelha, "
            "a previsão foi perfeita. Pontos longe da linha = erros maiores."
        )
        fig_scatter = px.scatter(
            x=y_valid, y=preds,
            labels={"x": "Preço real (₹)", "y": "Preço previsto (₹)"},
            opacity=0.4,
            color_discrete_sequence=["#4C9BE8"],
        )
        fig_scatter.add_shape(
            type="line",
            x0=float(y_valid.min()), y0=float(y_valid.min()),
            x1=float(y_valid.max()), y1=float(y_valid.max()),
            line=dict(color="red", dash="dash", width=2),
        )
        fig_scatter.update_layout(margin=dict(t=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.subheader("O modelo tem tendência a errar para algum lado?")
        st.caption(
            "Este gráfico mostra a diferença entre o preço real e o previsto. "
            "Centrado no zero = o modelo não favorece nem preços altos nem baixos."
        )
        residuals = np.array(y_valid) - preds
        fig_res = px.histogram(
            residuals, nbins=50,
            labels={"value": "Diferença (real − previsto em ₹)", "count": "Nº de voos"},
            color_discrete_sequence=["#4C9BE8"],
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
        fig_res.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig_res, use_container_width=True)

    st.info(
        "💡 **Como interpretar:** Um bom modelo tem os pontos perto da linha vermelha "
        "e a distribuição de erros centrada em zero. Se estiver torta para um lado, "
        "o modelo sistematicamente subestima ou superestima os preços."
    )


# ════════════════════════════════════════════════════════════════════════════
# ABA 4 — O QUE INFLUENCIA O PREÇO?
# ════════════════════════════════════════════════════════════════════════════
with tab_importance:
    st.header("📌 O que mais influencia o preço da passagem?")
    st.markdown(
        "Alguns modelos conseguem nos dizer **quais informações foram mais importantes** "
        "para chegar no preço previsto. Isso ajuda a entender a lógica por trás das previsões."
    )

    if model_choice == "Rede Neural (LSTM)":
        st.warning(
            "🧠 A **Rede Neural** aprende de forma distribuída — os fatores importantes ficam "
            "espalhados por milhares de conexões internas, tornando difícil identificar um único "
            "fator mais importante. Selecione **Random Forest** ou **XGBoost** na barra "
            "lateral para ver quais informações mais influenciaram o modelo."
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

        # Nomes amigáveis para as features principais
        friendly_names = {
            "duration_mins":      "⏱️ Duração do voo",
            "total_stops":        "🔄 Nº de escalas",
            "dep_time_mins":      "🌅 Horário de partida",
            "arrival_time_mins":  "🌆 Horário de chegada",
            "journey_day":        "📅 Dia da viagem",
            "journey_month":      "🗓️ Mês da viagem",
        }
        def friendly(name):
            for key, label in friendly_names.items():
                if name == key:
                    return label
            if name.startswith("Airline_"):
                return f"✈️ Companhia: {name.replace('Airline_','')}"
            if name.startswith("Source_"):
                return f"🛫 Origem: {name.replace('Source_','')}"
            if name.startswith("Destination_"):
                return f"🛬 Destino: {name.replace('Destination_','')}"
            if name.startswith("Additional_Info_"):
                return f"ℹ️ Info: {name.replace('Additional_Info_','')}"
            return name

        top_n = st.slider("Quantos fatores exibir?", 5, 20, 12)
        top_idx   = np.argsort(importances)[-top_n:]
        top_names = [friendly(feature_names[i]) for i in top_idx]
        top_vals  = importances[top_idx]

        fig_fi = go.Figure(go.Bar(
            x=top_vals,
            y=top_names,
            orientation="h",
            marker_color="#4C9BE8",
            text=[f"{v:.1%}" for v in top_vals],
            textposition="outside",
        ))
        fig_fi.update_layout(
            title=f"Fatores que mais influenciam o preço — {model_labels[model_choice]}",
            xaxis_title="Importância relativa",
            xaxis_tickformat=".0%",
            yaxis_title="",
            margin=dict(t=50, l=200),
            height=max(400, top_n * 35),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        st.info(
            "💡 **Como ler este gráfico:** A barra maior indica o fator que mais influenciou "
            "as previsões do modelo. Por exemplo, se 'Duração do voo' tem a maior barra, "
            "significa que voos mais longos/curtos causam a maior variação de preço."
        )

        with st.expander("📋 Ver todos os fatores em tabela"):
            fi_df = pd.DataFrame({
                "Fator":      [friendly(feature_names[i]) for i in np.argsort(importances)[::-1]],
                "Importância": [f"{v:.2%}" for v in sorted(importances, reverse=True)],
            })
            st.dataframe(fi_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# ABA 5 — PREVER PREÇOS NOVOS
# ════════════════════════════════════════════════════════════════════════════
with tab_test:
    st.header("🎯 Prever preços de voos desconhecidos")
    st.markdown(
        "Aqui usamos o **melhor modelo** para prever o preço de **2.671 voos** "
        "do conjunto de teste — voos que o modelo nunca viu e que não têm o preço real registrado."
    )

    st.info(
        "💡 **Por que existe um conjunto de teste separado?** "
        "Para simular o uso real: na prática, queremos prever preços de voos futuros "
        "que ainda não aconteceram. O conjunto de teste representa exatamente isso."
    )

    try:
        raw_test = load_raw(test_path)
        X_test   = preprocess_test(raw_test, preprocessor, features)

        with st.spinner("⏳ Gerando previsões..."):
            baseline_models_test = train_all_baseline(X_train, y_train, X_valid, y_valid)
            best_name  = results_df.iloc[0]["Modelo"] if "results_df" in dir() else "XGBoost"
            best_model = baseline_models_test.get(best_name, baseline_models_test["XGBoost"])
            y_pred     = best_model.predict(X_test)

        df_out = raw_test.copy()
        df_out["Preço Previsto (₹)"] = np.round(y_pred).astype(int)

        st.success(f"✅ Modelo usado: **{best_name}** — {len(df_out):,} previsões geradas.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("✈️ Voos previstos", f"{len(df_out):,}")
        m2.metric("💰 Preço médio previsto", f"₹ {df_out['Preço Previsto (₹)'].mean():,.0f}")
        m3.metric("📍 Preço mediano", f"₹ {df_out['Preço Previsto (₹)'].median():,.0f}")
        m4.metric("📊 Faixa de preços",
                  f"₹ {df_out['Preço Previsto (₹)'].min():,.0f} – {df_out['Preço Previsto (₹)'].max():,.0f}")

        st.subheader("Como os preços previstos se distribuem?")
        st.caption("A forma deste gráfico deve ser parecida com a distribuição dos dados de treino — isso indica que o modelo generalizou bem.")
        fig_pred = px.histogram(
            df_out, x="Preço Previsto (₹)", nbins=60,
            color_discrete_sequence=["#4C9BE8"],
            labels={"Preço Previsto (₹)": "Preço previsto (₹)", "count": "Nº de voos"},
        )
        fig_pred.update_layout(yaxis_title="Número de voos", margin=dict(t=10))
        st.plotly_chart(fig_pred, use_container_width=True)

        st.subheader("Amostra das previsões")
        st.caption("Os 20 primeiros voos do conjunto de teste com o preço previsto pelo modelo.")
        display_cols = {
            "Airline": "Companhia",
            "Source": "Origem",
            "Destination": "Destino",
            "Total_Stops": "Escalas",
            "Preço Previsto (₹)": "Preço Previsto (₹)",
        }
        df_display = df_out[list(display_cols.keys())].rename(columns=display_cols).head(20)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.divider()
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar todas as previsões (CSV)",
            data=csv,
            file_name="predictions_test_set.csv",
            mime="text/csv",
            help="Arquivo com os 2.671 voos e seus preços previstos.",
        )

    except FileNotFoundError:
        st.warning(
            f"⚠️ O arquivo `{test_path}` não foi encontrado. "
            "Verifique o nome do arquivo na barra lateral."
        )
    except Exception as e:
        st.error(f"❌ Erro ao gerar previsões: {e}")
