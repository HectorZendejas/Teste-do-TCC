import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

st.set_page_config(page_title='Dashboard ML: Preço de Passagem', layout='wide')

@st.cache_data
def load_data(path='Data_Train.xlsx'):
    df = pd.read_excel(path)
    return df

@st.cache_data
def preprocess(df):
    df = df.copy()
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True, errors='coerce')
    df['journey_day'] = df['Date_of_Journey'].dt.day
    df['journey_month'] = df['Date_of_Journey'].dt.month
    df['journey_year'] = df['Date_of_Journey'].dt.year
    df.drop(columns=['Date_of_Journey'], inplace=True)

    def duration_to_minutes(value):
        value = str(value).replace(' ', '')
        parts = value.split('h')
        hours = int(parts[0]) if parts[0] != '' else 0
        mins = int(parts[1].replace('m', '')) if len(parts) > 1 and parts[1] != '' else 0
        return hours * 60 + mins

    df['duration_mins'] = df['Duration'].apply(duration_to_minutes)
    df.drop(columns=['Duration'], inplace=True)

    stops_map = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    df['total_stops'] = df['Total_Stops'].map(stops_map)
    df.drop(columns=['Total_Stops'], inplace=True)

    features = [
        'Airline', 'Source', 'Destination', 'Route', 'Dep_Time', 'Arrival_Time', 'Additional_Info',
        'journey_day', 'journey_month', 'journey_year', 'duration_mins', 'total_stops'
    ]
    X = df[features]
    y = df['Price']

    numeric_cols = ['journey_day', 'journey_month', 'journey_year', 'duration_mins', 'total_stops']
    categorical_cols = ['Airline', 'Source', 'Destination', 'Route', 'Dep_Time', 'Arrival_Time', 'Additional_Info']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ])

    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed, y, preprocessor, X

@st.cache_data
def train_model(model_name, params, X_train, y_train):
    if model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=42, n_jobs=-1)
    elif model_name == 'XGBoost':
        model = XGBRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], learning_rate=params['learning_rate'], subsample=params['subsample'], random_state=42, n_jobs=-1, verbosity=0)
    else:
        model = KNeighborsRegressor(n_neighbors=params['n_neighbors'], weights=params['weights'], n_jobs=-1)

    model.fit(X_train, y_train)
    return model

st.title('Dashboard Comparativo de Modelos de Machine Learning')

st.markdown(
    'Este dashboard permite treinar e comparar `Random Forest`, `XGBoost` e `KNN` para prever o preço de passagens aéreas usando `Data_Train.xlsx`.'
)

raw_df = load_data()
st.sidebar.header('Configurações do Dashboard')
model_choice = st.sidebar.selectbox('Escolha o modelo', ['Random Forest', 'XGBoost', 'KNN'])

st.sidebar.markdown('### Parâmetros do modelo')
if model_choice == 'Random Forest':
    rf_n = st.sidebar.slider('n_estimators', 50, 300, 100, step=50)
    rf_depth = st.sidebar.select_slider('max_depth', options=['None', 5, 10, 20, 30])
    rf_depth = None if rf_depth == 'None' else int(rf_depth)
    params = {'n_estimators': rf_n, 'max_depth': rf_depth}
elif model_choice == 'XGBoost':
    xgb_n = st.sidebar.slider('n_estimators', 50, 300, 100, step=50)
    xgb_depth = st.sidebar.select_slider('max_depth', options=[3, 5, 7, 9])
    xgb_lr = st.sidebar.select_slider('learning_rate', options=[0.01, 0.05, 0.1, 0.2])
    xgb_sub = st.sidebar.select_slider('subsample', options=[0.6, 0.8, 1.0])
    params = {'n_estimators': xgb_n, 'max_depth': xgb_depth, 'learning_rate': xgb_lr, 'subsample': xgb_sub}
else:
    knn_k = st.sidebar.slider('n_neighbors', 1, 15, 5)
    knn_w = st.sidebar.selectbox('weights', ['uniform', 'distance'])
    params = {'n_neighbors': knn_k, 'weights': knn_w}

st.sidebar.header('Divisão de dados')
test_size = st.sidebar.slider('Tamanho do teste (%)', 10, 40, 20, step=5)

X, y, preprocessor, X_raw = None, None, None, None
X, y, preprocessor, X_raw = preprocess(raw_df)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size / 100, random_state=42)

st.subheader('Dados carregados')
col1, col2, col3 = st.columns(3)
col1.metric('Linhas', raw_df.shape[0], '')
col2.metric('Colunas', raw_df.shape[1], '')
col3.metric('Modelos', 3, '')

st.dataframe(raw_df.head(5))

with st.expander('Visão geral das colunas'):
    st.write(raw_df.describe(include='all'))

model = train_model(model_choice, params, X_train, y_train)
preds = model.predict(X_valid)
rmse = mean_squared_error(y_valid, preds, squared=False)
r2 = r2_score(y_valid, preds)

st.subheader('Resultados do modelo')
st.metric('RMSE', f'{rmse:.2f}')
st.metric('R2', f'{r2:.4f}')

st.subheader('Gráficos de avaliação')
fig1 = px.scatter(x=y_valid, y=preds, labels={'x': 'Preço real', 'y': 'Preço previsto'}, title='Real vs Previsto')
fig1.add_shape(type='line', x0=y_valid.min(), y0=y_valid.min(), x1=y_valid.max(), y1=y_valid.max(), line=dict(color='red', dash='dash'))
fig2 = px.histogram((y_valid - preds), nbins=40, labels={'value': 'Resíduo'}, title='Distribuição dos resíduos')

st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

if model_choice in ['Random Forest', 'XGBoost']:
    st.subheader('Importância de variáveis')
    try:
        importance = model.feature_importances_
        if importance is not None:
            feature_names = preprocessor.transformers_[0][2] + list(model.get_booster().feature_names) if model_choice == 'XGBoost' else preprocessor.transformers_[0][2] + ['encoded_feature']
    except Exception:
        importance = None
    if importance is not None:
        st.write('A importância de variáveis deve ser extraída por ponto de base do pipeline em um notebook mais avançado.')
    else:
        st.info('A importância de variáveis não está disponível diretamente no pipeline neste app simplificado.')

st.markdown('---')
st.markdown('Use a barra lateral para ajustar hiperparâmetros e observar como o desempenho muda.')
