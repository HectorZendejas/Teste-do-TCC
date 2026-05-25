# Teste-do-TCC
# Análise Comparativa de Modelos de Machine Learning para Previsão de Preços de Passagens Aéreas

**Fundação Edson Queiroz — Universidade de Fortaleza (UNIFOR)**  
Centro de Ciências Tecnológicas — Curso de Ciências da Computação  
Trabalho de Conclusão de Curso — 2026

**Autores:**
- Brenno Damiany Castro Vidal
- Caio Victor Ferreira da Silva
- Diego Henrique Santos Queiroz
- Hector Zendejas Rebouças

---

## Sobre o Projeto

Este trabalho realiza uma **análise comparativa entre três algoritmos de Machine Learning** aplicados à previsão de preços de passagens aéreas domésticas na Índia. O objetivo é identificar qual modelo apresenta melhor desempenho considerando precisão, capacidade de generalização e eficiência computacional.

A questão central de pesquisa é: **qual modelo de Machine Learning apresenta os melhores resultados na previsão de preços de passagens aéreas?**

Como produto prático, foi desenvolvido um **MVP (Minimum Viable Product)** em Streamlit que demonstra o funcionamento do modelo preditivo de forma interativa.

---

## Modelos Comparados

| Modelo | Referência | Descrição |
|--------|-----------|-----------|
| **Random Forest** | Breiman (2001) | Ensemble de árvores de decisão treinadas em subconjuntos aleatórios. Robusto a outliers e eficiente em dados tabulares. |
| **XGBoost** | Chen & Guestrin (2016) | Gradient Boosting otimizado: cada árvore corrige os erros da anterior. Alto desempenho em competições de ciência de dados. |
| **LSTM** | Hochreiter & Schmidhuber (1997) | Rede neural recorrente adaptada para dados tabulares (cada amostra tratada como sequência de 1 passo), capturando interações não-lineares entre features via mecanismos de porta. Implementado em PyTorch. |

---

## Dataset

| Atributo | Detalhe |
|----------|---------|
| **Fonte** | Voos domésticos na Índia |
| **Treino** | `Data_Train.xlsx` — 10.683 registros com coluna `Price` |
| **Teste** | `Test_set.xlsx` — 2.671 registros sem `Price` |
| **Variável alvo** | `Price` (em rúpias indianas — INR) |
| **Features** | Airline, Date_of_Journey, Source, Destination, Dep_Time, Arrival_Time, Duration, Total_Stops, Additional_Info |

---

## Estrutura do Projeto

```
TCCteste/
├── Data_Train.xlsx                       # Dataset de treino (com preços reais)
├── Test_set.xlsx                         # Dataset de teste (sem preços)
├── TCC_Data_Treatment_and_Modeling.ipynb # Notebook principal — pipeline completo
├── streamlit_dashboard.py                # MVP interativo (dashboard Streamlit)
├── predictions_test_set.csv              # Previsões geradas para o conjunto de teste
└── README.md
```

---

## Pipeline do Notebook

| # | Etapa | Descrição |
|---|-------|-----------|
| 1 | Importação | Bibliotecas: pandas, numpy, scikit-learn, xgboost, PyTorch |
| 2 | Carregamento | Leitura dos arquivos Excel |
| 3 | Inspeção + EDA | Distribuição de preços, relação com escalas e companhias aéreas |
| 4 | Funções | Transformações de variáveis (data, duração, horários, paradas) |
| 5 | Pré-processamento | Aplicação das transformações + remoção de outliers (top 1% de `Price`) |
| 6 | Feature Engineering | OneHotEncoding para categóricas + StandardScaler para numéricas |
| 7 | Divisão | Treino / Validação — 80% / 20% (`random_state=42`) |
| 8 | Treinamento | Random Forest, XGBoost e LSTM (baseline) |
| 9 | Análise | Importância de features, tuning XGBoost (RandomizedSearchCV, 5-fold CV) |
| 10 | Conclusões | Comparação final e previsões no conjunto de teste |

---

## Pré-processamento Aplicado

- **Data de viagem** → extração de `journey_day` e `journey_month`
- **Duração** (`"2h 50m"`) → `duration_mins` (minutos inteiros)
- **Escalas** (`"1 stop"`, `"non-stop"`) → `total_stops` (inteiro)
- **Horários** (`"22:20"`) → `dep_time_mins` / `arrival_time_mins` (minutos desde meia-noite)
- **Outliers** → remoção do top 1% de `Price` (acima de ~₹35.000)
- **Colunas removidas** → `Route` (128 valores únicos → explosão dimensional) e `journey_year` (valor único: 2019)

---

## Métricas de Avaliação

| Métrica | Definição | Interpretação |
|---------|-----------|--------------|
| **RMSE** | Raiz do Erro Quadrático Médio | Erro médio em INR por previsão. Menor = melhor. |
| **R²** | Coeficiente de Determinação | Proporção da variância de `Price` explicada pelo modelo. Próximo de 1,0 = melhor. |

---

## Requisitos

- Python 3.9+ (testado com 3.14.3)
- Instalar dependências:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch streamlit plotly openpyxl
```

---

## Como Executar

### Notebook (análise completa)

Abra no Jupyter Lab, Jupyter Notebook ou VS Code:

```bash
jupyter notebook TCC_Data_Treatment_and_Modeling.ipynb
```

Execute as células em ordem. O notebook é autocontido: cada seção funciona independentemente desde que as anteriores tenham sido executadas.

### Dashboard Streamlit (MVP interativo)

```bash
streamlit run streamlit_dashboard.py
```

Acesse em `http://localhost:8501`. O dashboard inclui:

| Aba | Conteúdo |
|-----|----------|
| 📊 Conhecendo os Dados | EDA: distribuição de preços, análise por escalas e companhia |
| 🏆 Qual modelo é melhor? | Comparação baseline dos três modelos (RMSE e R²) |
| 🔬 Experimentar um Modelo | Ajuste de hiperparâmetros em tempo real + gráficos Real vs. Previsto |
| 📌 O que influencia o preço? | Importância de features (Random Forest e XGBoost) |
| 🎯 Prever Preços Novos | Previsões para o conjunto de teste + download do CSV |

> **Nota:** o treinamento do LSTM pode levar ~30 segundos na primeira execução. Os modelos são armazenados em cache para reruns subsequentes.

---

## Referências

- BREIMAN, L. Random Forests. *Machine Learning*, v. 45, n. 1, p. 5–32, 2001.
- CHEN, T.; GUESTRIN, C. XGBoost: A Scalable Tree Boosting System. *ACM SIGKDD*, 2016.
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. MIT Press, 2016.
- HOCHREITER, S.; SCHMIDHUBER, J. Long Short-Term Memory. *Neural Computation*, v. 9, n. 8, p. 1735–1780, 1997.
- MITCHELL, T. *Machine Learning*. McGraw-Hill, 1997.
- PASZKE, A. et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*, 2019.
- PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. *JMLR*, v. 12, p. 2825–2830, 2011.
- GROVER, S. et al. Airline Fare Prediction Using Machine Learning. 2019.
