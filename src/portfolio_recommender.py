"""
The MIT License (MIT)

Copyright 2025 Natalia Stekolnikova, David Moreno

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Fechas globales
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

# Título y disclaimer
st.title("💼 Asesor de Cartera de Inversión")
st.markdown("*Esta simulación se basa en datos históricos y no garantiza rentabilidades futuras. Invierte con responsabilidad.*")

@st.cache_data
def choose_tickers():
    df = pd.read_csv('./data/processed/top_100_growth_stocks.csv')
    if df.empty:
        st.error("❌ No se encontraron datos de acciones. Por favor, verifica la fuente de datos.")
        return [], []
    tickers = df['Ticker'].tolist()
    names = df['name'].tolist()

    tickers_validos = []
    names_validos = []
    for ticker, name in zip(tickers, names):
        try:
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                tickers_validos.append(ticker)
                names_validos.append(name)
        except:
            pass
    return tickers_validos, names_validos

@st.cache_data
def cargar_datos():
    tickers, names = choose_tickers()
    symbols = dict(zip(names, tickers))
    data = pd.DataFrame()
    for company, symbol in symbols.items():
        df = yf.download(symbol, start=start_date, end=end_date)
        if not df.empty and 'Close' in df.columns:
            data[company] = df['Close']
    return data.dropna(), tickers, names

def optimizar_cartera(mean_returns, cov_matrix, perfil, volatilities, threshold=0.6):
    """
    Portfolio optimization with risk-adjusted constraints.
    
    Parameters:
    - mean_returns: Expected annualized returns for each asset
    - cov_matrix: Covariance matrix of asset returns
    - perfil: User's risk profile (e.g., "Riesgo mínimo", "Rentabilidad máxima")
    - volatilities: Annualized volatilities of assets (standard deviation of daily returns * sqrt(252))
    - threshold: Volatility level above which assets are considered highly unstable
    
    Business logic:
    - If an asset is highly volatile and the user has a conservative risk profile,
    that asset is either excluded or capped at a small weight.
    - Aggressive profiles are allowed to take on more risk, but weights are still limited for extremely volatile assets.
    """

    n = len(mean_returns)
    if n == 0:
        st.error("❌ No hay suficientes datos de retorno para calcular una cartera óptima.")
        return np.array([])

    # Objective functions for each risk profile.
    # These balance return and volatility in different proportions.
    def annualized_return(w):
        return np.sum(mean_returns * w) * 252

    def annualized_volatility(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * np.sqrt(252)

    objetivos = {
        "Riesgo mínimo": lambda w: annualized_volatility(w),  # prioritize stability
        "Riesgo bajo": lambda w: 0.75*annualized_volatility(w) - 0.25*annualized_return(w),
        "Riesgo medio": lambda w: 0.50*annualized_volatility(w) - 0.50*annualized_return(w),
        "Riesgo alto": lambda w: 0.25*annualized_volatility(w) - 0.75*annualized_return(w),
        "Rentabilidad máxima": lambda w: -annualized_return(w)  # maximize return
    }

    # Define allocation bounds dynamically based on volatility and risk profile
    bounds = []
    for i in range(n):
        vol = volatilities[i]
        
        # If asset is too volatile, limit or exclude it depending on profile
        if vol > threshold:
            if perfil in ["Riesgo mínimo", "Riesgo bajo"]:
                bounds.append((0, 0))  # exclude entirely for conservative users
            elif perfil == "Riesgo medio":
                bounds.append((0, 0.1))  # allow up to 10%
            else:
                bounds.append((0, 0.2))  # aggressive: allow up to 20%
        else:
            bounds.append((0, 1))  # normal allocation range

    # Initial equal weight guess
    w0 = np.ones(n) / n
    
    # Total weight constraint: sum of allocations must equal 1 (100%)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # Minimize the selected objective
    result = minimize(objetivos[perfil], w0, bounds=bounds, constraints=constraints)

    return result.x

def determinar_perfil(respuestas):
    total = sum(respuestas)
    if total <= 8:
        return "Riesgo mínimo"
    elif total <= 12:
        return "Riesgo bajo"
    elif total <= 17:
        return "Riesgo medio"
    elif total <= 21:
        return "Riesgo alto"
    else:
        return "Rentabilidad máxima"

# 🎛️ Barra lateral con preguntas
with st.sidebar:
    st.header("🧠 Configura tu perfil")
    opciones = {
        "Nivel de seguridad deseado": ["Máxima seguridad", "Alta seguridad", "Equilibrado", "Alta rentabilidad", "Máxima rentabilidad"],
        "Diversificación deseada": ["Muy conservadora", "Conservadora", "Equilibrada", "Agresiva", "Muy agresiva"],
        "Expectativa de rentabilidad": ["Muy baja", "Moderada-baja", "Media", "Alta", "Muy alta"],
        "Nivel máximo de pérdida aceptable": ["0%", "Hasta 5%", "Hasta 10%", "Hasta 20%", "Más de 20%"],
        "Horizonte temporal": ["< 1 año", "1-3 años", "3-5 años", "5-10 años", ">10 años"]
    }

    respuestas = []
    for pregunta, valores in opciones.items():
        eleccion = st.selectbox(pregunta, valores, key=pregunta)
        respuestas.append(valores.index(eleccion) + 1)

    ejecutar = st.button("📊 Generar cartera")

# Solo si se pulsa el botón
if ejecutar:
    st.subheader("📊 Resultados de la simulación")
    perfil = determinar_perfil(respuestas)
    st.markdown(f"**Perfil de riesgo detectado:** `{perfil}`")

    with st.spinner("🔍 Calculando cartera óptima..."):
        data, tickers, names = cargar_datos()
        returns = data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # Calcular volatilidades anuales
        # Volatilidad anualizada = desviación estándar de los retornos diarios * sqrt(252)
        volatilities = returns.std() * np.sqrt(252)
        pesos = optimizar_cartera(mean_returns, cov_matrix, perfil, volatilities)

        # Mostrar tabla de pesos
        cartera = {data.columns[i]: round(pesos[i]*100, 2) for i in range(len(data.columns)) if pesos[i] > 0.001}
        ordenada = dict(sorted(cartera.items(), key=lambda x: x[1], reverse=True))
        st.write("### 📌 Composición de la cartera:")
        st.dataframe(pd.DataFrame(ordenada.items(), columns=["Empresa", "Porcentaje"]))

        # Evolución
        inversion_inicial = 1000
        cartera_retornos = (data * pesos).sum(axis=1)
        cartera_valores = cartera_retornos / cartera_retornos.iloc[0] * inversion_inicial

        st.line_chart(cartera_valores, height=300)

        # Estadísticas
        rolling_max = cartera_valores.cummax()
        drawdowns = (cartera_valores - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        drawdown_start = drawdowns.idxmin()
        try:
            recovery = cartera_valores.loc[drawdown_start:]
            drawdown_end = recovery[recovery >= recovery.iloc[0]].index[0]
        except:
            drawdown_end = cartera_valores.index[-1]

        valor_final = cartera_valores.iloc[-1]
        ganancia_pct = (valor_final - inversion_inicial) / inversion_inicial * 100

        st.markdown(f"**📈 Rentabilidad final estimada:** `{valor_final:.2f} €`")
        st.markdown(f"**📉 Máximo drawdown:** `{max_drawdown*100:.2f}%` entre `{drawdown_start.date()}` y `{drawdown_end.date()}`")
        st.markdown(f"**🔎 Rentabilidad acumulada:** `{ganancia_pct:.2f}%`")