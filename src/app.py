import streamlit as st
from credit_dashboard import show_credit_dashboard
from portfolio_recommender import show_portfolio_recommender

st.set_page_config(page_title="Financial AI Dashboard", layout="wide")

st.sidebar.title("🔍 Selecciona Módulo")
module = st.sidebar.radio("Ir a:", ["Credit Risk Dashboard", "Portfolio Recommender"])

if module == "Credit Risk Dashboard":
    show_credit_dashboard()
elif module == "Portfolio Recommender":
    show_portfolio_recommender()
    