
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.title("Predicción de erosión por regresión lineal (Y)")

archivo = st.file_uploader("Sube tu archivo CSV con datos de Y", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo, encoding="latin1")
    df = df.dropna(subset=["Abscisa"])

    # Selección de columna (fecha) para evaluar
    opciones_fechas = df.columns[2:]
    columna_a_evaluar = st.selectbox("Selecciona una columna/fecha para evaluar:", opciones_fechas)

    umbral_minimo = st.number_input("Ingrese el valor mínimo permitido:", min_value=0.0, value=0.6, step=0.1)

    # Calcular margen y alerta
    df["Margen"] = (df[columna_a_evaluar] - umbral_minimo).round(3)
    df["Alerta"] = df["Margen"].apply(lambda x: "ALERTA" if x < 0 else "OK")

    df_resultado = df[["Abscisa", columna_a_evaluar, "Margen", "Alerta"]]

    def resaltar_alertas(row):
        return ['background-color: red; color: white' if val == "ALERTA" else '' for val in row]

    st.subheader("Resultados con alertas")
    st.dataframe(df_resultado.style.apply(resaltar_alertas, axis=1))

    # Gráfico
    st.subheader("Distribución de márgenes")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df["Margen"], kde=True, ax=ax)
    ax.axvline(0, color='red', linestyle='--', label='Umbral crítico')
    ax.set_title("Histograma de márgenes respecto al umbral")
    ax.set_xlabel("Margen (valor observado - umbral)")
    ax.legend()
    st.pyplot(fig)
