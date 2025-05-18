
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

st.set_page_config(page_title="Monitoreo y Predicción de Erosión", layout="wide")

st.title("🌊 Monitoreo y Predicción de Erosión Costera")
tabs = st.tabs(["🏠 Inicio", "🔵 Variable Y", "🟢 Variable X", "🟡 Variable A", "📊 Análisis combinado", "📈 Indicadores globales"])

# Función general para procesar cada archivo
def analizar_variable(nombre_variable):
    archivo = st.file_uploader(f"📁 Subir archivo de {nombre_variable}", type=["csv"], key=nombre_variable)
    if archivo:
        df = pd.read_csv(archivo, encoding="latin1")
        df = df.dropna(subset=["Abscisa"])
        fecha_cols = df.columns[2:]
        st.subheader("📌 Datos originales")
        st.dataframe(df)

        umbral = st.number_input(f"Ingrese el valor umbral para {nombre_variable}:", value=0.3 if nombre_variable == "Y" else 1.0)

        # --- Comparación con primera fecha ---
        fecha_base = fecha_cols[0]
        st.subheader("📉 Diferencia con la primera fecha registrada")
        df_dif = df[["Abscisa"]].copy()
        for col in fecha_cols:
            df_dif[col] = (df[col] - df[fecha_base]).round(3)
        st.dataframe(df_dif)

        # --- Comparación con la fecha anterior ---
        st.subheader("🔁 Diferencia con fecha anterior")
        df_diff_ant = df[["Abscisa"]].copy()
        for i in range(1, len(fecha_cols)):
            col = fecha_cols[i]
            col_ant = fecha_cols[i - 1]
            df_diff_ant[col] = (df[col] - df[col_ant]).round(3)
        st.dataframe(df_diff_ant)

        # --- Resta con umbral ---
        st.subheader("🚨 Comparación con umbral")
        df_umbral = df[["Abscisa"]].copy()
        for col in fecha_cols:
            df_umbral[col] = (df[col] - umbral).round(3)
        st.dataframe(df_umbral)

        # --- Gráfica de comparación con umbral ---
        st.subheader("📊 Heatmap de valores vs. umbral")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(df_umbral[fecha_cols].astype(float), cmap="coolwarm", annot=False, ax=ax)
        ax.set_title(f"Heatmap de diferencia con umbral ({nombre_variable})")
        st.pyplot(fig)

        # --- Predicción regresión polinómica ---
        st.subheader("📈 Predicción con regresión polinómica por abscisa")
        fechas = [datetime.strptime(f, "%m/%d/%Y") for f in fecha_cols]
        dias = np.array([(f - fechas[0]).days for f in fechas]).reshape(-1, 1)
        df[fecha_cols] = df[fecha_cols].apply(pd.to_numeric, errors='coerce')

        resultados = []
        for _, row in df.iterrows():
            abscisa = row["Abscisa"]
            y_vals = row[fecha_cols].astype(float).values.reshape(-1, 1)
            if np.isnan(y_vals).any():
                continue
            mejor_r2 = -np.inf
            for grado in range(1, 5):
                modelo = make_pipeline(PolynomialFeatures(degree=grado), LinearRegression())
                modelo.fit(dias, y_vals)
                y_pred = modelo.predict(dias)
                r2 = r2_score(y_vals, y_pred)
                if r2 > mejor_r2:
                    mejor_r2 = r2
                    mejor_grado = grado
                    mejor_modelo = modelo
                    mejor_pred = y_pred
            actual = y_vals[-1][0]
            estado = "ALERTA" if actual < umbral else "OK"
            try:
                dias_pred = np.arange(dias[-1][0], dias[-1][0] + 365, 30).reshape(-1, 1)
                pred_futuro = mejor_modelo.predict(dias_pred)
                cruce = next((dias_pred[i][0] for i, val in enumerate(pred_futuro) if val[0] <= umbral), None)
                if cruce:
                    fecha_cruce = fechas[0] + timedelta(days=int(cruce))
                    fecha_cruce_str = fecha_cruce.strftime("%Y-%m-%d")
                else:
                    fecha_cruce_str = "No aplica"
            except:
                fecha_cruce_str = "Error"
            resultados.append({
                "Abscisa": abscisa,
                "Actual": round(actual, 3),
                "R²": round(mejor_r2, 4),
                "Grado": mejor_grado,
                "Estado": estado,
                "Cruce estimado": fecha_cruce_str
            })

        df_pred = pd.DataFrame(resultados)
        st.dataframe(df_pred)

# ====================
# Pestañas activas
# ====================

with tabs[0]:
    st.header("🏠 Inicio")
    st.markdown("""
    Bienvenido al sistema de monitoreo y predicción de erosión costera. Selecciona una pestaña para comenzar el análisis.
    """)

with tabs[1]:
    st.header("🔵 Variable Y")
    analizar_variable("Y")

with tabs[2]:
    st.header("🟢 Variable X")
    analizar_variable("X")

with tabs[3]:
    st.header("🟡 Variable A")
    analizar_variable("A")

with tabs[4]:
    st.header("📊 Análisis Combinado")
    st.markdown("🚧 En construcción: Aquí se integrarán los resultados de Y, X y A.")

with tabs[5]:
    st.header("📈 Indicadores Globales")
    st.markdown("🚦 Aquí se mostrarán los indicadores generales de estabilidad.")
