import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("🌊 Monitoreo Predictivo del Gasoducto Ciénaga - Tasajera")

# 📥 Cargar archivos
st.sidebar.header("📂 Subir archivos de variables")
archivos = {
    "Y": st.sidebar.file_uploader("Subir CSV de variable Y", type=["csv"]),
    "X": st.sidebar.file_uploader("Subir CSV de variable X", type=["csv"]),
    "A": st.sidebar.file_uploader("Subir CSV de variable A", type=["csv"]),
}

# 📌 Tabs
tabs = st.tabs(["🔹 Variable Y", "🔹 Variable X", "🔹 Variable A", "🌍 Indicador Global"])

def procesar_variable(nombre, archivo, tab):
    with tab:
        st.header(f"🔍 Análisis para variable {nombre}")
        if archivo is None:
            st.info(f"Sube el archivo CSV de la variable {nombre}.")
            return None, None, None

        try:
            df = pd.read_csv(archivo, encoding="latin1")
            df = df.dropna(subset=["Abscisa"])
        except:
            st.error("Error al leer el archivo.")
            return None, None, None

        umbral = st.number_input(f"Ingrese el umbral mínimo para {nombre}:", min_value=0.0, value=0.3, step=0.1, key=f"umbral_{nombre}")
        fecha_cols = df.columns[2:]
        fechas = [datetime.strptime(f, "%m/%d/%Y") for f in fecha_cols]
        ultima_col = fecha_cols[-1]

        # Indicadores simplificados al final
        try:
            datos = df[ultima_col].astype(float)
            total = len(datos)
            debajo = (datos < umbral).sum()
            st.subheader("📌 Indicadores Básicos")
            st.markdown(f"""
            - **Promedio del último día:** {round(datos.mean(), 3)}
            - **Desviación estándar:** {round(datos.std(), 3)}
            - **% de puntos bajo el umbral ({umbral}):** {round((debajo/total)*100, 1)}%
            """)
        except Exception as e:
            st.error(f"Error en indicadores: {e}")

        return df[["Abscisa", ultima_col]].rename(columns={ultima_col: nombre}), umbral, nombre

# Procesar cada variable
df_y, umbral_y, name_y = procesar_variable("Y", archivos["Y"], tabs[0])
df_x, umbral_x, name_x = procesar_variable("X", archivos["X"], tabs[1])
df_a, umbral_a, name_a = procesar_variable("A", archivos["A"], tabs[2])

# 🌍 Indicador Global Tab
with tabs[3]:
    st.header("🌍 Indicador Global por Abscisa")
    if df_y is not None and df_x is not None and df_a is not None:
        try:
            df_global = df_y.merge(df_x, on="Abscisa").merge(df_a, on="Abscisa")
            df_global["Indicador Global (0-3)"] = (
                (df_global[name_y] < umbral_y).astype(int) +
                (df_global[name_x] < umbral_x).astype(int) +
                (df_global[name_a] < umbral_a).astype(int)
            )
            st.markdown("Número de variables en las que cada abscisa está por debajo del umbral.")
            st.dataframe(df_global[["Abscisa", name_y, name_x, name_a, "Indicador Global (0-3)"]])
        except Exception as e:
            st.error(f"Error al calcular el indicador global: {e}")
    else:
        st.info("Debes subir los tres archivos para ver el indicador global.")
