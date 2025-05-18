
import streamlit as st
import pandas as pd
import numpy as np

st.title("App de Regresión Lineal - Ejemplo Estable")

archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo is not None:
    if archivo.size == 0:
        st.warning("⚠️ El archivo está vacío.")
    else:
        try:
            df = pd.read_csv(archivo, encoding='latin1')
            df = df.dropna(subset=["Abscisa"])
            st.success("✅ Archivo cargado correctamente.")
            st.dataframe(df)
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")
