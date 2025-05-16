
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📡 Monitoreo y predicción de erosión - Indicadores Globales")

# Carga de archivos
col1, col2, col3 = st.columns(3)
with col1:
    archivo_y = st.file_uploader("📄 Sube archivo Y", type=["csv"])
with col2:
    archivo_x = st.file_uploader("📄 Sube archivo X", type=["csv"])
with col3:
    archivo_a = st.file_uploader("📄 Sube archivo A", type=["csv"])

df_y, df_x, df_a = None, None, None
if archivo_y:
    df_y = pd.read_csv(archivo_y, encoding="latin1").dropna(subset=["Abscisa"])
if archivo_x:
    df_x = pd.read_csv(archivo_x, encoding="latin1").dropna(subset=["Abscisa"])
if archivo_a:
    df_a = pd.read_csv(archivo_a, encoding="latin1").dropna(subset=["Abscisa"])

def calcular_alertas(df, fecha, umbral):
    df = df.copy()
    df["Margen"] = pd.to_numeric(df[fecha], errors="coerce") - umbral
    df["Alerta"] = df["Margen"].apply(lambda x: "ALERTA" if x < 0 else "OK")
    return df

if all([df_y is not None, df_x is not None, df_a is not None]):
    st.header("📊 Indicadores Globales")

    fechas_comunes = sorted(list(set(c for c in df_y.columns if "/" in c) &
                                 set(c for c in df_x.columns if "/" in c) &
                                 set(c for c in df_a.columns if "/" in c)),
                            key=lambda x: datetime.strptime(x, "%m/%d/%Y"))

    if fechas_comunes:
        fecha_base = st.selectbox("Selecciona la fecha a evaluar", fechas_comunes)

        col1, col2, col3 = st.columns(3)
        with col1:
            umbral_y = st.number_input("Umbral mínimo Y", value=0.6)
        with col2:
            umbral_x = st.number_input("Umbral mínimo X", value=2.0)
        with col3:
            umbral_a = st.number_input("Umbral mínimo A", value=0.3)

        df_y = calcular_alertas(df_y, fecha_base, umbral_y)
        df_x = calcular_alertas(df_x, fecha_base, umbral_x)
        df_a = calcular_alertas(df_a, fecha_base, umbral_a)

        merged = df_y[["Abscisa"]].copy()
        merged = merged.merge(df_y[["Abscisa", "Alerta"]].rename(columns={"Alerta": "Alerta_Y"}), on="Abscisa")
        merged = merged.merge(df_x[["Abscisa", "Alerta"]].rename(columns={"Alerta": "Alerta_X"}), on="Abscisa")
        merged = merged.merge(df_a[["Abscisa", "Alerta"]].rename(columns={"Alerta": "Alerta_A"}), on="Abscisa")

        def estado(row):
            count = sum([row["Alerta_Y"] == "ALERTA", row["Alerta_X"] == "ALERTA", row["Alerta_A"] == "ALERTA"])
            if count >= 2:
                return "🔴 CRÍTICO"
            elif count == 1:
                return "🟡 RIESGO"
            else:
                return "🟢 ESTABLE"

        merged["Estado combinado"] = merged.apply(estado, axis=1)

        st.subheader("📈 Resultados")
        col1, col2, col3 = st.columns(3)
        total = len(merged)
        crit = (merged["Estado combinado"] == "🔴 CRÍTICO").sum()
        risk = (merged["Estado combinado"] == "🟡 RIESGO").sum()
        ok = (merged["Estado combinado"] == "🟢 ESTABLE").sum()
        with col1:
            st.metric("🔴 Críticas", f"{crit} ({round(100*crit/total,1)}%)")
        with col2:
            st.metric("🟡 En riesgo", f"{risk} ({round(100*risk/total,1)}%)")
        with col3:
            st.metric("🟢 Estables", f"{ok} ({round(100*ok/total,1)}%)")

        st.markdown("**📌 ¿Qué significan estos indicadores?**")
        st.markdown("""
        - 🔴 **Críticas**: cruzan umbral en 2 o más variables.
        - 🟡 **En riesgo**: cruzan en solo una.
        - 🟢 **Estables**: sin cruces.
        """)
        st.dataframe(merged)
    else:
        st.warning("No hay fechas comunes entre los tres archivos.")
else:
    st.info("Sube los tres archivos para activar los indicadores.")
