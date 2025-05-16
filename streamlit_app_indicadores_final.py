
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📡 Monitoreo y predicción de erosión - Múltiples variables (X, Y, A)")

# Carga de archivos CSV por variable
col1, col2, col3 = st.columns(3)
with col1:
    archivo_y = st.file_uploader("📄 Sube archivo de datos Y", type=["csv"])
with col2:
    archivo_x = st.file_uploader("📄 Sube archivo de datos X", type=["csv"])
with col3:
    archivo_a = st.file_uploader("📄 Sube archivo de datos A", type=["csv"])

# Leer CSVs
df_y, df_x, df_a = None, None, None
if archivo_y:
    df_y = pd.read_csv(archivo_y, encoding="latin1").dropna(subset=["Abscisa"])
if archivo_x:
    df_x = pd.read_csv(archivo_x, encoding="latin1").dropna(subset=["Abscisa"])
if archivo_a:
    df_a = pd.read_csv(archivo_a, encoding="latin1").dropna(subset=["Abscisa"])

# Función de predicción
def prediccion_regresion(df, umbral):
    fechas = [c for c in df.columns if "/" in c]
    fechas_dt = [datetime.strptime(f, "%m/%d/%Y") for f in fechas]
    dias = np.array([(f - fechas_dt[0]).days for f in fechas_dt]).reshape(-1, 1)
    predicciones = []
    for _, row in df.iterrows():
        try:
            y = pd.to_numeric(row[fechas], errors='coerce').values.reshape(-1, 1)
            if np.isnan(y).any():
                continue
            modelo = LinearRegression().fit(dias, y)
            pendiente = modelo.coef_[0][0]
            intercepto = modelo.intercept_[0]
            actual = y[-1][0]
            estado = "Sí" if actual < umbral else "No"
            fecha_cruce = "No aplica"
            if pendiente < 0:
                dias_cruce = (umbral - intercepto) / pendiente
                fecha_cruce = (fechas_dt[0] + timedelta(days=dias_cruce)).strftime("%Y-%m-%d")
            predicciones.append({
                "Abscisa": row["Abscisa"],
                "Pendiente": round(pendiente, 4),
                "Actual (m)": round(actual, 3),
                f"¿Bajo {umbral}m?": estado,
                f"Cruce estimado de {umbral}m": fecha_cruce
            })
        except:
            continue
    return pd.DataFrame(predicciones)

# Tabs
tabs = []
if df_y is not None:
    tabs.append("🔹 Variable Y")
if df_x is not None:
    tabs.append("🔹 Variable X")
if df_a is not None:
    tabs.append("🔹 Variable A")
if all([df_y is not None, df_x is not None, df_a is not None]):
    tabs.append("🔀 Análisis combinado")
    tabs.append("📊 Indicadores globales")

if tabs:
    seleccion = st.tabs(tabs)
    idx = 0

    if df_y is not None:
        with seleccion[idx]:
            st.header("🔹 Análisis Y")
            fechas = [c for c in df_y.columns if "/" in c]
            fecha_eval = st.selectbox("Fecha Y", fechas)
            umbral_y = st.number_input("Umbral mínimo Y", value=0.6)
            df_y["Margen"] = pd.to_numeric(df_y[fecha_eval], errors="coerce") - umbral_y
            df_y["Alerta"] = df_y["Margen"].apply(lambda x: "ALERTA" if x < 0 else "OK")
            st.dataframe(df_y[["Abscisa", fecha_eval, "Margen", "Alerta"]])
            pred_y = prediccion_regresion(df_y, umbral_y)
            st.dataframe(pred_y)
        idx += 1

    if df_x is not None:
        with seleccion[idx]:
            st.header("🔹 Análisis X")
            fechas = [c for c in df_x.columns if "/" in c]
            fecha_eval = st.selectbox("Fecha X", fechas)
            umbral_x = st.number_input("Umbral mínimo X", value=2.0)
            df_x["Margen"] = pd.to_numeric(df_x[fecha_eval], errors="coerce") - umbral_x
            df_x["Alerta"] = df_x["Margen"].apply(lambda x: "ALERTA" if x < 0 else "OK")
            st.dataframe(df_x[["Abscisa", fecha_eval, "Margen", "Alerta"]])
            pred_x = prediccion_regresion(df_x, umbral_x)
            st.dataframe(pred_x)
        idx += 1

    if df_a is not None:
        with seleccion[idx]:
            st.header("🔹 Análisis A")
            fechas = [c for c in df_a.columns if "/" in c]
            fecha_eval = st.selectbox("Fecha A", fechas)
            umbral_a = st.number_input("Umbral mínimo A", value=0.3)
            df_a["Margen"] = pd.to_numeric(df_a[fecha_eval], errors="coerce") - umbral_a
            df_a["Alerta"] = df_a["Margen"].apply(lambda x: "ALERTA" if x < 0 else "OK")
            st.dataframe(df_a[["Abscisa", fecha_eval, "Margen", "Alerta"]])
            pred_a = prediccion_regresion(df_a, umbral_a)
            st.dataframe(pred_a)
        idx += 1

    if all([df_y is not None, df_x is not None, df_a is not None]):
        with seleccion[idx]:
            st.header("🔀 Análisis combinado")
            pred_y = prediccion_regresion(df_y, umbral_y)
            pred_x = prediccion_regresion(df_x, umbral_x)
            pred_a = prediccion_regresion(df_a, umbral_a)
            merged = df_y[["Abscisa"]].copy()
            merged = merged.merge(pred_y, on="Abscisa", how="left")
            merged = merged.merge(pred_x, on="Abscisa", how="left", suffixes=("", "_X"))
            merged = merged.merge(pred_a, on="Abscisa", how="left", suffixes=("", "_A"))

            def semaforo(row):
                total = 0
                for col in merged.columns:
                    if "¿Bajo" in col and row[col] == "Sí":
                        total += 1
                return "🔴 CRÍTICO" if total >= 2 else "🟡 RIESGO" if total == 1 else "🟢 ESTABLE"

            merged["Estado combinado"] = merged.apply(semaforo, axis=1)
            st.dataframe(merged)
        idx += 1

        with seleccion[idx]:
            st.header("📊 Indicadores globales")
            col1, col2, col3 = st.columns(3)
            criticas = merged["Estado combinado"].eq("🔴 CRÍTICO").sum()
            riesgo = merged["Estado combinado"].eq("🟡 RIESGO").sum()
            estables = merged["Estado combinado"].eq("🟢 ESTABLE").sum()
            total = len(merged)
            with col1:
                st.metric("🔴 Zonas críticas", f"{criticas} ({round(100*criticas/total,1)}%)")
            with col2:
                st.metric("🟡 En riesgo", f"{riesgo} ({round(100*riesgo/total,1)}%)")
            with col3:
                st.metric("🟢 Estables", f"{estables} ({round(100*estables/total,1)}%)")

            st.markdown("**📌 ¿Qué significan estos indicadores?**")
            st.markdown("""
            - 🔴 **Críticas**: Abscisas que cruzan el umbral en 2 o más variables. Riesgo alto de exposición.
            - 🟡 **En riesgo**: Abscisas que cruzan en solo una variable.
            - 🟢 **Estables**: No presentan cruces en ninguna variable actualmente.
            """)
