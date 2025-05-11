
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("Análisis y Predicción de Erosión (Y)")

archivo = st.file_uploader("📄 Sube el archivo CSV con datos de Y", type=["csv"])

if archivo:
    df = pd.read_csv(archivo, encoding="latin1")
    df = df.dropna(subset=["Abscisa"])

    opciones_fechas = df.columns[2:]
    fechas_dt = [datetime.strptime(f, "%m/%d/%Y") for f in opciones_fechas]
    dias = np.array([(f - fechas_dt[0]).days for f in fechas_dt]).reshape(-1, 1)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Vista general", 
        "📈 Comparación con primera y anterior",
        "🔥 Mapa de calor",
        "📆 Erosión mensual",
        "📉 Predicción y alertas"
    ])

    with tab1:
        st.subheader("Tabla original")
        st.dataframe(df)

    with tab2:
        st.subheader("Comparación con la primera fecha")
        df_dif = df[["Abscisa"]].copy()
        for col in opciones_fechas:
            df_dif[col] = (df[col] - df[opciones_fechas[0]]).round(3)
        st.dataframe(df_dif)

        st.subheader("Comparación con la fecha anterior")
        df_diff_ant = df[["Abscisa"]].copy()
        for i in range(1, len(opciones_fechas)):
            df_diff_ant[opciones_fechas[i]] = (df[opciones_fechas[i]] - df[opciones_fechas[i - 1]]).round(3)
        st.dataframe(df_diff_ant)

    with tab3:
        st.subheader("Mapa de calor: diferencia con respecto a primera fecha")
        df_heat = df_dif.drop(columns=["Abscisa"])
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df_heat.astype(float), cmap="coolwarm", xticklabels=opciones_fechas, yticklabels=df["Abscisa"])
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with tab4:
        st.subheader("Promedio mensual de erosión")
        df_melt = df.melt(id_vars=["Abscisa"], value_vars=opciones_fechas, var_name="Fecha", value_name="Valor")
        df_melt["Fecha"] = pd.to_datetime(df_melt["Fecha"], format="%m/%d/%Y")
        df_melt["Mes"] = df_melt["Fecha"].dt.to_period("M").astype(str)
        promedio_mensual = df_melt.groupby("Mes")["Valor"].mean()
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.barplot(x=promedio_mensual.index, y=promedio_mensual.values, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Promedio de profundidad por mes")
        ax.set_ylabel("Profundidad promedio")
        st.pyplot(fig)

    with tab5:
        st.subheader("Predicción por regresión lineal y alertas")

        umbral = st.number_input("🔻 Umbral mínimo (m)", value=0.6, step=0.1)

        columna_alerta = st.selectbox("📅 Fecha a evaluar para alerta:", opciones_fechas)

        # Crear columnas Margen y Alerta
        df["Margen"] = (df[columna_alerta] - umbral).round(3)
        df["Alerta"] = df["Margen"].apply(lambda x: "ALERTA" if x < 0 else "OK")
        st.write("📌 Alerta por umbral en fecha seleccionada:")
        st.dataframe(df[["Abscisa", columna_alerta, "Margen", "Alerta"]])

        # Predicción por regresión lineal
        resultados = []
        for _, row in df.iterrows():
            abscisa = row["Abscisa"]
            y_vals = pd.to_numeric(row[opciones_fechas], errors='coerce').values.reshape(-1, 1)
            if np.isnan(y_vals).any():
                continue
            modelo = LinearRegression()
            modelo.fit(dias, y_vals)
            pendiente = modelo.coef_[0][0]
            intercepto = modelo.intercept_[0]
            actual = y_vals[-1][0]
            estado = "Sí" if actual < umbral else "No"
            if pendiente < 0:
                dias_cruce = (umbral - intercepto) / pendiente
                fecha_cruce = fechas_dt[0] + timedelta(days=dias_cruce)
                fecha_cruce_str = fecha_cruce.strftime("%Y-%m-%d")
            else:
                fecha_cruce_str = "No aplica"
            resultados.append({
                "Abscisa": abscisa,
                "Pendiente": round(pendiente, 4),
                "Actual (m)": round(actual, 3),
                f"¿Bajo {umbral}m?": estado,
                f"Cruce estimado de {umbral}m": fecha_cruce_str
            })

        df_pred = pd.DataFrame(resultados)
        st.write("📈 Resultados de regresión lineal por abscisa")
        st.dataframe(df_pred)
        st.download_button("📥 Descargar resultados de predicción", df_pred.to_csv(index=False), file_name="prediccion_umbral.csv")
