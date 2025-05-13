
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

    # Gráfico de márgenes
    st.subheader("Distribución de márgenes")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df["Margen"], kde=True, ax=ax)
    ax.axvline(0, color='red', linestyle='--', label='Umbral crítico')
    ax.set_title("Histograma de márgenes respecto al umbral")
    ax.set_xlabel("Margen (valor observado - umbral)")
    ax.legend()
    st.pyplot(fig)

    # Análisis de regresión lineal para todas las abscisas
    st.subheader("Predicción de cruce del umbral por regresión lineal")
    fechas = [datetime.strptime(f, "%m/%d/%Y") for f in opciones_fechas]
    dias = np.array([(f - fechas[0]).days for f in fechas]).reshape(-1, 1)

    resultados_pred = []

    for _, row in df.iterrows():
        abscisa = row["Abscisa"]
        try:
            y_vals = pd.to_numeric(row[opciones_fechas], errors='coerce').values.reshape(-1, 1)
            if np.isnan(y_vals).any():
                continue

            modelo = LinearRegression()
            modelo.fit(dias, y_vals)
            pendiente = modelo.coef_[0][0]
            intercepto = modelo.intercept_[0]
            actual = y_vals[-1][0]
            estado = "Sí" if actual < umbral_minimo else "No"

            if pendiente < 0:
                dias_cruce = (umbral_minimo - intercepto) / pendiente
                fecha_cruce = fechas[0] + timedelta(days=dias_cruce)
                fecha_cruce_str = fecha_cruce.strftime("%Y-%m-%d")
            else:
                fecha_cruce_str = "No aplica"

            resultados_pred.append({
                "Abscisa": abscisa,
                "Pendiente": round(pendiente, 4),
                "Actual (m)": round(actual, 3),
                f"¿Bajo {umbral_minimo}m?": estado,
                f"Cruce estimado de {umbral_minimo}m": fecha_cruce_str
            })
        except:
            continue

    df_pred = pd.DataFrame(resultados_pred)
    st.dataframe(df_pred)
    st.download_button("📥 Descargar predicción", df_pred.to_csv(index=False), file_name="prediccion_umbral.csv")

    # GRÁFICAS por abscisa (Opción A y B)
    st.subheader("📍 Evolución de la profundidad por abscisa")

    fechas_dt = [datetime.strptime(f, "%m/%d/%Y") for f in opciones_fechas]
    abscisa_elegida = st.selectbox("Selecciona una abscisa para graficar:", df["Abscisa"].unique())
    valores = df[df["Abscisa"] == abscisa_elegida][opciones_fechas].values.flatten()
    fig_b, ax_b = plt.subplots()
    ax_b.plot(fechas_dt, valores, marker='o')
    ax_b.set_title(f"Evolución de la profundidad - {abscisa_elegida}")
    ax_b.set_xlabel("Fecha")
    ax_b.set_ylabel("Profundidad (Y)")
    ax_b.grid(True)
    st.pyplot(fig_b)

    st.markdown("---")
    st.subheader("📌 Todas las abscisas")
    for _, row in df.iterrows():
        abscisa = row["Abscisa"]
        y_vals = pd.to_numeric(row[opciones_fechas], errors='coerce').values.flatten()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(fechas_dt, y_vals, marker='o')
        ax.set_title(f"Abscisa: {abscisa}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Profundidad")
        ax.grid(True)
        st.pyplot(fig)
