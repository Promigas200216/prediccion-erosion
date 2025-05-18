
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

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
tabs = st.tabs(["🔹 Variable Y", "🔹 Variable X", "🔹 Variable A", "📊 Indicadores Globales"])

# 🧠 Lógica reutilizable
def procesar_variable(nombre, archivo, tab):
    with tab:
        st.header(f"🔍 Análisis para variable {nombre}")
        if archivo is None:
            st.info(f"Sube el archivo CSV de la variable {nombre}.")
            return

        try:
            df = pd.read_csv(archivo, encoding="latin1")
            df = df.dropna(subset=["Abscisa"])
        except:
            st.error("Error al leer el archivo.")
            return

        # Umbral
        umbral = st.number_input(f"Ingrese el umbral mínimo para {nombre}:", min_value=0.0, value=0.3, step=0.1)

        fecha_cols = df.columns[2:]
        fechas = [datetime.strptime(f, "%m/%d/%Y") for f in fecha_cols]
        dias = np.array([(f - fechas[0]).days for f in fechas]).reshape(-1, 1)

        st.subheader("📌 Visualización de datos")
        st.dataframe(df)

        st.subheader("📉 Comparación con la primera fecha")
        df_dif = df[["Abscisa"]].copy()
        for col in fecha_cols:
            df_dif[col] = (df[col] - df[fecha_cols[0]]).round(3)
        st.dataframe(df_dif)

        st.subheader("📈 Comparación con la fecha anterior")
        df_step = df[["Abscisa"]].copy()
        for i in range(1, len(fecha_cols)):
            anterior = fecha_cols[i-1]
            actual = fecha_cols[i]
            df_step[actual] = (df[actual] - df[anterior]).round(3)
        st.dataframe(df_step)

        st.subheader("🚦 Análisis respecto al umbral")
        df_umbral = df[["Abscisa"]].copy()
        df_umbral["Actual"] = df[fecha_cols[-1]]
        df_umbral["Margen"] = (df_umbral["Actual"] - umbral).round(3)
        df_umbral["Alerta"] = df_umbral["Margen"].apply(lambda x: "ALERTA" if x < 0 else "OK")
        st.dataframe(df_umbral.style.apply(lambda row: ['background-color: red; color: white' if v == "ALERTA" else '' for v in row], axis=1))

        # 🔮 Regresión lineal
        st.subheader("📅 Predicción de cruce del umbral (Regresión Lineal)")
        resultados_pred = []
        for _, row in df.iterrows():
            abscisa = row["Abscisa"]
            y_vals = row[fecha_cols].values.astype(float).reshape(-1, 1)
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
                fecha_cruce = fechas[0] + timedelta(days=int(dias_cruce))
                fecha_cruce_str = fecha_cruce.strftime("%Y-%m-%d")
            else:
                fecha_cruce_str = "No aplica"
            resultados_pred.append({
                "Abscisa": abscisa,
                "Pendiente": round(pendiente, 4),
                "Actual": round(actual, 3),
                f"¿Bajo {umbral}?": estado,
                "Cruce estimado": fecha_cruce_str
            })
        st.dataframe(pd.DataFrame(resultados_pred))

        # 📊 Histograma
        st.subheader("📊 Histograma del margen")
        fig, ax = plt.subplots()
        sns.histplot(df_umbral["Margen"], kde=True, ax=ax)
        ax.axvline(0, color='red', linestyle='--')
        ax.set_title("Distribución del margen respecto al umbral")
        st.pyplot(fig)

        # 📈 Gráfica por abscisa
        st.subheader("📌 Seleccionar abscisa para gráfico temporal")
        abscisa_sel = st.selectbox("Elige una abscisa:", df["Abscisa"].unique(), key=nombre)
        serie = df[df["Abscisa"] == abscisa_sel][fecha_cols].values.flatten()
        fig2, ax2 = plt.subplots()
        ax2.plot(fechas, serie, marker='o')
        ax2.axhline(umbral, color='red', linestyle='--')
        ax2.set_title(f"Evolución temporal de {nombre} en abscisa {abscisa_sel}")
        ax2.set_ylabel("Valor observado")
        ax2.set_xlabel("Fecha")
        st.pyplot(fig2)

# Aplicar por variable
procesar_variable("Y", archivos["Y"], tabs[0])
procesar_variable("X", archivos["X"], tabs[1])
procesar_variable("A", archivos["A"], tabs[2])

# 📊 Indicadores globales
with tabs[3]:
    st.header("📊 Indicadores globales")
    if all(v is not None for v in archivos.values()):
        try:
            resultados = []
            resumen_abs = []

            for var, archivo in archivos.items():
                df = pd.read_csv(archivo, encoding="latin1").dropna(subset=["Abscisa"])
                fecha_cols = df.columns[2:]
                ult_col = fecha_cols[-1]
                datos = df[ult_col].astype(float)

                media = datos.mean()
                std = datos.std()
                min_val = datos.min()
                max_val = datos.max()

                resultados.append({
                    "Variable": var,
                    "Fecha más reciente": ult_col,
                    "Media": round(media, 3),
                    "Desviación estándar": round(std, 3),
                    "Mínimo": round(min_val, 3),
                    "Máximo": round(max_val, 3)
                })

                for _, row in df.iterrows():
                    resumen_abs.append({
                        "Variable": var,
                        "Abscisa": row["Abscisa"],
                        "Valor actual": row[ult_col]
                    })

            df_ind = pd.DataFrame(resultados)
            st.dataframe(df_ind)

            # 📌 Tabla resumen por abscisa
            st.subheader("📋 Resumen por abscisa y variable")
            df_resumen = pd.DataFrame(resumen_abs)
            st.dataframe(df_resumen)

            # 📈 Gráfica resumen
            st.subheader("📈 Comparación de medias por variable")
            fig, ax = plt.subplots()
            sns.barplot(x="Variable", y="Media", data=df_ind, ax=ax, palette="Set2")
            ax.set_title("Media de valores por variable en la última fecha")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error al calcular indicadores globales: {e}")
    else:
        st.info("Carga los tres archivos CSV para ver los indicadores.")
