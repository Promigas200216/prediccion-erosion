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
tabs = st.tabs(["🔹 Variable Y", "🔹 Variable X", "🔹 Variable A"])

# Función para procesar una variable y devolver DataFrame final
def procesar_variable(nombre, archivo, tab, agregar_global=False):
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

        st.subheader("📊 Histograma del margen")
        fig, ax = plt.subplots()
        sns.histplot(df_umbral["Margen"], kde=True, ax=ax)
        ax.axvline(0, color='red', linestyle='--')
        ax.set_title("Distribución del margen respecto al umbral")
        st.pyplot(fig)

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

        # Indicadores Básicos
        st.subheader("📌 Indicadores Básicos")
        try:
            ultima_col = fecha_cols[-1]
            datos = df[ultima_col].astype(float)
            total = len(datos)
            debajo = (datos < umbral).sum()
            st.markdown(f"""
            - **Promedio del último día:** {round(datos.mean(), 3)}
            - **Desviación estándar:** {round(datos.std(), 3)}
            - **% de puntos bajo el umbral ({umbral}):** {round((debajo/total)*100, 1)}%
            """)
        except Exception as e:
            st.error(f"Error al calcular los indicadores: {e}")

        if agregar_global:
            return df[["Abscisa", ultima_col]].rename(columns={ultima_col: nombre, "Abscisa": "Abscisa"}), umbral, nombre
        else:
            return None, None, None

# Ejecutar
df_y, umbral_y, name_y = procesar_variable("Y", archivos["Y"], tabs[0])
df_x, umbral_x, name_x = procesar_variable("X", archivos["X"], tabs[1])
df_a, umbral_a, name_a = procesar_variable("A", archivos["A"], tabs[2], agregar_global=True)

# Indicador global final
if df_y is not None and df_x is not None and df_a is not None:
    try:
        df_global = df_y.merge(df_x, on="Abscisa").merge(df_a, on="Abscisa")
        df_global["Indicador Global (0-3)"] = (
            (df_global[name_y] < umbral_y).astype(int) +
            (df_global[name_x] < umbral_x).astype(int) +
            (df_global[name_a] < umbral_a).astype(int)
        )
        st.subheader("🌍 Indicador Global por Abscisa")
        st.markdown("Número de variables en las que cada abscisa está por debajo del umbral.")
        st.dataframe(df_global[["Abscisa", name_y, name_x, name_a, "Indicador Global (0-3)"]])
    except Exception as e:
        st.error(f"Error al calcular el indicador global: {e}")
