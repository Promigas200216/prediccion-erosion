
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Predicción por Regresión Lineal", layout="wide")
st.title("📊 Monitoreo y Predicción de Erosión (Regresión Lineal)")

# Carga de archivos
st.sidebar.header("📥 Subir archivos CSV")
archivo_y = st.sidebar.file_uploader("Cargar datos de variable Y", type=["csv"])
archivo_x = st.sidebar.file_uploader("Cargar datos de variable X", type=["csv"])
archivo_a = st.sidebar.file_uploader("Cargar datos de variable A", type=["csv"])

tabs = st.tabs(["Variable Y", "Variable X", "Variable A", "Indicadores Globales"])

def analizar_variable(nombre_var, archivo, tab):
    with tab:
        st.header(f"📘 Análisis de variable {nombre_var}")
        if archivo is not None:
            df = pd.read_csv(archivo, encoding="latin1")
            df = df.dropna(subset=["Abscisa"])
            opciones_fechas = df.columns[2:]
            st.subheader("📌 Visualización de datos")
            st.dataframe(df)

            umbral_minimo = st.number_input(f"Ingresar umbral para {nombre_var}:", min_value=0.0, value=0.3, step=0.1)

            # Calcular margen
            columna_eval = opciones_fechas[-1]
            df["Margen"] = (df[columna_eval] - umbral_minimo).round(3)
            df["Alerta"] = df["Margen"].apply(lambda x: "ALERTA" if x < 0 else "OK")
            st.subheader("🚨 Tabla de alertas")
            st.dataframe(df[["Abscisa", columna_eval, "Margen", "Alerta"]])

            st.subheader("📉 Histograma de márgenes")
            fig, ax = plt.subplots()
            sns.histplot(df["Margen"], kde=True, ax=ax)
            ax.axvline(0, color='red', linestyle='--', label='Umbral')
            ax.set_title("Distribución de Márgenes")
            ax.set_xlabel("Margen (Valor observado - umbral)")
            ax.legend()
            st.pyplot(fig)

            # Predicción por regresión lineal
            st.subheader("📈 Predicción con regresión lineal")
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
                    estado = "OK" if actual >= umbral_minimo else "ALERTA"

                    if pendiente < 0:
                        dias_cruce = (umbral_minimo - intercepto) / pendiente
                        fecha_cruce = fechas[0] + timedelta(days=dias_cruce)
                        fecha_cruce_str = fecha_cruce.strftime("%Y-%m-%d")
                    else:
                        fecha_cruce_str = "No aplica"

                    resultados_pred.append({
                        "Abscisa": abscisa,
                        "Actual": round(actual, 2),
                        "R²": round(modelo.score(dias, y_vals), 4),
                        "Pendiente": round(pendiente, 4),
                        "Estado": estado,
                        "Cruce estimado": fecha_cruce_str
                    })
                except:
                    continue

            df_pred = pd.DataFrame(resultados_pred)
            st.dataframe(df_pred)
            st.download_button("📥 Descargar predicciones", df_pred.to_csv(index=False), file_name=f"prediccion_{nombre_var}.csv")
        else:
            st.warning(f"Por favor, sube el archivo CSV para la variable {nombre_var}.")

# Aplicar análisis a cada pestaña
analizar_variable("Y", archivo_y, tabs[0])
analizar_variable("X", archivo_x, tabs[1])
analizar_variable("A", archivo_a, tabs[2])

# Indicadores globales
with tabs[3]:
    st.header("📊 Indicadores globales")
    if all([archivo_y, archivo_x, archivo_a]):
        dfs = {}
        for nombre, archivo, umbral in zip(["Y", "X", "A"], [archivo_y, archivo_x, archivo_a], [0.3, 2.0, 1.0]):
            try:
    df = pd.read_csv(archivo, encoding='latin1')
    if df.empty:
        st.warning(f"⚠️ El archivo de {nombre} está vacío.")
        continue
    df = df.dropna(subset=["Abscisa"])
except Exception as e:
    st.error(f"❌ Error al leer el archivo de {nombre}: {str(e)}")
    continue
            col_eval = df.columns[-1]
            df["Alerta"] = df[col_eval] < umbral
            dfs[nombre] = df[["Abscisa", "Alerta"]].rename(columns={"Alerta": f"Alerta_{nombre}"})

        df_merged = dfs["Y"].merge(dfs["X"], on="Abscisa").merge(dfs["A"], on="Abscisa")
        df_merged["Total_alertas"] = df_merged[["Alerta_Y", "Alerta_X", "Alerta_A"]].sum(axis=1)

        def clasificar(fila):
            if fila["Total_alertas"] >= 2:
                return "Crítica"
            elif fila["Total_alertas"] == 1:
                return "En riesgo"
            return "Estable"

        df_merged["Estado"] = df_merged.apply(clasificar, axis=1)
        st.dataframe(df_merged[["Abscisa", "Estado"]])
    else:
        st.info("Sube los tres archivos para ver los indicadores combinados.")
