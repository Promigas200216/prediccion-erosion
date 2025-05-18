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
    # Este bloque permanece sin cambios
    pass  # OMITIDO para brevedad

# Aplicar por variable
procesar_variable("Y", archivos["Y"], tabs[0])
procesar_variable("X", archivos["X"], tabs[1])
procesar_variable("A", archivos["A"], tabs[2])

# 📊 Indicadores globales (corregido)
with tabs[3]:
    st.header("📊 Indicadores globales")
    st.markdown("""
    Esta pestaña resume los principales indicadores para las variables monitoreadas:
    - **Umbral**: valor de referencia ajustable para identificar puntos críticos.
    - **% en alerta**: porcentaje de puntos bajo el umbral.
    - **Promedio y Desviación estándar**: resumen de comportamiento al cierre.
    - **Pendiente promedio**: tendencia general (negativa implica deterioro).
    - **% con cruce estimado**: puntos cuya tendencia indica cruce futuro del umbral.
    - **Fecha promedio de cruce**: predicción aproximada del momento de mayor riesgo.
    - **Abscisa más crítica**: punto con menor valor final registrado.
    """)

    umbral_global = st.number_input("📉 Umbral global para alerta:", min_value=0.0, value=0.3, step=0.1)

    def analizar_variable(nombre, archivo, umbral):
        if archivo is None:
            st.warning(f"⚠️ No se ha cargado el archivo para la variable {nombre}.")
            return None
        try:
            df = pd.read_csv(archivo, encoding="latin1")
            st.write(f"✅ Archivo de {nombre} cargado correctamente con {df.shape[0]} filas.")
            df = df.dropna(subset=["Abscisa"])
        except Exception as e:
            st.error(f"❌ Error al leer el archivo de {nombre}: {e}")
            return None

        try:
            fecha_cols = df.columns[2:]
            try:
                fechas = [datetime.strptime(f, "%m/%d/%Y") for f in fecha_cols]
            except Exception as e:
                st.error(f"❌ Error al interpretar fechas para {nombre}: {e}")
                return None

            dias = np.array([(f - fechas[0]).days for f in fechas]).reshape(-1, 1)
            ultima_fecha = fecha_cols[-1]
            valores_finales = df[ultima_fecha].astype(float)
            total = len(valores_finales)
            alerta_pct = 100 * sum(valores_finales < umbral) / total if total > 0 else 0
            promedio = valores_finales.mean()
            desviacion = valores_finales.std()
            pendientes = []
            fechas_cruce = []
            valores_finales_np = valores_finales.values

            abscisa_min = df.iloc[valores_finales_np.argmin()]["Abscisa"]

            for _, row in df.iterrows():
                y_vals = row[fecha_cols].values.astype(float).reshape(-1, 1)
                if np.isnan(y_vals).any():
                    continue
                modelo = LinearRegression()
                modelo.fit(dias, y_vals)
                pendiente = modelo.coef_[0][0]
                pendientes.append(pendiente)
                intercepto = modelo.intercept_[0]
                if pendiente < 0:
                    dias_cruce = (umbral - intercepto) / pendiente
                    fecha_cruce = fechas[0] + timedelta(days=int(dias_cruce))
                    fechas_cruce.append(fecha_cruce)

            pendiente_prom = np.mean(pendientes)
            cruce_pct = 100 * len(fechas_cruce) / total if total > 0 else 0
            fecha_prom_cruce = np.mean(fechas_cruce).strftime("%Y-%m-%d") if fechas_cruce else "No aplica"

            return {
                "Variable": nombre,
                "Umbral": umbral,
                "% en alerta": f"{alerta_pct:.1f}%",
                "Prom. final": round(promedio, 3),
                "Desv.": round(desviacion, 3),
                "Pend. prom.": round(pendiente_prom, 4),
                "% con cruce": f"{cruce_pct:.1f}%",
                "Fecha prom. cruce": fecha_prom_cruce,
                "Abscisa más crítica": abscisa_min
            }

        except Exception as e:
            st.error(f"❌ Error al analizar la variable {nombre}: {e}")
            return None

    resumen_final = []
    for nombre in ["Y", "X", "A"]:
        r = analizar_variable(nombre, archivos[nombre], umbral_global)
        if r:
            resumen_final.append(r)

    if resumen_final:
        df_resumen = pd.DataFrame(resumen_final)
        st.dataframe(df_resumen)
    else:
        st.warning("⚠️ No se han podido calcular los indicadores. Verifica que los archivos estén cargados correctamente.")
