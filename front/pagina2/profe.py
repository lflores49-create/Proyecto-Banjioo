import streamlit as st
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Regresión Financiera Dinámica", layout="wide")

# --- ENCABEZADO PRINCIPAL (estilo dashboard) ---
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #111;
        font-size: 52px;
        font-weight: 700;
        padding: 20px 0 10px 0;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 20px;
        font-style: italic;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .divider {
        border: none;
        height: 2px;
        background: linear-gradient(to right, #2E8B57, #4682B4);
        margin-bottom: 30px;
        width: 80%;
        margin-left: auto;
        margin-right: auto;
        border-radius: 10px;
    }
    </style>

    <h1 class="main-title">📈 Análisis Predictivo de Activos Bursátiles</h1>
    <p class="subtitle">Modelos de regresión y correlación con datos en tiempo real de Yahoo Finance</p>
    <hr class="divider">
""", unsafe_allow_html=True)

st.markdown("""
Esta aplicación te permite analizar la relación entre activos financieros utilizando datos históricos de **Yahoo Finance** y modelos de regresión de **Statsmodels**.
""")

# --- SECCIÓN DE ENTRADA ---
st.sidebar.header("Configuración del modelo")

# Lista de tickers populares (puedes ampliarla)
tickers_default = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NFLX", "SPY", "F", "GM"]

tickers = st.sidebar.multiselect(
    "Selecciona los activos disponibles:",
    tickers_default,
    default=["F", "SPY", "AAPL", "TSLA"]
)

if len(tickers) < 2:
    st.warning("Selecciona al menos dos activos para continuar.")
    st.stop()

# Fechas
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Fecha de inicio", pd.to_datetime("2023-01-01"))
end_date = col2.date_input("Fecha final", pd.to_datetime("today"))

# --- DESCARGA DE DATOS ---
st.write("### 🔄 Descargando datos desde Yahoo Finance...")

with st.spinner("Obteniendo información de los activos..."):
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, threads=True)

if raw.empty:
    st.error("❌ No se descargaron datos para los tickers o fechas seleccionados.")
    st.stop()

# --- LIMPIEZA AUTOMÁTICA DEL MULTIINDEX ---
if isinstance(raw.columns, pd.MultiIndex):
    try:
        data = raw['Adj Close'].dropna()
    except KeyError:
        # si no existe 'Adj Close', usar Close
        try:
            data = raw.xs('Close', axis=1, level=1).dropna()
            st.warning("Usando precios de cierre ('Close') ya que no se encontraron precios ajustados.")
        except Exception:
            st.error("No se pudo encontrar una columna válida de precios ('Adj Close' o 'Close').")
            st.stop()
else:
    if 'Adj Close' in raw.columns:
        data = raw['Adj Close'].dropna()
    elif 'Close' in raw.columns:
        data = raw['Close'].dropna()
        st.warning("Usando 'Close' porque no existe 'Adj Close'.")
    else:
        st.error("No se encontraron columnas válidas ('Adj Close' o 'Close').")
        st.stop()

# --- MENSAJE RESUMIDO DE ÉXITO ---
st.success("✅ Datos descargados correctamente.")

# Mostrar un resumen limpio
st.markdown(f"""
**📅 Periodo:** {data.index.min().date()} → {data.index.max().date()}  
**📊 Activos cargados:** {", ".join(data.columns)}
""")


# --- SELECCIÓN DE VARIABLES ---
dep_var = st.selectbox("Selecciona el activo dependiente (a explicar):", tickers)
indep_vars = st.multiselect(
    "Selecciona los activos regresores (explicativos):",
    [t for t in tickers if t != dep_var],
    default=[t for t in tickers if t != dep_var][:2]
)

if not indep_vars:
    st.warning("Selecciona al menos un activo regresor.")
    st.stop()

# --- PREPARAR DATOS ---
df = data[[dep_var] + indep_vars].pct_change().dropna() * 100  # Retornos %
Y = df[dep_var]
X = df[indep_vars]
X = sm.add_constant(X)

# --- MODELO DE REGRESIÓN ---
model = sm.OLS(Y, X).fit()

# --- RESULTADOS ---
st.subheader("📊 Resultados de la regresión")
st.write(model.summary())

# --- GRÁFICOS ---
st.subheader("📈 Visualización de retornos")

fig, ax = plt.subplots(figsize=(10, 5))
for col in indep_vars:
    ax.scatter(df[col], df[dep_var], alpha=0.6, label=col)
ax.set_xlabel("Retornos de los regresores (%)")
ax.set_ylabel(f"Retornos de {dep_var} (%)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- RESIDUALES ---
st.subheader("🔍 Análisis de residuales")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(model.resid, color='purple')
ax2.set_title("Residuales del modelo")
ax2.grid(True)
st.pyplot(fig2)

# --- CORRELACIONES ---
st.subheader("📉 Matriz de correlaciones")
st.dataframe(df.corr().style.background_gradient(cmap="coolwarm"))

# --- DESCARGAR RESULTADOS ---
csv = df.to_csv().encode("utf-8")
st.download_button("📥 Descargar datos usados", data=csv, file_name="datos_regresion.csv", mime="text/csv", key="download_regresion")

# --- CUADRO INFORMATIVO FINAL (versión con "Nota" arriba) ---
st.markdown("""
    <style>
    .info-box {
        background-color: #f8f9fa;
        border-left: 6px solid #2E8B57;
        border-radius: 10px;
        padding: 18px 28px;
        margin-top: 45px;
        margin-bottom: 25px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    .info-title {
        font-size: 22px;
        font-weight: 700;
        color: #111;
        margin-bottom: 10px;
    }
    .info-text {
        font-size: 16px;
        color: #333;
        line-height: 1.6;
    }
    </style>

    <div class="info-box">
        <p class="info-title">📝 Nota</p>
        <p class="info-text">
            <b>Autores:</b> ChatGPT 🤖 y Leonardo Flores 💼<br>
            <b>Supervisado por:</b> Andrés Ferro 🧠, gran programador y teso en finanzas.
        </p>
    </div>
""", unsafe_allow_html=True)
