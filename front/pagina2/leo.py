import streamlit as st
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import numpy as np
import tempfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter


# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Regresión Financiera Dinámica - Avanzada", layout="wide")

st.markdown("""
    <style>
    .main-title { text-align: center; color: #111; font-size: 42px; font-weight: 700; padding: 10px 0; }
    .subtitle { text-align: center; color: #555; font-size: 16px; margin-top: -10px; margin-bottom: 20px; }
    </style>
    <h1 class="main-title">📈 Regresión Financiera Dinámica — Versión Avanzada</h1>
    <p class="subtitle">Incluye diagnóstico estadístico, CAPM beta, interpretación automática, predicción con intervalos y gráficos de ajuste.</p>
""", unsafe_allow_html=True)

# --- SIDEBAR: CONFIGURACIÓN DE DATOS Y MODELO ---
st.sidebar.header("Configuración de datos y modelo")

tickers_default = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NFLX", "SPY", "F", "GM"]

tickers = st.sidebar.multiselect("Selecciona los activos:", tickers_default, default=["F", "SPY", "AAPL", "TSLA"])
if len(tickers) < 2:
    st.sidebar.warning("Selecciona al menos dos activos para continuar.")

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Fecha de inicio", pd.to_datetime("2023-01-01"))
end_date = col2.date_input("Fecha final", pd.to_datetime("today"))

use_adjusted = st.sidebar.checkbox("Usar precios ajustados (Adj Close) cuando estén disponibles", value=True)

# CAPM settings
st.sidebar.markdown("---")
st.sidebar.header("CAPM y predicción")
market_ticker = st.sidebar.text_input("Ticker de mercado (para CAPM)", value="SPY")
rf_annual = st.sidebar.number_input("Tasa libre de riesgo anual (%)", value=4.0, min_value=0.0, step=0.1)

# --- DESCARGA DE DATOS ---
st.write("### 🔄 Descargando datos desde Yahoo Finance...")
with st.spinner("Obteniendo información de los activos..."):

    # evitar duplicados en la lista a descargar
    to_download = list(dict.fromkeys(tickers + [market_ticker]))

    raw = yf.download(
        tickers=to_download,
        start=start_date,
        end=end_date,
        group_by="ticker",      # 👈 NECESARIO para obtener 'Close', 'Adj Close', etc.
        auto_adjust=not use_adjusted,
        threads=True
    )

# --- LIMPIEZA Y SELECCIÓN DE COLUMNAS CLOSE / ADJ CLOSE ---

# Caso: datos vacíos
if raw.empty:
    st.error("❌ No se descargaron datos para los tickers o fechas seleccionados.")
    st.stop()

# Si el DataFrame tiene columnas multinivel (cuando hay varios tickers)
if isinstance(raw.columns, pd.MultiIndex):
    level1 = raw.columns.get_level_values(1)
    level1_names = [str(c).lower() for c in level1]

    adj_variants = {"adj close", "adj_close", "adjclose", "adjusted close"}
    close_variants = {"close", "closing price"}

    found = False

    # Buscar precios ajustados
    if use_adjusted and any(v in level1_names for v in adj_variants):
        adj_key = [col for col in level1 if str(col).lower() in adj_variants][0]
        data = raw.xs(adj_key, axis=1, level=1).dropna(how="all")
        found = True

    # Si no existen ajustados → usar Close normal
    elif any(v in level1_names for v in close_variants):
        close_key = [col for col in level1 if str(col).lower() in close_variants][0]
        data = raw.xs(close_key, axis=1, level=1).dropna(how="all")
        found = True

    if not found:
        st.error(f"❌ No se encontraron columnas válidas ('Adj Close' o 'Close'). Columnas detectadas: {set(level1)}")
        st.stop()

# Caso: datos sin MultiIndex (solo un ticker)
else:
    lower_cols = [str(c).lower() for c in raw.columns]

    if use_adjusted and "adj close" in lower_cols:
        data = raw[[col for col in raw.columns if str(col).lower() == "adj close"][0]].to_frame()
    elif "close" in lower_cols:
        data = raw[[col for col in raw.columns if str(col).lower() == "close"][0]].to_frame()
    else:
        st.error("❌ No se encontraron columnas válidas ('Adj Close' o 'Close') en datos sin MultiIndex.")
        st.stop()



# keep only selected tickers (and market)
available = [c for c in data.columns]
missing = [t for t in tickers + [market_ticker] if t not in available]
if missing:
    st.warning(f"Algunos tickers no estaban disponibles y se excluyen: {', '.join(missing)}")

data = data[[c for c in data.columns if c in tickers + [market_ticker]]].dropna(how='all')

st.success("✅ Datos descargados correctamente.")
st.markdown(f"""
**📅 Periodo:** {data.index.min().date()} → {data.index.max().date()}
**📊 Activos cargados:** {", ".join([c for c in data.columns if c in tickers])}
""")

# --- SELECCIÓN DE VARIABLE DEPENDIENTE E INDEPENDIENTES ---
st.subheader("Selección de variables")
colA, colB = st.columns(2)
dep_var = colA.selectbox("Selecciona el activo dependiente (a explicar):", [t for t in tickers if t in data.columns])
indep_vars = colB.multiselect("Selecciona los activos regresores (explicativos):", [t for t in tickers if t != dep_var and t in data.columns], default=[t for t in tickers if t != dep_var and t in data.columns][:2])
if not indep_vars:
    st.warning("Selecciona al menos un activo regresor.")
    st.stop()

# --- PREPARAR DATOS: retornos en % diarios ---
# validar que market_ticker esté en columnas; si no, continuar pero advertir
if market_ticker not in data.columns:
    st.warning(f"Ticker de mercado '{market_ticker}' no está en los datos; algunas funcionalidades (CAPM) no estarán disponibles.")

returns = data[[dep_var] + indep_vars + ([market_ticker] if market_ticker in data.columns else [])].pct_change().dropna() * 100
Y = returns[dep_var]
X = returns[indep_vars]
X = sm.add_constant(X)

# --- AJUSTE DEL MODELO ---
try:
    use_robust_se = st.sidebar.checkbox("Usar errores estándar robustos (HC1)", value=False)
    
    model = sm.OLS(Y, X).fit(
        cov_type='HC1' if use_robust_se else 'nonrobust'
    )

except PerfectSeparationError:
    st.error("Error de perfect separation. Revisa tus datos o variables (posiblemente colinealidad perfecta).")
    st.stop()


# --- Remover regresores NO significativos si se pidió ---
allow_autoremove = st.sidebar.checkbox("Eliminar regresores no significativos", value=False)

if allow_autoremove:

    # 1. Recalcular p-values sin contar la constante
    pvals = model.pvalues.drop('const', errors='ignore')

    # 2. Identificar regresores con p-valor mayor al alpha elegido
    to_remove = pvals[pvals > alpha].index.tolist()

    if to_remove:
        st.info(f"Eliminando automáticamente regresores no significativos: {', '.join(to_remove)}")

        # 3. Actualizar lista de variables independientes
        indep_vars = [v for v in indep_vars if v not in to_remove]

        # 4. Reconstruir matriz X con las nuevas variables
        X = returns[indep_vars].copy()
        X = sm.add_constant(X)

        # 5. Recalcular el modelo
        model = sm.OLS(Y, X).fit(cov_type='HC1' if use_robust_se else None)

    else:
        st.success("No se encontraron regresores no significativos según el nivel de α.")


# --- MOSTRAR RESUMEN ---
st.subheader("📊 Resultados de la regresión (tabla resumida)")
st.write(model.summary())

# --- Interpretación automática ---
st.subheader("🔎 Interpretación automática del modelo")

# --- Nivel de significancia ---
alpha = st.sidebar.selectbox(
    "Nivel de significancia (α)", 
    [0.01, 0.05, 0.10], 
    index=1
)


def interpret_model(m, alpha=0.05):
    lines = []

    # R²
    r2 = m.rsquared
    lines.append(f"R² = {r2:.3f}. ")
    if r2 < 0.1:
        lines.append("R² bajo: el modelo explica poco de la variación de retornos (común en series financieras).")
    elif r2 < 0.4:
        lines.append("R² moderado: el modelo captura parte de la variación del activo.")
    else:
        lines.append("R² alto para retornos: buena capacidad explicativa (verificar sobreajuste).")

    # Alpha
    if 'const' in m.params.index:
        alpha_hat = float(m.params['const'])
        lines.append(f"Alpha estimado = {alpha_hat:.4f} (% diario).")

    # Significancia por variable
    pvals = m.pvalues
    for var in pvals.index:
        if var == 'const':
            continue
        
        coef = float(m.params[var])
        pv = float(pvals[var])

        if coef > 0:
            sign = "positivo"
        elif coef < 0:
            sign = "negativo"
        else:
            sign = "cercano a 0"

        if pv < alpha:
            lines.append(f"{var}: coef={coef:.4f}, p={pv:.3f} → significativo. Relación {sign}.")
        else:
            lines.append(f"{var}: coef={coef:.4f}, p={pv:.3f} → NO significativo. Considerar removerlo.")

    # F-test
    if hasattr(m, 'f_pvalue') and m.f_pvalue is not None:
        if m.f_pvalue < alpha:
            lines.append(f"El F-test p={m.f_pvalue:.3f} indica que al menos un coeficiente es distinto de cero.")
        else:
            lines.append(f"El F-test p={m.f_pvalue:.3f} sugiere que el modelo en conjunto no es significativo.")

    return "\n".join(lines)


# --- ALERTAS POR NO SIGNIFICANCIA ---
insig = model.pvalues.drop('const', errors='ignore')[model.pvalues.drop('const', errors='ignore') > alpha]
if not insig.empty:
    st.warning(f"Regresores no significativos (p > {alpha}): {', '.join(insig.index)}")
else:
    st.success("Todos los regresores son estadísticamente significativos al nivel seleccionado.")

# --- PRUEBAS DIAGNÓSTICAS ---
st.subheader("🧪 Pruebas diagnósticas de residuales")

# Jarque-Bera
jb_stat, jb_pvalue, jb_skew, jb_kurt = jarque_bera(model.resid)
st.write(f"Jarque-Bera: estadístico={jb_stat:.3f}, p-valor={jb_pvalue:.3f} (normalidad) — skew={jb_skew:.3f}, kurt={jb_kurt:.3f}")
if jb_pvalue < 0.05:
    st.warning("Los residuales no siguen una distribución normal (Jarque-Bera p < 0.05). Esto es frecuente en retornos financieros.")
else:
    st.info("No hay evidencia fuerte contra normalidad de los residuales (JB p >= 0.05).")

# Durbin-Watson
dw = durbin_watson(model.resid)
st.write(f"Durbin-Watson: {dw:.3f} (valores cercanos a 2 indican no autocorrelación)")
if dw < 1.5:
    st.warning("Posible autocorrelación positiva en los residuales (DW < 1.5). Considera modelos con términos autoregresivos.")
elif dw > 2.5:
    st.warning("Posible autocorrelación negativa en los residuales (DW > 2.5).")
else:
    st.info("DW en rango aproximado de no autocorrelación (≈2).")

# Breusch-Pagan (heterocedasticidad)
bp_test = het_breuschpagan(model.resid, model.model.exog)
bp_stat, bp_pvalue = bp_test[0], bp_test[1]
st.write(f"Breusch-Pagan: estadístico={bp_stat:.3f}, p-valor={bp_pvalue:.3f} (homocedasticidad)")
if bp_pvalue < 0.05:
    st.warning("Evidencia de heterocedasticidad (Breusch-Pagan p < 0.05). Considera errores robustos o modelar la varianza (GARCH).")
else:
    st.info("No hay evidencia fuerte de heterocedasticidad (BP p >= 0.05).")

# --- MATRIZ DE CORRELACIONES Y GRÁFICOS ---
st.subheader("📉 Correlaciones y scatter plots")
# --- Matriz de correlaciones robusta ---
cols_corr = [dep_var] + indep_vars

# Convertir a lista única preservando orden
cols_corr = list(dict.fromkeys(cols_corr))

# Filtrar dataframe
corr_df = returns[cols_corr].copy()

# Eliminar columnas duplicadas (si quedaron)
corr_df = corr_df.loc[:, ~corr_df.columns.duplicated()]

# Mostrar matriz de correlación estilizada
try:
    st.dataframe(corr_df.corr().style.background_gradient(cmap='coolwarm'))
except:
    # Fallback si todavía hay duplicados o algún problema
    st.dataframe(corr_df.corr())
    st.warning("⚠ No se pudo aplicar estilo por columnas duplicadas. Verifica tus activos.")


# --- Gráficos de dispersión por variable independiente ---
st.subheader("📉 Dispersión de cada regresor vs dependiente")

fig, ax = plt.subplots(figsize=(10, 5))

for col in indep_vars:
    if col == dep_var:
        continue  # evitar graficar la dependiente contra sí misma

    # Seleccionar series
    x = returns[col]
    y = returns[dep_var]

    # Alinear por índice (fechas comunes)
    x, y = x.align(y, join='inner', axis=0)

    # Eliminar NaNs
    x = x.dropna()
    y = y.dropna()

    if x.empty or y.empty:
        st.warning(f"No hay datos válidos para graficar {col} vs {dep_var}.")
        continue

    # Graficar
    ax.scatter(x.values, y.values, alpha=0.5, label=col)

ax.set_xlabel("Retornos de los regresores (%)")
ax.set_ylabel(f"Retornos de {dep_var} (%)")
ax.legend()
ax.grid(True)
st.pyplot(fig)


# --- RESIDUALES ---
st.subheader("🔍 Serie de residuales y distribución")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(model.resid)
ax2.set_title("Residuales del modelo")
ax2.grid(True)
st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.hist(model.resid, bins=30)
ax3.set_title('Histograma de residuales')
st.pyplot(fig3)

# --- ACTUAL VS FITTED ---
st.subheader("📈 Actual vs Fitted")
fitted = model.fittedvalues
fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(Y.index, Y.values, label='Actual', linewidth=1)
ax4.plot(fitted.index, fitted.values, label='Fitted', linewidth=1)
ax4.legend()
ax4.set_title('Actual (retornos) vs Fitted')
ax4.grid(True)
st.pyplot(fig4)

# --- CAPM: calcular beta contra el mercado seleccionado ---
st.subheader("⚖️ Beta CAPM (vs. mercado seleccionado)")
if market_ticker not in returns.columns:
    st.error(f"El ticker de mercado {market_ticker} no está disponible en los datos descargados.")
else:
    # convertir rf anual a tasa diaria aproximada (asumiendo 252 días hábiles)
    rf_daily_pct = (1 + rf_annual/100) ** (1/252) - 1
    rf_daily_pct *= 100
    Ri = returns[dep_var] - rf_daily_pct
    Rm = returns[market_ticker] - rf_daily_pct
    Xcapm = sm.add_constant(Rm)
    capm_model = sm.OLS(Ri, Xcapm).fit()
    beta_capm = capm_model.params[market_ticker]
    alpha_capm = capm_model.params['const']
    st.write(f"Beta CAPM estimada = {beta_capm:.4f}")
    st.write(f"Alpha CAPM (exceso diario) = {alpha_capm:.4e}")
    st.write(capm_model.summary())

# --- MÓDULO DE PREDICCIÓN CON INTERVALOS ---
st.subheader("🔮 Predicción y escenarios (intervalos de confianza)")
st.markdown("Introduce valores hipotéticos (retornos %) para los regresores para obtener una predicción de retorno del dependiente.")

pred_inputs = {}
cols = st.columns(len(indep_vars))
for i, var in enumerate(indep_vars):
    # prefill con última observación
    last = returns[var].iloc[-1]
    pred_inputs[var] = cols[i].number_input(f"{var} (retorno %)", value=float(last), format="%.6f")

if st.button("Calcular predicción"):
    exog = [1.0] + [pred_inputs[v] for v in indep_vars]
    exog_df = pd.DataFrame([exog], columns=['const'] + indep_vars)
    try:
        pred = model.get_prediction(exog_df)
        summary_frame = pred.summary_frame(alpha=0.05)
        yhat = summary_frame['mean'].iloc[0]
        ci_lower = summary_frame['mean_ci_lower'].iloc[0]
        ci_upper = summary_frame['mean_ci_upper'].iloc[0]
        st.write(f"Predicción retorno esperado para {dep_var}: {yhat:.4f} %")
        st.write(f"Intervalo de confianza 95%: [{ci_lower:.4f} %, {ci_upper:.4f} %]")
        st.dataframe(summary_frame.T)
    except Exception as e:
        st.error(f"No se pudo calcular la predicción: {e}")

# --- DESCARGA DE DATOS Y RESULTADOS ---
st.subheader("📥 Descargar datos y resultados")
csv = returns.to_csv().encode('utf-8')
st.download_button("📥 Descargar datos (retornos)", data=csv, file_name='datos_regresion.csv', mime='text/csv')

# Exportar coeficientes y resumen interpretativo
coef_df = pd.DataFrame({'coef': model.params, 'pvalue': model.pvalues})
coef_csv = coef_df.to_csv().encode('utf-8')
st.download_button("📥 Descargar coeficientes (CSV)", data=coef_csv, file_name='coeficientes_regresion.csv', mime='text/csv')

st.markdown("---")
st.info("Sugerencia: revisa las alertas de significancia y las pruebas de heterocedasticidad antes de usar predicciones operativas.")

# --- EXPORTAR REPORTE A PDF ---
def generar_reporte_pdf(summary_text, stats_text, file_path):
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Reporte de Regresión Financiera", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Resumen del Modelo:", styles['Heading2']))
    story.append(Paragraph(summary_text.replace("\n", "<br/>"), styles.get('Code', styles['Normal'])))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Estadísticas Adicionales:", styles['Heading2']))
    story.append(Paragraph(stats_text.replace("\n", "<br/>"), styles.get('Code', styles['Normal'])))
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    doc.build(story)

if st.button("📄 Generar reporte PDF"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        resumen = model.summary().as_text()
        # usar las estadísticas calculadas arriba
        stats_extra = f"DW: {dw:.4f}\\nJB stat: {jb_stat:.4f}, JB p: {jb_pvalue:.4f}\\nBP stat: {bp_stat:.4f}, BP p: {bp_pvalue:.4f}"
        generar_reporte_pdf(resumen, stats_extra, tmp.name)
        with open(tmp.name, "rb") as f:
            st.download_button(
                "⬇️ Descargar Reporte PDF",
                data=f.read(),
                file_name="reporte_regresion.pdf",
                mime="application/pdf"
            )
