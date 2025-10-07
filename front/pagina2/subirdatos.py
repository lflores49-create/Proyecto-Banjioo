import streamlit as st
import pandas as pd
import webbrowser

st.set_page_config(page_title="Frontend de Datos", layout="centered")

st.title("Gestión de datos")

# Crear dos columnas para los botones
col1, col2 = st.columns(2)

# --- Botón Subir datos.xls ---
with col1:
    archivo = st.file_uploader("Subir datos.xls", type=["xls", "xlsx"])
    if archivo is not None:
        df = pd.read_excel(archivo)
        st.success("✅ Archivo cargado con éxito")
        st.dataframe(df)

# --- Botón Buscar datos ---
with col2:
    if st.button("Buscar datos"):
        st.markdown("[Ir a Yahoo Finanzas](https://finance.yahoo.com/)", unsafe_allow_html=True)
