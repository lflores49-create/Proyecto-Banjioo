import streamlit as st
import pandas as pd
import streamlit as st

# Título de la página
st.title("Bienvenid@s")

# Centrar los botones usando Markdown y el parámetro de alineación
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Botón para iniciar sesión
    if st.button('Iniciar sesión'):
        st.session_state.page = "login"

    # Botón para crear cuenta
    if st.button('Crear cuenta'):
        st.session_state.page = "register"

# Página principal o de login
if 'page' in st.session_state:
    if st.session_state.page == "login":
        # Redirigir a la página de iniciar sesión
        st.write("Aquí iría el formulario para iniciar sesión.")
        # Puedes agregar más contenido para el login aquí
    elif st.session_state.page == "register":
        # Redirigir a la página de crear cuenta
        st.write("Aquí iría el formulario para crear cuenta.")
        # Puedes agregar más contenido para crear cuenta aquí
else:
    st.write("Elige una opción.")
