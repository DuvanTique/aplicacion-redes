import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "no" 

import torch
from PIL import Image
import cv2
from ultralytics import YOLO
import streamlit as st

# Configuración avanzada de información botánica
PLANT_INFO = {
    "guasca": {
        "nombre_cientifico": "Galinsoga parviflora",
        "nombres_comunes": ["Guasca", "Galinsoga", "Soldadito"],
        "categoria": "Maleza agrícola",
        "biologia": """
        **Biología:**
        - 🌱 Planta anual herbácea
        - 📏 Altura: 20-60 cm
        - 🌼 Flores: Cabezuelas pequeñas con flores blancas y amarillas
        - 🌍 Origen: América tropical
        """,
        "impacto": """
        **Impacto agrícola:**
        - Competencia agresiva con cultivos
        - Reduce rendimientos hasta 40% en infestaciones severas
        - Hospedero de plagas y enfermedades
        """,
        "control": """
        **Métodos de control:**
        - 🚌 Cultural: Mulching y rotación de cultivos
        - ✂️ Mecánico: Escardado manual temprano
        - 🧪 Químico: Herbicidas con flumioxazin
        """,
        "referencias": [
            ("Wikipedia", "https://es.wikipedia.org/wiki/Galinsoga_parviflora"),
            ("Atropocene", "https://antropocene.it/es/2023/03/16/galinsoga-parviflora-3"),
            ("Herbario Nacional Colombiano", "http://www.biovirtual.unal.edu.co/es/")
        ],
        "imagen": "src/assets/97.jpg"
    },
    "violetilla": {
        "nombre_cientifico": "Veronica persica",
        "nombres_comunes": ["Violetilla", "Verónica", "Speedwell azul"],
        "categoria": "Maleza de jardín",
        "biologia": """
        **Biología:**
        - 🌱 Anual o bienal postrada
        - 📏 Longitud tallos: 10-50 cm
        - 🌸 Flores: Azul-violeta con estrías oscuras
        - 🌍 Distribución: Cosmopolita
        """,
        "impacto": """
        **Impacto ecológico:**
        - Coloniza rápidamente suelos alterados
        - Compite con especies nativas
        - Indicadora de suelos compactados
        """,
        "control": """
        **Métodos de control:**
        - 🚜 Mecánico: Eliminación manual con raíz
        - 🌿 Cultural: Mejorar drenaje del suelo
        - 🧫 Biológico: Hongos micorrízicos
        """,
        "referencias": [
            ("Wikipedia", "https://es.wikipedia.org/wiki/Veronica_persica"),
            ("SumitAgro", "https://summit-agro.com/ar/es/2024/08/21/veronica-persica/"),
            ("Herbario Virtual", "https://herbariovirtualbanyeres.blogspot.com/2010/04/veronica-persica.html")
        ],
        "imagen": "src/assets/6.jpg"
    }
}

# Cargar modelo con caché inteligente
@st.cache_resource(ttl=3600, show_spinner="Cargando modelo de IA...")
def load_model():
    return YOLO("src/model/best.pt")

# Configuración de la interfaz
st.set_page_config(
    page_title="AgroAI - Identificación Botánica",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .header-font {
        font-size:32px !important;
        font-weight:bold !important;
        color:#2e7d32 !important;
    }
    .info-expander {
        background-color: #f8f9fa !important;
        border-radius: 10px !important;
    }
    .reference-link {
        color: #1a73e8 !important;
        text-decoration: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Barra lateral
with st.sidebar:
    st.header("Configuración")
    confidence_threshold = st.slider("Umbral de confianza", 0.0, 1.0, 0.5)
    show_details = st.checkbox("Mostrar detalles técnicos", True)

# Cuerpo principal
st.markdown('<p class="header-font">🌱 AgroAI - Identificación Inteligente de Plantas</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Carga una imagen para análisis botánico... Nota: La aplicación esta configurada para detectar flores de la especie galisonga parviflora y veronica pervisca", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with st.spinner("🔍 Analizando imagen..."):
        # Procesamiento de imagen
        image = Image.open(uploaded_file).convert("RGB")
        temp_path = f"temp_{uploaded_file.name}"
        image.save(temp_path)
        model = load_model()
        # Inferencia del modelo
        results = model(temp_path, conf=confidence_threshold)
        plotted_img = results[0].plot()
        result_image = Image.fromarray(plotted_img[..., ::-1])
        
        # Detección de clases
        detected_classes = {}
        for detection in results[0].boxes:
            class_id = int(detection.cls)
            class_name = model.names[class_id]
            if class_name in PLANT_INFO:
                conf = float(detection.conf)
                detected_classes[class_name] = max(conf, detected_classes.get(class_name, 0))

    # Mostrar resultados
    st.success("✅ Análisis completado")
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("📸 Imagen Original")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("🔬 Resultados de Detección")
        st.image(result_image, use_container_width=True)
        
        if detected_classes:
            st.subheader("📚 Información Botánica Detallada")
            for plant, confidence in detected_classes.items():
                info = PLANT_INFO[plant]
                
                with st.expander(f"🌿 **{plant.capitalize()}** ({info['nombre_cientifico']}) - Confianza: {confidence:.1%}", expanded=True):
                    tab1, tab2, tab3, tab4 = st.tabs(["Biología", "Impacto", "Control", "Referencias"])
                    
                    with tab1:
                        st.markdown(info["biologia"])
                        st.image(info["imagen"], width=300)
                        
                    with tab2:
                        st.markdown(info["impacto"])
                        st.metric("Nivel de Riesgo", "Alto" if plant == "guasca" else "Moderado")
                        
                    with tab3:
                        st.markdown(info["control"])
                        
                    with tab4:
                        for ref_name, ref_url in info["referencias"]:
                            st.markdown(f"[{ref_name}]({ref_url})", unsafe_allow_html=True)
        else:
            st.warning("⚠️ No se detectaron especies de interés en la imagen")

    # Descargas
    with st.expander("📁 Opciones de Descarga"):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="Descargar imagen analizada",
                data=cv2.imencode(".jpg", plotted_img)[1].tobytes(),
                file_name=f"resultado_{uploaded_file.name}",
                mime="image/jpeg"
            )
        with col_d2:
            report_content = f"Reporte de análisis - {uploaded_file.name}\n\n"
            for plant, conf in detected_classes.items():
                report_content += f"- {plant} ({conf:.1%} confianza)\n"
            st.download_button(
                label="Descargar reporte",
                data=report_content,
                file_name=f"reporte_{uploaded_file.name.split('.')[0]}.txt",
                mime="text/plain"
            )

else:
    st.info("ℹ️ Carga una imagen de planta para comenzar el análisis")