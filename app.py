import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------------------
# Cargar modelo
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("modelo_perros_stanford.keras")

model = load_model()

# -------------------------------
# Cargar labels desde txt
# -------------------------------
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# -------------------------------
# Interfaz
# -------------------------------
st.title("🐶 Clasificador de Razas de Perros")

uploaded_file = st.file_uploader(
    "Sube una imagen de un perro",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Procesamiento
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Mostrar imagen
    st.image(image, caption="Imagen cargada", use_container_width=True)

    # Preprocesamiento (IMPORTANTE: 128x128 como tu modelo)
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    predictions = model.predict(img_array)[0]

    # Obtener mejor predicción
    max_index = np.argmax(predictions)
    best_label = labels[max_index]
    best_prob = predictions[max_index]

    # -------------------------------
    # Resultados
    # -------------------------------
    st.subheader("📊 Probabilidades por raza")

    # Mostrar TODAS las probabilidades
    for i, prob in enumerate(predictions):
        st.write(f"{labels[i]}: {prob:.2%}")

    # Gráfica (BONUS)
    st.bar_chart(predictions)

    st.divider()

    # Mejor resultado
    st.success(f"🏆 Raza más probable: {best_label} ({best_prob:.2%})")