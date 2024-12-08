
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Título y descripción
st.title("Predicción de Éxito de Negocios")
st.markdown("""
Este modelo predice el éxito de negocios usando un **RandomForestClassifier** basado en las columnas **rating**, **num_of_reviews**, 
**price_numeric**, **latitude**, y **longitude**.
""")

# Cargar archivo directamente desde la ruta
ruta_archivo = "HawaiiMuestra.csv"

@st.cache_data
def cargar_datos():
    datos = pd.read_csv(ruta_archivo)
    return datos

Hawaii = cargar_datos()

@st.cache_data
def cargar_datos():
    datos = pd.read_csv(ruta_archivo)
    # Mapear la columna 'price' a valores numéricos
    price_mapping = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4, '₩': 1, '₩₩': 2}
    datos['price_numeric'] = datos['price'].map(price_mapping).fillna(0)
    # Crear columna 'success' (éxito si rating >= 4 y num_of_reviews > 50)
    datos['success'] = ((datos['rating'] >= 4) & (datos['num_of_reviews'] > 50)).astype(int)
    return datos

Hawaii = cargar_datos()

# Vista previa de los datos
st.subheader("Vista previa de los datos:")
st.dataframe(Hawaii.head())

# Variables seleccionadas
features = ['rating', 'num_of_reviews', 'price_numeric', 'latitude', 'longitude']
target = 'success'

# Preparar datos
X = Hawaii[features].dropna()
y = Hawaii.loc[X.index, target]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Resultados del modelo
y_pred = model.predict(X_test)
st.subheader("Resultados del Modelo")
st.write(f"**Accuracy del modelo:** {accuracy_score(y_test, y_pred):.2f}")
st.text("Reporte de Clasificación:")
st.text(classification_report(y_test, y_pred))

# Importancia de variables
st.subheader("Importancia de Variables")
feature_importances = pd.Series(model.feature_importances_, index=features)
fig, ax = plt.subplots()
feature_importances.sort_values(ascending=False).plot(kind='bar', ax=ax)
st.pyplot(fig)

# Mapa interactivo con predicciones
st.subheader("Mapa Interactivo de Predicción")
mapa_hawaii = folium.Map(location=[20.7967, -156.3319], zoom_start=7)
marker_cluster = MarkerCluster().add_to(mapa_hawaii)

for _, row in Hawaii.iterrows():
    features_input = pd.DataFrame([[row['rating'], row['num_of_reviews'], row['price_numeric'],
                                    row['latitude'], row['longitude']]], columns=features)
    prediction = model.predict(features_input)[0]
    prob_success = model.predict_proba(features_input)[0][1] * 100

    # Popup para el mapa
    popup_text = f"""
    <b>{row['name_x']}</b><br>
    Dirección: {row['address']}<br>
    Rating: {row['rating']}<br>
    Reseñas: {row['num_of_reviews']}<br>
    Probabilidad de éxito: {prob_success:.2f}%
    """
    color = "green" if prediction == 1 else "red"

    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color=color, icon="info-sign")
    ).add_to(marker_cluster)

# Mostrar el mapa
folium_static(mapa_hawaii)