import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Cargar datos y entrenar modelo (cacheado)
@st.cache_data
def cargar_datos_y_modelo(url):
    # Cargar datos
    Hawaii = pd.read_csv(url)
    price_mapping = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4, '₩': 1, '₩₩': 2}
    Hawaii['price_numeric'] = Hawaii['price'].map(price_mapping).fillna(0)
    Hawaii['success'] = ((Hawaii['rating'] > 4) & (Hawaii['num_of_reviews'] > 50)).astype(int)
    Hawaii = Hawaii.dropna(subset=['latitude', 'longitude'])
    
    # Entrenar modelo
    features = ['rating', 'num_of_reviews', 'price_numeric', 'latitude', 'longitude']
    X = Hawaii[features]
    y = Hawaii['success']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Predicciones
    Hawaii['prediction'] = model.predict(X)
    Hawaii['prob_success'] = model.predict_proba(X)[:, 1] * 100
    
    return Hawaii, model

# Cargar datos
url_csv = "https://raw.githubusercontent.com/PerlaMarGreta/Hawaii/master/HawaiiMuestra.csv"
Hawaii, model = cargar_datos_y_modelo(url_csv)

# Título y descripción
st.title("Predicción de Éxito de Negocios")
st.markdown("""
Esta aplicación predice el éxito de negocios usando un modelo de Machine Learning (**RandomForestClassifier**) 
y genera un mapa interactivo con las predicciones.
""")

# Filtrar negocios con alta probabilidad de éxito
st.subheader("Mapa Interactivo con Predicciones")
Hawaii_filtrado = Hawaii[Hawaii['prob_success'] > 80]  # Optimización: Mostrar negocios exitosos

# Crear el mapa
mapa_hawaii = folium.Map(location=[Hawaii['latitude'].mean(), Hawaii['longitude'].mean()], zoom_start=7)
feature_group = folium.FeatureGroup(name="Predicciones de Éxito")

# Agregar marcadores
for _, row in Hawaii_filtrado.iterrows():
    popup_text = f"""
    <b>{row['name_x']}</b><br>
    Dirección: {row['address']}<br>
    Rating: {row['rating']}<br>
    Reseñas: {row['num_of_reviews']}<br>
    Probabilidad de éxito: {row['prob_success']:.2f}%
    """
    color = "green" if row['prediction'] == 1 else "red"

    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color=color, icon="info-sign")
    ).add_to(feature_group)

feature_group.add_to(mapa_hawaii)
folium.LayerControl().add_to(mapa_hawaii)

# Mostrar el mapa
folium_static(mapa_hawaii)
