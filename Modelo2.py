from sklearn.model_selection import train_test_split
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st  # Importar Streamlit

# Cargar datos
Hawaii = pd.read_csv("HawaiiMuestra.csv")

# Mapear la columna 'price' a valores numéricos
price_mapping = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4, '₩': 1, '₩₩': 2}
Hawaii['price_numeric'] = Hawaii['price'].map(price_mapping).fillna(0)

# Crear variable objetivo (éxito si rating > 4 y num_of_reviews > 50)
Hawaii['success'] = ((Hawaii['rating'] > 4) & (Hawaii['num_of_reviews'] > 50)).astype(int)
st.write("Distribución de la columna 'success':")
st.write(Hawaii['success'].value_counts())

# Variables seleccionadas
features = ['rating', 'num_of_reviews', 'price_numeric', 'latitude', 'longitude']
target = 'success'

# Preparación de datos
X = Hawaii[features].dropna()
y = Hawaii.loc[X.index, target]

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write(f"Tamaño del conjunto de entrenamiento: {X_train.shape}")
st.write(f"Tamaño del conjunto de prueba: {X_test.shape}")

from sklearn.metrics import accuracy_score, classification_report

# Entrenar el modelo Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicción y evaluación
y_pred = model.predict(X_test)
st.write(f"**Accuracy del modelo:** {accuracy_score(y_test, y_pred):.2f}")
st.write("**Reporte de clasificación:**")
st.text(classification_report(y_test, y_pred))

# Importancia de las variables
st.subheader("Importancia de Variables")
feature_importances = pd.Series(model.feature_importances_, index=features)
fig, ax = plt.subplots(figsize=(8, 6))
feature_importances.sort_values(ascending=False).plot(kind='bar', ax=ax)
plt.title("Importancia de Variables")
st.pyplot(fig)  # Reemplaza plt.show()

# Crear mapa interactivo
st.subheader("Mapa Interactivo de Predicción")
mapa_hawaii = folium.Map(location=[20.7967, -156.3319], zoom_start=7)
marker_cluster = MarkerCluster().add_to(mapa_hawaii)

for _, row in Hawaii.iterrows():
    features_input = pd.DataFrame([[row['rating'], row['num_of_reviews'], row['price_numeric'],
                                    row['latitude'], row['longitude']]],
                                  columns=features)
    prediction = model.predict(features_input)[0]
    prob_success = model.predict_proba(features_input)[0][1] * 100

    popup_text = f"""
    <b>{row['name_x']}</b><br>
    Dirección: {row['address']}<br>
    Rating: {row['rating']}<br>
    Reseñas: {row['num_of_reviews']}<br>
    Precio: {row['price']}<br>
    Probabilidad de éxito: {prob_success:.2f}%
    """
    color = "green" if prediction == 1 else "red"

    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color=color, icon="info-sign")
    ).add_to(marker_cluster)

from streamlit_folium import folium_static
folium_static(mapa_hawaii)  # Mostrar el mapa en Streamlit