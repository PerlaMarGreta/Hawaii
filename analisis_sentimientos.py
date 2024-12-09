import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from prophet import Prophet

def analizar_sentimientos():
       
        # Título
        st.title("Análisis de Sentimientos en Reseñas a lo Largo del Tiempo")

        # 1. Cargar y procesar datos
        @st.cache_data
        def cargar_datos():
            with st.spinner("Cargando datos..."):
                # Cargar archivo CSV
                df = pd.read_csv("Sintetica.csv")
                
                # Convertir 'time' a formato fecha
                if pd.api.types.is_numeric_dtype(df['time']):
                    df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
                else:
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
                
                # Eliminar fechas inválidas
                df = df.dropna(subset=['time'])
                
                return df

        df = cargar_datos()

        # 2. Clasificación de sentimientos
        @st.cache_data
        def clasificar_sentimientos(data):
            with st.spinner("Analizando sentimientos..."):
                data['sentiment'] = data['text'].apply(
                    lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0
                )
                data['sentiment_label'] = data['sentiment'].apply(
                    lambda x: 'Positivo' if x > 0 else 'Negativo' if x < 0 else 'Neutral'
                )
                return data

        df = clasificar_sentimientos(df)

        # 3. Mostrar datos clasificados
        st.subheader("Datos Clasificados por Sentimiento")
        st.dataframe(df[['time', 'text', 'sentiment_label']].head(10))

        # 4. Visualización de distribución de sentimientos
        st.subheader("Distribución de Sentimientos")
        sentiment_counts = df['sentiment_label'].value_counts()

        fig_pie = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            title="Porcentaje de Reseñas Positivas, Negativas y Neutrales",
            color=sentiment_counts.index,
            color_discrete_map={'Positivo': 'green', 'Negativo': 'red', 'Neutral': 'gray'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # 5. Predicción de evolución de reseñas para todas las categorías
        st.subheader("Predicción de la Evolución de Sentimientos")

        def entrenar_prediccion(df, sentiment_label):
            # Filtrar por sentimiento
            sentiment_df = df[df['sentiment_label'] == sentiment_label]
            grouped = sentiment_df.groupby(sentiment_df['time'].dt.to_period("M")).size().reset_index(name='count')
            grouped['time'] = grouped['time'].dt.to_timestamp()
            grouped.rename(columns={'time': 'ds', 'count': 'y'}, inplace=True)
            
            # Entrenar modelo Prophet
            model = Prophet()
            model.fit(grouped)
            
            # Predicción
            future = model.make_future_dataframe(periods=12, freq='M')
            forecast = model.predict(future)
            
            return forecast, grouped

        # Entrenar modelos para cada sentimiento
        sentiments = ['Positivo', 'Negativo', 'Neutral']
        forecast_results = {}

        for sentiment in sentiments:
            forecast_results[sentiment] = entrenar_prediccion(df, sentiment)

        # Visualizar predicciones
        fig = px.line(title="Predicción de Sentimientos en el Tiempo")

        for sentiment, (forecast, grouped) in forecast_results.items():
            fig.add_scatter(x=grouped['ds'], y=grouped['y'], mode='markers', name=f'{sentiment} - Datos reales')
            fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name=f'{sentiment} - Predicción')

        st.plotly_chart(fig, use_container_width=True)

        # 6. Generar recomendaciones
        st.subheader("Recomendaciones de Estrategia")

        # Contar tendencia de predicciones
        ultimo_mes = forecast_results['Negativo'][0]['yhat'].iloc[-1]
        if ultimo_mes > 10:  # Umbral para reseñas negativas
            st.warning("""
            🚨 **Aumento en reseñas negativas detectado.**  
            Recomendamos:  
            - Analizar los comentarios negativos más frecuentes.
            - Mejorar las áreas críticas que los clientes mencionan repetidamente.  
            - Implementar encuestas de satisfacción para identificar problemas específicos.
            """)
        else:
            st.success("""
            ✅ **Reseñas negativas bajo control.**  
            Continúe monitoreando y manteniendo un buen servicio al cliente.
            """)

        st.markdown("""
        **Conclusión:**
        - La predicción de sentimientos proporciona una visión clara sobre cómo evolucionan las reseñas.
        - Utilice esta información para tomar acciones proactivas y mejorar la satisfacción del cliente.
        """)
