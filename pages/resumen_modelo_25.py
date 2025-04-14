import nltk
nltk.download('punkt')
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from pages.entrenamiento_modelo_25 import limpiar_y_filtrar

def resumir_noticia(noticia, modelo, tokenizer, umbral, max_oraciones, max_len=50):
    # Dividir en oraciones y limpiar
    oraciones = [s.strip() for s in nltk.sent_tokenize(noticia, language='spanish') if len(s.strip()) > 0]
    
    # Truncar si hay demasiadas oraciones
    if len(oraciones) > max_oraciones:
        oraciones = oraciones[:max_oraciones]
    
    # Preprocesar oraciones
    X_filtrado = [limpiar_y_filtrar(oracion) for oracion in oraciones]
    
    # Tokenizar y rellenar
    secuencias = tokenizer.texts_to_sequences(oraciones)
    padded = pad_sequences(secuencias, maxlen=max_len, padding='post', truncating='post')
    
    # Predecir con batch_size
    predicciones = modelo.predict(padded, batch_size=32, verbose=0)
    
    # Seleccionar oraciones relevantes
    resumen = [oraciones[i] for i in range(len(oraciones)) if predicciones[i] >= umbral]
    
    return resumen, X_filtrado

