import sys
import os
import joblib
import pandas as pd

# Añadir el directorio raíz del proyecto al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Nombres de las emociones
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

def load_model():
    model = joblib.load('model/emotion_model.pkl')
    vectorizer = joblib.load('model/vectorizer.pkl')
    return model, vectorizer

def predict_emotion(text, model, vectorizer):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return emotion_labels[prediction[0]]

if __name__ == "__main__":
    # Cargar el modelo y el vectorizador
    model, vectorizer = load_model()
    
    # Texto de ejemplo para prueba
    example_text = "Hello, how are you?"
    
    # Predecir la emoción
    predicted_emotion = predict_emotion(example_text, model, vectorizer)
    print(f'Text: "{example_text}"')
    print(f'Predicted Emotion: {predicted_emotion}')
