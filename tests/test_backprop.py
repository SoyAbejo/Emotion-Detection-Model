import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import argparse
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from src.backpropagation_training import EmotionClassifier

# Nombres de las emociones en español e inglés
emotion_labels_es = ["tristeza", "alegría", "amor", "ira", "miedo", "sorpresa"]
emotion_labels_en = ["sadness", "joy", "love", "anger", "fear", "surprise"]

def load_model(language):
    # Cargar el modelo entrenado en PyTorch
    model = torch.load(f'model/back_emotion_model_nn_{language}.pth')
    model.eval()  # Colocar el modelo en modo de evaluación
    vectorizer = joblib.load(f'model/back_vectorizer_nn_{language}.pkl')  # Cargar el vectorizador
    return model, vectorizer

def predict_emotion(text, model, vectorizer, labels):
    # Vectorizar el texto
    text_tfidf = vectorizer.transform([text]).toarray()
    text_tensor = torch.tensor(text_tfidf, dtype=torch.float32)

    # Hacer la predicción
    with torch.no_grad():
        output = model(text_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return labels[prediction]

if __name__ == "__main__":
    # Configurar el argumento de idioma
    parser = argparse.ArgumentParser(description="Predecir la emoción usando un modelo entrenado con backpropagation.")
    parser.add_argument('--language', type=str, default="es", choices=["es", "en"], 
                        help="Idioma del dataset: 'es' para español o 'en' para inglés.")
    args = parser.parse_args()

    # Obtener el idioma desde los argumentos
    language = args.language

    # Seleccionar las etiquetas correspondientes al idioma
    emotion_labels = emotion_labels_es if language == "es" else emotion_labels_en

    # Cargar el modelo y el vectorizador en función del idioma
    model, vectorizer = load_model(language)

    # Texto de ejemplo para prueba
    example_text = "Te quiero" if language == "es" else "I love you"

    # Predecir la emoción
    predicted_emotion = predict_emotion(example_text, model, vectorizer, emotion_labels)
    print(f'Texto: "{example_text}"')
    print(f'Emoción predicha: {predicted_emotion}')
