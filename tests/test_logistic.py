import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import argparse

# Nombres de las emociones en español e inglés
emotion_labels_es = ["tristeza", "alegría", "amor", "ira", "miedo", "sorpresa"]
emotion_labels_en = ["sadness", "joy", "love", "anger", "fear", "surprise"]

def load_model(language):
    # Cargar el modelo entrenado
    model = joblib.load(f'model/lr_emotion_model_{language}.pkl')
    vectorizer = joblib.load(f'model/lr_vectorizer_{language}.pkl')
    return model, vectorizer

def predict_emotion(text, model, vectorizer, labels):
    # Vectorizar el texto de entrada
    text_tfidf = vectorizer.transform([text])
    
    # Hacer la predicción
    prediction = model.predict(text_tfidf)
    
    # Devolver la etiqueta correspondiente
    return prediction[0]

if __name__ == "__main__":
    # Configurar el argumento de idioma
    parser = argparse.ArgumentParser(description="Predecir la emoción usando un modelo entrenado con Logistic Regression.")
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
    example_text = input("como te sientes\n")

    # Predecir la emoción
    predicted_emotion = predict_emotion(example_text, model, vectorizer, emotion_labels)
    
    # Imprimir el resultado
    print(f'Texto: "{example_text}"')
    print(f'Emoción predicha: {predicted_emotion}')
