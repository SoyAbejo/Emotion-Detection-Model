import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_preprocessing import load_and_split_data

# Definir la red neuronal
class EmotionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

def train_model(X_train, y_train, input_size, hidden_size, output_size, emotion_mapping, epochs=50):
    # Convertir y_train de etiquetas a índices numéricos según el idioma
    y_train = y_train.map(emotion_mapping).values  # Convertir a numpy array
    y_train = torch.tensor(y_train, dtype=torch.long)  # Convertir a tensor

    # Convertir los datos a tensores de PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)

    # Crear el modelo
    model = EmotionClassifier(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Entrenamiento del modelo con retropropagación
    for epoch in range(epochs):
        outputs = model(X_train)  # Forward pass
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()  # Backward pass
        loss.backward()  # Retropropagación
        optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model

def evaluate_model(model, X_test, y_test, emotion_mapping):
    model.eval()  # Poner el modelo en modo de evaluación
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.map(emotion_mapping).values, dtype=torch.long)  # Convertir y_test a índices numéricos

    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
    
    print(f'Precisión en el conjunto de prueba: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    # Configurar el argumento de idioma
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo de detección de emociones.")
    parser.add_argument('--language', type=str, default="es", choices=["es", "en"], 
                        help="Idioma del dataset: 'es' para español o 'en' para inglés.")
    args = parser.parse_args()

    # Obtener el idioma desde los argumentos
    language = args.language

    # Mapeo de etiquetas según el idioma
    emotion_mapping_es = {
        "tristeza": 0,
        "alegría": 1,
        "amor": 2,
        "ira": 3,
        "miedo": 4,
        "sorpresa": 5
    }

    emotion_mapping_en = {
        "sadness": 0,
        "joy": 1,
        "love": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5
    }

    emotion_mapping = emotion_mapping_es if language == "es" else emotion_mapping_en

    # Cargar y dividir los datos en función del idioma
    X_train, X_test, y_train, y_test = load_and_split_data(language=language)

    # Vectorizar los datos
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()

    # Definir tamaños de entrada y salida para la red
    input_size = X_train_tfidf.shape[1]
    hidden_size = 256  # Tamaño de la capa oculta
    output_size = 6    # Número de emociones

    # Entrenar el modelo
    model = train_model(X_train_tfidf, y_train, input_size, hidden_size, output_size, emotion_mapping, epochs=10)

    # Evaluar el modelo después del entrenamiento
    evaluate_model(model, X_test_tfidf, y_test, emotion_mapping)

    # Guardar el modelo y el vectorizador con nombres basados en el idioma
    torch.save(model, f'model/back_emotion_model_nn_{language}.pth')  # Guardar el modelo de PyTorch
    joblib.dump(vectorizer, f'model/back_vectorizer_nn_{language}.pkl')  # Guardar el vectorizador con joblib
