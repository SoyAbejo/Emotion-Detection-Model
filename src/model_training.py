import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import argparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from src.data_preprocessing import load_and_split_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Ampliar GridSearchCV para optimización de hiperparámetros
    parameters = {
        'alpha': [0.1, 0.5, 1.0, 1.5, 2.0, 5.0],
        'fit_prior': [True, False]
    }
    nb = MultinomialNB()
    clf = GridSearchCV(nb, parameters, cv=5, scoring='accuracy')
    clf.fit(X_train_tfidf, y_train)
    
    model = clf.best_estimator_
    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test, labels):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=labels)
    #PREDICCION
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title("Matriz de confusión")
    plt.show()
    return accuracy, report

if __name__ == "__main__":
    # Configurar el argumento de idioma
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo de detección de emociones con Naive Bayes.")
    parser.add_argument('--language', type=str, default="es", choices=["es", "en"], 
                        help="Idioma del dataset: 'es' para español o 'en' para inglés.")
    args = parser.parse_args()

    # Obtener el idioma desde los argumentos
    language = args.language

    # Mapeo de etiquetas por idioma
    emotion_labels_es = ["tristeza", "alegría", "amor", "ira", "miedo", "sorpresa"]
    emotion_labels_en = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    # Seleccionar las etiquetas correctas
    emotion_labels = emotion_labels_es if language == "es" else emotion_labels_en

    # Cargar y dividir los datos en función del idioma
    X_train, X_test, y_train, y_test = load_and_split_data(language=language)

    # Entrenar el modelo
    model, vectorizer = train_model(X_train, y_train)
    
    # Evaluar el modelo en el conjunto de prueba
    accuracy, report = evaluate_model(model, vectorizer, X_test, y_test, emotion_labels)
    print(f'Precisión: {accuracy}')
    print('Informe de clasificación:')
    print(report)
    
    # Guardar el modelo y el vectorizador con nombres basados en el idioma
    joblib.dump(model, f'model/emotion_model_{language}.pkl')
    joblib.dump(vectorizer, f'model/vectorizer_{language}.pkl')
