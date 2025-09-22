import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from src.data_preprocessing import load_and_split_data
from sklearn.feature_extraction.text import TfidfVectorizer


def train_lr_model(X_train, y_train):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Unigramas y bigramas
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Parámetros para optimización con GridSearch
    parameters = {
        'C': [0.1, 1, 10],  # valores razonables para C
        'solver': ['lbfgs'],  # solvers estables para multiclase
        'max_iter': [5000]  # más iteraciones para converger
    }

    lr = LogisticRegression(class_weight='balanced')
    clf = GridSearchCV(
        lr, 
        parameters, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1,   # usa todos los núcleos
        verbose=1    # muestra el progreso
    )
    clf.fit(X_train_tfidf, y_train)
    
    model = clf.best_estimator_
    return model, vectorizer


def evaluate_model(model, vectorizer, X_test, y_test, labels):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=labels)
    return accuracy, report


if __name__ == "__main__":
    # Configurar el argumento de idioma
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo de detección de emociones con Logistic Regression.")
    parser.add_argument('--language', type=str, default="es", choices=["es", "en"], 
                        help="Idioma del dataset: 'es' para español o 'en' para inglés.")
    args = parser.parse_args()

    # Obtener el idioma desde los argumentos
    language = args.language

    # Mapeo de etiquetas según el idioma
    emotion_labels_es = ["tristeza", "alegría", "amor", "ira", "miedo", "sorpresa"]
    emotion_labels_en = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    # Seleccionar el conjunto de etiquetas correcto
    emotion_labels = emotion_labels_es if language == "es" else emotion_labels_en

    # Cargar y dividir los datos en función del idioma
    X_train, X_test, y_train, y_test = load_and_split_data(language=language)

    # Entrenar el modelo con Logistic Regression
    model, vectorizer = train_lr_model(X_train, y_train)
    
    # Evaluar el modelo en el conjunto de prueba
    accuracy, report = evaluate_model(model, vectorizer, X_test, y_test, emotion_labels)
    print(f'Precisión: {accuracy}')
    print('Informe de clasificación:')
    print(report)
    
    # Guardar el modelo y el vectorizador con nombres basados en el idioma
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, f'model/lr_emotion_model_{language}.pkl')
    joblib.dump(vectorizer, f'model/lr_vectorizer_{language}.pkl')
