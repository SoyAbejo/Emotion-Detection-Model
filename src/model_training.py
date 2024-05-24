import sys
import os
import joblib

# Añadir el directorio raíz del proyecto al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from src.data_preprocessing import load_and_split_data
from sklearn.feature_extraction.text import TfidfVectorizer

# Nombres de las emociones
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

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
    return accuracy, report

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()
    model, vectorizer = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, vectorizer, X_test, y_test, emotion_labels)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)
    
    # Guardar el modelo y el vectorizador
    joblib.dump(model, 'model/emotion_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
