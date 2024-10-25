import unittest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from src.data_preprocessing import load_and_split_data
from src.model_training import train_model, evaluate_model

class TestEmotionDetectionModel(unittest.TestCase):
    language = "es"  # Configuración por defecto, cambiar a "en" si necesitas inglés

    # Etiquetas de emociones en español e inglés
    emotion_labels_es = ["tristeza", "alegría", "amor", "ira", "miedo", "sorpresa"]
    emotion_labels_en = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    @classmethod
    def setUpClass(cls):
        # Cargar los datos según el idioma seleccionado
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = load_and_split_data(language=cls.language)
        cls.model, cls.vectorizer = train_model(cls.X_train, cls.y_train)
    
    def test_model_training(self):
        self.assertIsInstance(self.model, MultinomialNB)
        self.assertIsInstance(self.vectorizer, TfidfVectorizer)
    
    def test_model_accuracy(self):
        # Etiquetas de emociones en función del idioma
        emotion_labels = self.emotion_labels_es if self.language == "es" else self.emotion_labels_en
        accuracy, report = evaluate_model(self.model, self.vectorizer, self.X_test, self.y_test, emotion_labels)
        self.assertGreater(accuracy, 0.6)  # Ejemplo: esperar que la precisión sea mayor que 0.6

if __name__ == '__main__':
    # Ejecutar las pruebas unitarias
    unittest.main()
