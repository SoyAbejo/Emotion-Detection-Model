import unittest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from src.data_preprocessing import load_and_split_data
from src.model_training import train_model, evaluate_model

class TestEmotionDetectionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = load_and_split_data()
        cls.model, cls.vectorizer = train_model(cls.X_train, cls.y_train)
    
    def test_model_training(self):
        self.assertIsInstance(self.model, MultinomialNB)
        self.assertIsInstance(self.vectorizer, TfidfVectorizer)
    
    def test_model_accuracy(self):
        accuracy, report = evaluate_model(self.model, self.vectorizer, self.X_test, self.y_test, ["sadness", "joy", "love", "anger", "fear", "surprise"])
        self.assertGreater(accuracy, 0.6)  # Ejemplo: esperar que la precisión sea mayor que 0.6

if __name__ == '__main__':
    unittest.main()
