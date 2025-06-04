import os
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Ensure the stopwords corpus is available without forcing a download
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Archivos para almacenar los datasets traducidos y originales
TRANSLATED_DATA_FILE_ES = "translated_emotion_dataset_es.csv"
ORIGINAL_DATA_FILE_EN = "emotion_dataset_en.csv"

def translate_texts(texts, model_name="Helsinki-NLP/opus-mt-en-es"):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    translated_texts = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_texts.append(translated_text)
    return translated_texts

def preprocess_data(df, language):
    stop_words = set(stopwords.words(language))
    stemmer = SnowballStemmer(language)

    def clean_text(text):
        tokens = text.split()
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    df['text'] = df['text'].apply(clean_text)
    return df

def load_and_split_data(language="es"):
    # Verificar si queremos los datos en español o en inglés
    if language == "es":
        data_file = TRANSLATED_DATA_FILE_ES
    else:
        data_file = ORIGINAL_DATA_FILE_EN

    # Cargar el dataset adecuado según el idioma
    if os.path.exists(data_file):
        print(f"Cargando datos desde el archivo {data_file}...")
        df = pd.read_csv(data_file)
    else:
        # Cargar el dataset en inglés desde la fuente original
        try:
            dataset = load_dataset("dair-ai/emotion")
            df = pd.DataFrame(dataset["train"])
        except Exception as e:
            raise RuntimeError(
                "Failed to download dataset. Provide it locally or ensure internet access."
            ) from e

        if language == "es":
            print("Traduciendo el dataset al español...")
            df['text'] = translate_texts(df['text'])

        # Mapeo de etiquetas emocionales numéricas a texto según el idioma
        emotion_mapping_es = {
            0: "tristeza",
            1: "alegría",
            2: "amor",
            3: "ira",
            4: "miedo",
            5: "sorpresa"
        }
        emotion_mapping_en = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }
        emotion_mapping = emotion_mapping_es if language == "es" else emotion_mapping_en

        df['label'] = df['label'].map(emotion_mapping)

        # Verificar si el mapeo dejó valores NaN
        print("Etiquetas después del mapeo:", df['label'].unique())
        print("Número de valores NaN en etiquetas:", df['label'].isnull().sum())

        # Eliminar filas con valores NaN en las etiquetas
        df = df.dropna(subset=['label'])

        if df.empty:
            print("Error: El dataset está vacío después de eliminar los NaN")
            return None, None, None, None

        # Guardar los datos traducidos o en inglés original
        print(f"Guardando los datos en {data_file}...")
        df.to_csv(data_file, index=False)

    # Preprocesar los datos traducidos o en inglés
    df = preprocess_data(df, 'spanish' if language == 'es' else 'english')
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
