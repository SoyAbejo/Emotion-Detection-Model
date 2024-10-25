# Emotion Detection Model

Final project for the Building AI course

## Summary

This project develops an AI model capable of detecting underlying emotions in user-provided text. The model classifies text into predefined emotional categories such as happiness, sadness, anger, surprise, etc., and can be trained in either Spanish or English using various machine learning models.

## Background

Emotion detection in text is essential in natural language processing (NLP), with applications in customer service, mental health analysis, social media monitoring, and more. This project explores the importance of emotional intelligence in AI, aiming to enhance human-machine interactions.

- **Customer Service**: Automated systems adjust responses based on detected customer emotions.
- **Mental Health Apps**: Detecting signs of distress in user input can trigger supportive actions.
- **Social Media Monitoring**: Brands can analyze public sentiment about their products or services.

The motivation for this project is the increasing integration of AI in everyday life and the need for AI to understand and respond to human emotions.

## Usage

The model can be integrated into various platforms and applications such as customer service, mental health apps, and social media analysis. The user inputs text, and the system analyzes it to return the detected emotion in real-time.

**New Feature**: The model now supports both Spanish and English. The original English dataset has been translated into Spanish using `Helsinki-NLP` with `MarianTokenizer` and `MarianMTModel`.

## Data Sources and AI Methods

This project uses the following emotion-labeled dataset:

- [Emotion Dataset from Hugging Face](https://huggingface.co/datasets/dair-ai/emotion): Contains text data with emotion labels, intended for educational and research use.

We train models with Naive Bayes, Logistic Regression, and Backpropagation, using these NLP and machine learning techniques:

- **Preprocessing**: Tokenization, stop-word removal, and stemming.
- **Feature Extraction**: TF-IDF for converting text into numerical features.
- **Models**:
  - **Naive Bayes**: A simple and efficient model for text classification.
  - **Logistic Regression**: Powerful for classification tasks with high accuracy.
  - **Neural Network with Backpropagation**: Although less accurate in this case, included to explore potential future improvements.

## Model Comparison

Below is the performance of each model in both languages:

### Naive Bayes
- **English**: 78.96% accuracy, with strong results for "sadness" and "fear."
- **Spanish**: 71.53% accuracy, with challenges in predicting "miedo" (fear).

### Backpropagation
- **English and Spanish**: 29.56% accuracy, indicating this model is suboptimal for the task.

### Logistic Regression
- **English**: 87.68% accuracy, showing the best performance overall.
- **Spanish**: 79.09% accuracy, with good balance across all classes.

Logistic Regression in English is recommended as the primary model, followed by Logistic Regression in Spanish for bilingual applications.

## Model Training and Evaluation

### Training and Evaluation Instructions

1. **Install Dependencies**: Ensure all dependencies are installed.
    ```bash
    pip install -r requirements.txt
    ```

2. **Train and Evaluate the Model**: Run the training script for your chosen language (en or es).
    ```bash
    python src/logistic_regression_training.py --language en
    ```

3. **Run Unit Tests**: Optional, to verify model functions.
    ```bash
    python -m unittest discover -s tests
    ```

4. **Run Personal Tests**: To manually test the trained model on new text inputs, you can use the test script for your chosen language (en or es). This script will load the saved model and vectorizer, and make predictions on the provided text input. Run the following command:
```bash
    python3 tests/test_logistic.py --language en
```
By default, the script includes an example text. You can modify the script to test different inputs.

## Analysis of Results

1. **Overall Accuracy**:
   - Logistic Regression in English provides the best accuracy (87.68%) and balanced performance across all classes.
   
2. **Performance by Class**:
   - **Sadness and Joy**: High recall and precision values, indicating strong performance.
   - **Surprise**: Lower accuracy due to fewer examples in the dataset.

3. **Recommendations for Improvement**:
   - **Data Balancing**: Use techniques like SMOTE for underrepresented classes.
   - **Advanced Models**: Consider models like LSTMs or transformers to enhance performance.

## Challenges

- **Data Quality**: Ensuring the dataset is diverse and representative.
- **Text Ambiguity**: Emotions can be subtle or mixed, making them hard to classify.
- **Ethics**: Responsible use is essential, especially in areas like mental health.

## Future Work

- **Model Improvement**: Incorporate more sophisticated models and larger datasets for better accuracy.
- **Deployment**: Develop APIs or integrate the model into existing platforms.
- **User Feedback**: Collect real-user feedback to improve the system further.

## Acknowledgments

- [Emotion Dataset from Hugging Face](https://huggingface.co/datasets/dair-ai/emotion)
- Inspiration from various NLP and AI projects shared in the AI community.
