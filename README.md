# Emotion Detection Model

Final project for the Building AI course

## Summary

This project aims to develop an AI model capable of detecting the underlying emotion in a user-provided text prompt. The model will classify the text into predefined emotional categories, such as happiness, sadness, anger, surprise, etc. Building AI course project.

## Background

Emotional detection in text is a crucial aspect of natural language processing (NLP) with applications in customer service, mental health analysis, social media monitoring, and more. Detecting emotions can help understand user sentiment and improve interactions between humans and machines.

* Understanding customer emotions can improve service in call centers.
* Identifying emotional distress in social media posts can trigger mental health support.
* Enhancing user experience in AI-driven applications by making them more empathetic.

My motivation for this project stems from the growing importance of emotional intelligence in technology. As AI systems become more integrated into daily life, ensuring they can understand and respond to human emotions is vital.

## How is it used?

The model can be integrated into various platforms:

* **Customer service:** Automated systems can use the model to gauge customer sentiment and adjust responses accordingly.
* **Mental health apps:** Detecting signs of emotional distress in user inputs can trigger supportive measures.
* **Social media monitoring:** Brands can analyze public sentiment about their products or services.

Users will input text prompts into the system, which will then analyze the text and return the detected emotion. This can be used in real-time applications where understanding user sentiment is critical.

**Note:** The current model is designed to work with English text only, as the dataset used contains texts in English.

## Data sources and AI methods

The project will utilize a publicly available dataset containing text labeled with emotions:

* [Emotion Dataset from Hugging Face](https://huggingface.co/datasets/dair-ai/emotion) - This dataset includes text data with corresponding emotion labels. **Licensing Information:** The dataset should be used for educational and research purposes only.

The model will leverage NLP techniques and machine learning algorithms, specifically the Naive Bayes classifier, due to its simplicity and effectiveness in text classification tasks.

* **Text preprocessing:** Tokenization, stop-word removal, and stemming/lemmatization.
* **Feature extraction:** Using techniques like TF-IDF to convert text into numerical features.
* **Modeling:** Training a Naive Bayes classifier to predict emotions based on text input.

## Why Naive Bayes?

### Simplicity and Efficiency
Naive Bayes is known for its simplicity and efficiency, making it a suitable choice for a basic model. It requires less computational power compared to more complex models, which allows for faster training and prediction times.

### Performance in Text Classification
Despite its simplicity, Naive Bayes performs exceptionally well for text classification tasks. It works well with small to medium-sized datasets, providing robust results even when the assumptions of feature independence are not fully met.

### Interpretability
The results and workings of a Naive Bayes model are easier to interpret compared to more complex models like neural networks. This transparency can be beneficial when explaining the model's decisions to stakeholders or end-users.

### Comparison with Other Techniques
- **Hill climbing and Nearest neighbor method:** These techniques are not typically used for text classification problems and are more suitable for optimization tasks and classification based on distance metrics, respectively.
- **Bayes Rule and Naive Bayes classifier:** Both are based on Bayes' theorem, but Naive Bayes simplifies the computation by assuming feature independence, making it more practical for our use case.
- **Linear regression and Logistic regression:** While logistic regression could be a good alternative, it generally requires more computational resources and might not perform significantly better than Naive Bayes for this task.
- **Neural network and Deep learning:** These models can provide higher accuracy but at the cost of increased complexity, longer training times, and higher computational requirements. For a basic, interpretable, and efficient model, Naive Bayes is preferred.

## Model Training and Evaluation

### Steps to Train and Evaluate the Model

1. **Install Dependencies**: Make sure all dependencies are installed.
```bash
   pip install -r requirements.txt
```   

2. **Train and Evaluate the Model**: Run the training script to load data, preprocess it, train the model, and evaluate its performance.
```bash
    python3 src/model_training.py
```

### Steps to Test the Model

1. **Run Unit Tests (Optional)**: To ensure that the model and related functions work correctly, you can run the unit tests. Navigate to the project root directory and use the following command:
```bash
   python3 -m unittest discover -s tests
```   
This command will discover and execute all unit tests located in the tests directory.

2. **Run Personal Tests**: To manually test the trained model on new text inputs, you can use the personal test script. This script will load the saved model and vectorizer, and make predictions on the provided text input. Run the following command:
```bash
    python3 tests/test_model.py
```
By default, the script includes an example text. You can modify the script to test different inputs.

### Results
The model achieved an accuracy of 66.34% on the test set. Below are the detailed performance metrics with the names of the emotions:
```bash
Accuracy: 0.70875
Classification Report:
              precision    recall  f1-score   support

     sadness       0.71      0.94      0.81       946
         joy       0.65      0.98      0.78      1021
        love       1.00      0.11      0.20       296
       anger       0.92      0.47      0.62       427
        fear       0.92      0.36      0.52       397
    surprise       1.00      0.02      0.03       113

    accuracy                           0.71      3200
   macro avg       0.87      0.48      0.49      3200
weighted avg       0.78      0.71      0.66      3200
```

## Challenges

* **Data quality:** Ensuring the dataset used is diverse and representative of different contexts.
* **Ambiguity in text:** Some emotions can be subtle or mixed, making them hard to classify correctly.
* **Ethical considerations:** Ensuring the system is used responsibly, especially in sensitive areas like mental health.

## What next?

To enhance the project:

* **Model improvement:** Incorporate more sophisticated models and larger datasets for better accuracy.
* **Deployment:** Develop APIs or integrate the model into existing platforms.
* **User feedback:** Gather feedback from real users to continually refine the system.

Skills and assistance needed:

* Collaboration with NLP experts to refine the model.
* Input from domain experts in customer service and mental health.
* Development support for creating user-friendly interfaces and APIs.

## Acknowledgments

* [Emotion Dataset from Hugging Face](https://huggingface.co/datasets/dair-ai/emotion)
* Inspiration from various NLP and AI projects shared in the AI community.