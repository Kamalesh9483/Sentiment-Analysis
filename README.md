# Sentiment-Analysis
# IMDB Sentiment Analysis using LSTM
This project demonstrates sentiment analysis on movie reviews from the IMDB dataset using a Long Short-Term Memory (LSTM) model. The goal is to classify movie reviews as positive or negative based on the text content.

Steps
1. Data Loading
The IMDB dataset is loaded from tensorflow.keras.datasets.imdb, where each review is represented as a sequence of integers corresponding to words in the review.

2. Text Preprocessing
Convert text to lowercase.

Remove punctuation and special characters using regular expressions.

Tokenize the text into words and remove stopwords.

Apply stemming (with the PorterStemmer from nltk).

3. Tokenization and Padding
The preprocessed text is tokenized using Tokenizer from Keras, which converts text into sequences of integers.

Sequences are padded to a uniform length to ensure consistent input size for the model.

4. Model Building
A Sequential model is built with:

Embedding layer to represent words as vectors.

LSTM layer to capture the temporal dependencies of words in a sentence.

A Dense layer with a sigmoid activation function for binary classification (positive or negative sentiment).

5. Model Training
The model is compiled using the Adam optimizer and binary cross-entropy loss function.

The model is trained for 5 epochs with a batch size of 64, using the training and validation data.

6. Model Evaluation
After training, the model is evaluated on the test data using accuracy and classification report metrics.

Predictions are made on sample test sentences to demonstrate the model's capabilities.

7. Example Predictions
Example sentences are tested to predict whether the review is positive or negative.
