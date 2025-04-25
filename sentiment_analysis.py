import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.utils import class_weight
import numpy as np

# Load IMDb dataset from Keras
def load_imdb_data():
    # Load the IMDb dataset from Keras
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)

    # Pad sequences to the same length
    max_len = 100
    X_train = pad_sequences(train_data, maxlen=max_len, padding='post', truncating='post')
    X_test = pad_sequences(test_data, maxlen=max_len, padding='post', truncating='post')

    # Prepare labels
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    
    return X_train, y_train, X_test, y_test

# Load data
X_train, y_train, X_test, y_test = load_imdb_data()

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Build LSTM Model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=64, input_length=100))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))  # Added Dropout
model.add(LSTM(64))
model.add(Dropout(0.5))  # Added Dropout
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32, class_weight=class_weights)

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

# Prediction function
def predict_sentiment(text):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts([text])
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=100, padding='post')
    pred = model.predict(pad)[0][0]
    return "Positive" if pred > 0.5 else "Negative"

# Test examples
print(predict_sentiment("I loved the movie! It was amazing"))
print(predict_sentiment("It was the worst movie ever. So boring"))
