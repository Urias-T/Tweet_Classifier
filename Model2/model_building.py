import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import pickle
import os
import re
# import nltk
# nltk.download("stopwords")
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
import csv

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE


pwd = os.getcwd()

PATH = pwd.replace(os.sep, "/")

TRAIN_PATH = PATH + "/Dataset/nlp-getting-started/new_train.csv"

EMBEDDING_DIM = 100
MAXLEN = 15
TRUNCATING = "post"
PADDING = "post"
OOV_TOKEN = "<OOV>"
TEST_SPLIT = 0.2


def parse_data(filepath):

    tweets = []
    labels = []

    with open(filepath, encoding="utf-8") as file:
        csv_reader = csv.reader(file, delimiter=",")

        next(csv_reader)

        for row in csv_reader:
            tweet = row[0]
            label = row[1]

            tweets.append(tweet)
            labels.append(label)

    return tweets, labels


TWEETS, LABELS = parse_data(TRAIN_PATH)


STOPWORDS = set(stopwords.words("english"))

WORDNET_LEMMATIZER = WordNetLemmatizer()


def preprocess_data(texts, stop_words, wordnet_lemmatizer):

    tweets = []

    for tweet in texts:
        tweet = tweet.lower()
        tweet.replace("#", "")
        tweet = re.sub(r'[^\w]', ' ', tweet)
        tweet = word_tokenize(tweet)
        tweet = [wordnet_lemmatizer.lemmatize(t) for t in tweet]
        tweet = [word for word in tweet if word not in stop_words]
        tweet = " ".join(tweet)
        tweets.append(tweet)

    return tweets


TWEETS = preprocess_data(TWEETS, STOPWORDS, WORDNET_LEMMATIZER)


def split_data(tweets, labels, test_split):

    test_size = int(len(tweets) * test_split)

    train_tweets = tweets[: -test_size]
    val_tweets = tweets[-test_size:]

    train_labels = labels[: -test_size]
    val_labels = labels[-test_size:]

    return train_tweets, val_tweets, train_labels, val_labels


TRAIN_TWEETS, VAL_TWEETS, TRAIN_LABELS, VAL_LABELS = split_data(TWEETS, LABELS, TEST_SPLIT)


def fit_tokenizer(tweets, oov_token):

    tokenizer = Tokenizer(oov_token=oov_token)
    tokenizer.fit_on_texts(tweets)

    return tokenizer


TOKENIZER = fit_tokenizer(TWEETS, OOV_TOKEN)
VOCAB_SIZE = len(TOKENIZER.word_index)


def seq_pad_trunc(tweets, padding, maxlen, truncating, tokenizer):

    sequences = tokenizer.texts_to_sequences(tweets)

    padded = pad_sequences(sequences, padding=padding, truncating=truncating, maxlen=maxlen)

    return padded


TRAIN_PADDED = seq_pad_trunc(TRAIN_TWEETS, PADDING, MAXLEN, TRUNCATING, TOKENIZER)
TRAIN_LABELS = np.array(TRAIN_LABELS).astype("float32")

VAL_PADDED = seq_pad_trunc(VAL_TWEETS, PADDING, MAXLEN, TRUNCATING, TOKENIZER)
VAL_LABELS = np.array(VAL_LABELS).astype("float32")

smt = SMOTE()
TRAIN_PADDED, TRAIN_LABELS = smt.fit_resample(TRAIN_PADDED, TRAIN_LABELS)

GLOVE_FILE = PATH + "/GloVe/glove.6B/glove.6B.100d.txt"

GLOVE_EMBEDDINGS = {}

with open(GLOVE_FILE, encoding="utf8") as glove:
    for line in glove:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        GLOVE_EMBEDDINGS[word] = vector

EMBEDDINGS_MATRIX = np.zeros((VOCAB_SIZE+1, EMBEDDING_DIM))

for word, i in TOKENIZER.word_index.items():
    embedding_vector = GLOVE_EMBEDDINGS.get(word)
    if embedding_vector is not None:
        EMBEDDINGS_MATRIX[i] = embedding_vector


@tf.function
def f1_score(y, yhat, thresh=0.5):

    y_pred = tf.cast(tf.greater(yhat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    score = tf.reduce_mean(f1)

    return score


def create_model(maxlen, embeddings_matrix, vocab_size, embedding_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=maxlen, weights=[embeddings_matrix],
                                  trainable=False),
        tf.keras.layers.Conv1D(64, 5, activation="relu"),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.LSTM(64),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Dense(64, activaton="relu"),
        # tf.keras.Layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=[f1_score])

    return model


MODEL = create_model(MAXLEN, EMBEDDINGS_MATRIX, VOCAB_SIZE, EMBEDDING_DIM)

history = MODEL.fit(TRAIN_PADDED, TRAIN_LABELS, epochs=15, validation_data=(VAL_PADDED, VAL_LABELS))

MODEL.save("saved_model/tweet_classifier.h5")

pickle.dump(TOKENIZER, open("tokenizer", "wb"))

f1_score = history.history["f1_score"]
val_f1_score = history.history["val_f1_score"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(f1_score))

plt.plot(epochs, f1_score, "red", label="Training F1 Score")
plt.plot(epochs, val_f1_score, "blue", label="Validation F1 Score")
plt.title("Training and Validation Accuracies")

plt.plot(epochs, loss, "red", ls="--", label="Training Loss")
plt.plot(epochs, val_loss, "blue", ls="--", label="Validation Loss")
plt.title("Training and Validation Losses")

plt.legend()

plt.show()
