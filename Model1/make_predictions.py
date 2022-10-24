import tensorflow as tf
import pandas as pd
import numpy as np
import os
import csv
import re

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

pwd = os.getcwd()

PATH = pwd.replace(os.sep, "/")

TEST_PATH = PATH + "/Dataset/nlp-getting-started/test.csv"

MAXLEN = 15
TRUNCATING = "post"
PADDING = "post"
OOV_TOKEN = "<OOV>"
EMBEDDING_DIM = 100


def parse_data(file_path):

    tweets = []

    with open(file_path, encoding="utf-8") as file:
        csv_reader = csv.reader(file, delimiter=",")

        next(csv_reader)

        for row in csv_reader:
            tweet = row[-1]

            tweets.append(tweet)

    return tweets


TWEETS = parse_data(TEST_PATH)


def clean_data(old_tweets):
    clean_tweets = []

    for tweet in old_tweets:
        tweet = tweet.replace("#", "")
        tweet = tweet.replace("\'ve", "")
        tweet = tweet.replace("n\'t", " not")
        tweet = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)
        clean_tweets.append(tweet.lower())

    return clean_tweets


TWEETS = clean_data(TWEETS)


def fit_tokenizer(train_tweets, oov_token):

    tokenizer = Tokenizer(oov_token=oov_token)
    tokenizer.fit_on_texts(train_tweets)

    return tokenizer


TOKENIZER = fit_tokenizer(TWEETS, OOV_TOKEN)
VOCAB_SIZE = len(TOKENIZER.word_index)


def seq_pad_trunc(tweets, padding, truncating, maxlen, tokenizer):

    sequences = tokenizer.texts_to_sequences(tweets)

    padded = pad_sequences(sequences, padding=padding, truncating=truncating, maxlen=maxlen)

    return padded


TWEETS_PADDED = seq_pad_trunc(TWEETS, PADDING, TRUNCATING, MAXLEN, TOKENIZER)


GLOVE_FILE = PATH + "/GloVe/glove.6B/glove.6B.100d.txt"

GLOVE_EMBEDDINGS = {}

with open(GLOVE_FILE, encoding="utf8") as f:
    for line in f:
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
def f1_score(y, y_hat, thresh=0.5):

    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    score = tf.reduce_mean(f1)

    return score


MODEL_PATH = PATH + "/saved_model/tweet_classifier.h5"

MODEL = tf.keras.models.load_model(MODEL_PATH, custom_objects={"f1_score": f1_score})

predictions = MODEL.predict(TWEETS_PADDED)

predictions = [0 if x < 0.5 else 1 for x in predictions]

sample_submission = pd.read_csv(PATH + "/Dataset/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = predictions

sample_submission.to_csv("submission.csv", index=False)
