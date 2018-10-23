from keras.models import Sequential, Input, Model
from keras.layers import Dense, Flatten, Embedding, Average, Activation, Lambda, Dropout, LSTM, Bidirectional
from keras.initializers import Constant
import numpy as np
import keras.backend as K
from keras import regularizers

def create_baseline_model(embedding_matrix, input_len):
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
        embeddings_initializer=Constant(embedding_matrix), input_length=input_len,
        trainable=False, mask_zero=True))
    model.add(Lambda(lambda x: K.mean(x, axis=1)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def create_rnn_model(embedding_matrix, input_len):
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
        embeddings_initializer=Constant(embedding_matrix), input_length=input_len,
        trainable=False, mask_zero=True))
    model.add(LSTM(64, return_sequences=True, recurrent_dropout=0.5))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def create_bidir_rnn_model(embedding_matrix, input_len):
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
        embeddings_initializer=Constant(embedding_matrix), input_length=input_len,
        trainable=False, mask_zero=True))
    model.add(Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.5)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def create_train_emb_rnn_model(embedding_matrix, input_len):
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], input_length=input_len, mask_zero=True))
    model.add(LSTM(64, return_sequences=True, recurrent_dropout=0.5))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model