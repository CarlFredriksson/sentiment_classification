import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

def load_data():
    # neg = 0, pos = 1
    X_train, Y_train, X_test, Y_test = [], [], [], []

    # Prepare paths
    data_path = os.path.join(os.getcwd(), "data")
    train_neg_path = os.path.join(data_path, "train", "neg")
    train_pos_path = os.path.join(data_path, "train", "pos")
    test_neg_path = os.path.join(data_path, "test", "neg")
    test_pos_path = os.path.join(data_path, "test", "pos")

    # Load training data
    for filename in os.listdir(train_neg_path):
        with open(os.path.join(train_neg_path, filename), encoding="utf8") as f:
            review = f.readline()
            X_train.append(review)
            Y_train.append(0)
    for filename in os.listdir(train_pos_path):
        with open(os.path.join(train_pos_path, filename), encoding="utf8") as f:
            review = f.readline()
            X_train.append(review)
            Y_train.append(1)
    print("Training data loaded")

    # Load test data
    for filename in os.listdir(test_neg_path):
        with open(os.path.join(test_neg_path, filename), encoding="utf8") as f:
            review = f.readline()
            X_test.append(review)
            Y_test.append(0)
    for filename in os.listdir(test_pos_path):
        with open(os.path.join(test_pos_path, filename), encoding="utf8") as f:
            review = f.readline()
            X_test.append(review)
            Y_test.append(1)
    print("Test data loaded")

    return X_train, Y_train, X_test, Y_test

def preprocess_data(X_train, Y_train, X_test, Y_test, input_len):
    # Load vocabulary
    vocab_path = os.path.join(os.getcwd(), "data", "imdb.vocab")
    with open(vocab_path, encoding="utf8") as f:
        vocab = f.read().splitlines()
    print("Vocabulary length before tokenizing: " + str(len(vocab)))

    # Prepare tokenizer, the out of vocabulary token for GloVe is "unk"
    tokenizer = Tokenizer(oov_token="unk")
    tokenizer.fit_on_texts(vocab)
    vocab_len = len(tokenizer.word_index) + 1
    print("Vocabulary length after tokenizing: " + str(vocab_len))

    # Convert text to sequences of indices
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Pad sequences with zeros so they all have the same length
    X_train = pad_sequences(X_train, maxlen=input_len, padding="post")
    X_test = pad_sequences(X_test, maxlen=input_len, padding="post")

    # Convert training and test data to numpy arrays
    X_train = np.array(X_train, dtype="float32")
    Y_train = np.array(Y_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")
    Y_test = np.array(Y_test, dtype="float32")

    # Shuffle training and test data
    X_train, Y_train = shuffle(X_train, Y_train)
    X_test, Y_test = shuffle(X_test, Y_test)

    # Split test data into validation and test sets
    split_index = int(0.5 * X_test.shape[0])
    X_val = X_test[split_index:]
    Y_val = Y_test[split_index:]
    X_test = X_test[:split_index]
    Y_test = Y_test[:split_index]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, tokenizer

def create_embedding_matrix(tokenizer):
    # Load GloVe embedding vectors
    embedding_path = os.path.join(os.getcwd(), "data", "glove.6B", "glove.6B.100d.txt")
    word_to_embedding = {}
    with open(embedding_path, encoding="utf8") as f:
        for line in f.readlines():
            values = line.split()
            word = values[0]
            embedding_vec = np.asarray(values[1:], dtype="float32")
            word_to_embedding[word] = embedding_vec
    print("Embedding vectors loaded")

    # Create embedding matrix
    embedding_vec_dim = 100
    vocab_len = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_len, embedding_vec_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vec = word_to_embedding.get(word)
        if embedding_vec is not None:
            embedding_matrix[i] = embedding_vec
    print("Embedding matrix created")

    return embedding_matrix
