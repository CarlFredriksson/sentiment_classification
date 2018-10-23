import sc_utils
import model_factory
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

INPUT_LENGTH = 100

# Prepare data
X_train, Y_train, X_test, Y_test = sc_utils.load_data()
X_train, Y_train, X_val, Y_val, X_test, Y_test, tokenizer = sc_utils.preprocess_data(X_train, Y_train, X_test, Y_test, INPUT_LENGTH)
embedding_matrix = sc_utils.create_embedding_matrix(tokenizer)

print("X_train.shape: " + str(X_train.shape))
print("Y_train.shape: " + str(Y_train.shape))
print("X_val.shape: " + str(X_val.shape))
print("Y_val.shape: " + str(Y_val.shape))
print("X_test.shape: " + str(X_test.shape))
print("Y_test.shape: " + str(Y_test.shape))
print("embedding_matrix.shape: " + str(embedding_matrix.shape))

# Create model
#model = model_factory.create_baseline_model(embedding_matrix, INPUT_LENGTH)
model = model_factory.create_rnn_model(embedding_matrix, INPUT_LENGTH)
#model = model_factory.create_bidir_rnn_model(embedding_matrix, INPUT_LENGTH)
#model = model_factory.create_train_emb_rnn_model(embedding_matrix, INPUT_LENGTH)
model.summary()

# Train model
model.fit(X_train, Y_train, batch_size=200, epochs=30)

# Evaluate model on validation set
val_loss, val_accuracy = model.evaluate(X_val, Y_val, verbose=0)
print("Accuracy on validation set: " + str(val_accuracy * 100) + "%")

# Evaluate model on test set
test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy on test set: " + str(test_accuracy * 100) + "%")

# Test model on my own texts
reviews = [
    "This movie is bad. I don't like it it all. It's terrible.",
    "I love this movie. I've seen it many times and it's still awesome.",
    "I don't think this movie is as bad as most people say. It's actually pretty good."
    ]
print("Testing model on my own texts:")
print(reviews)
reviews = tokenizer.texts_to_sequences(reviews)
reviews = pad_sequences(reviews, maxlen=INPUT_LENGTH, padding="post")
reviews = np.array(reviews)
pred = model.predict(reviews)
print(pred)
print("The model predicts:")
sentiment_str = "Negative" if pred[0][0] < 0.5 else "Positive"
print(sentiment_str + " on the first text")
sentiment_str = "Negative" if pred[1][0] < 0.5 else "Positive"
print(sentiment_str + " on the second text")
sentiment_str = "Negative" if pred[2][0] < 0.5 else "Positive"
print(sentiment_str + " on the third text")
