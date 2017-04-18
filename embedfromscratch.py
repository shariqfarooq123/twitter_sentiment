""" This file is not used in project
    Just in case you want to build end-to-end model instead of training word2vec first. (Not recommended)
    """

from keras.layers import Convolution1D, MaxPooling1D, Dense, Flatten, Embedding, Dropout
from loader import imdb_gen
from keras.models import Sequential
from utils import get_imdb_vocab_size
from preprocess import W2VTransformer

vocab_size = get_imdb_vocab_size() # present vocab size is huge. You would need some preprocessing!!
max_words = 500
batch_size = 32
total_docs = 25000
steps = total_docs/batch_size
train_gen = imdb_gen('train',mode='int')
test_gen = imdb_gen('test',mode='int')

# Example model
model = Sequential([
    Embedding(vocab_size,300,input_length=max_words),
    Convolution1D(32,3,activation='relu',padding='same'),
    MaxPooling1D(),
    Flatten(),
    Dropout(0.4),
    Dense(100,activation='relu'),
    Dense(2,activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_gen,steps,epochs=5)