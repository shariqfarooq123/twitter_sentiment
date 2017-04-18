from loader import imdb_gen
from keras.layers import Convolution1D, MaxPooling1D, Dense,Flatten,Dropout,LSTM
from keras.models import Sequential
from preprocess import W2VTransformer
from keras.regularizers import l2

max_words = 500  # maximum word length of a review to keep to keep
n_features = 100
total_docs = 25000
batch_size = 32

train_gen = imdb_gen(data='train',max_words=max_words,batch_size=batch_size)
test_gen = imdb_gen(data='test',max_words=max_words,batch_size=batch_size)

model = Sequential([
    Convolution1D(32,3,activation='relu',input_shape=(max_words,n_features),padding='same',
                  kernel_regularizer=l2()),
    MaxPooling1D(),
    Convolution1D(32,3,activation='relu',padding='same', kernel_regularizer=l2()),
    MaxPooling1D(),
    Flatten(),
    #LSTM(40),  # use LSTM if you think model would get better; Warning! They are hard to train!!
    Dropout(0.2), # Dropout of 20% to avoid overfitting
    Dense(512,activation='relu'),
    Dropout(0.2),
    Dense(512,activation='relu'),
    Dense(2,activation='softmax')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

steps_per_epoch = total_docs/batch_size
model.fit_generator(train_gen,steps_per_epoch=steps_per_epoch,epochs=5)
print "Evaluating..."
print "This may take a while"
print model.evaluate_generator(test_gen,100)

model.save('one_d_500_100.h5')




