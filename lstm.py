# from loader import imdb_gen
# from keras.layers import Convolution1D, MaxPooling1D, Dense,Flatten,Dropout,LSTM
# from keras.models import Sequential
# from preprocess import W2VTransformer
#
# max_words = None
# n_features = 100
# total_docs = 25000
# batch_size = 32
#
# train_gen = imdb_gen(data='train',max_words=max_words,batch_size=batch_size)
# test_gen = imdb_gen(data='test',max_words=max_words,batch_size=batch_size)
#
# model = Sequential([
#     Convolution1D(64,3,activation='relu',input_shape=(max_words,n_features),padding='same'),
#     MaxPooling1D(),
#     Flatten(),
#     #LSTM(40),
#     Dropout(0.4),
#     Dense(250,activation='sigmoid'),
#     Dense(2,activation='softmax')
# ])
#
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#
# steps_per_epoch = total_docs/batch_size
# model.fit_generator(train_gen,steps_per_epoch=steps_per_epoch,epochs=20)
# print "Evaluating..."
# print "This may take a while"
# print model.evaluate_generator(test_gen,100)
#
# model.save('lstm.h5')
#
#
#
#
#
#
#
#
#
#
#
#
