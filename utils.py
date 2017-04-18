import cPickle
import numpy as np
from keras.models import load_model
import pandas as pd

def save_pickle(filename,obj):
    with open(filename,'wb') as file:
        cPickle.dump(obj,file,cPickle.HIGHEST_PROTOCOL)
    print "saved",filename


def load_pickle(filename):
    try:
        with open(filename,'rb') as file:
            obj = cPickle.load(file)
    except IOError:
        print filename,"doesnt exist!, returning None instead"
        return None
    return obj


def pad_with_vectors(X,max_words=500):
    """
    :param X: list of sequences of vectors 
    :param max_words: max number of word vectors to retain/pad
    :return: pretty padded numpy array 
    """
    X_ = [0 for i in xrange(len(X))]
    for i in xrange(len(X)): # X[i]  represent a doc
        if len(X[i]) < max_words:
            X_[i] = np.pad(X[i], (( max_words - len(X[i]),0),  # pre-pad with required number of zero vectors on axis 0
                                   (0, 0)  # axis=1
                                   ), mode='constant')
        else:
            X_[i] = X[i][-1*max_words:]

        try:
            assert len(X_[i]) == max_words
        except AssertionError:
            raise AssertionError("len is {} while as max_words is {}".format(len(X[i]), max_words))
            

    return np.array(X_)

def load_w2v_transformer(path="w2v_transformer.pkl"):
    """
    
    :param path: path to trained w2v transformer, see preprocess.py for class definition 
    :return: trained w2v transformer 
    """
    return load_pickle(path)


def get_sentiment_predictor(max_words=500):
    """ Return a predictor function.
    A function is returned to avoid loading model again and again during prediction of a batch."""
    model = load_model('models/one_d_500_100.h5')
    wtv = load_w2v_transformer()

    return (lambda X: model.predict(pad_with_vectors(wtv.transform(X),max_words=max_words)))


def get_imdb_vocab():
	
    idx2word = pd.read_csv("data/imdb/imdb.vocab",header=None).to_dict()[0]
    word2idx = dict([(w,i) for w,i in zip(idx2word.values(),idx2word.keys())])

    return {'word2idx':word2idx,
            'idx2word':idx2word}


def get_imdb_vocab_size():
    return len(get_imdb_vocab()['word2idx'].keys())
