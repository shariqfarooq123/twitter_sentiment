""" utility file for preprocessing"""

from sklearn.feature_extraction.text import TfidfVectorizer
from utils import save_pickle
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.utils import simple_preprocess
import glob
from utils import get_imdb_vocab
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords



def save_imdb_tfidf():
    """
     For comparison purposes only.
     apply tfidf to imdb data and save the resulting dataset.
     Not used in this project.
     
    """
    train_pos_files = glob.glob('data/imdb/train/pos/*')
    train_neg_files = glob.glob('data/imdb/train/neg/*')
    test_pos_files = glob.glob('data/imdb/test/pos/*')
    test_neg_files = glob.glob('data/imdb/test/neg/*')

    vocab = get_imdb_vocab()
    tfidf = TfidfVectorizer(input='filename',stop_words='english',max_df=0.5,vocabulary=vocab,
                            sublinear_tf=True)


    total_train = train_pos_files + train_neg_files
    x_train = tfidf.fit_transform(total_train)
    y_train = np.concatenate((np.ones(len(train_pos_files)),np.zeros(len(train_neg_files))))

    total_test = test_pos_files + test_neg_files
    x_test = tfidf.transform(total_test)
    y_test = np.concatenate((np.ones(len(test_pos_files)),np.zeros(len(test_neg_files))))

    train_data = (x_train, y_train)
    test_data = (x_test, y_test)
    data = {'train':train_data,'test':test_data}

    save_pickle('data/imdb_tfidf.pkl',data)


class SentenceGenerator(object):
    """ get every sentence from every file in the specified path.
        Used for word2vec training"""
    def __init__(self, path_regex):
        self.dirname = path_regex

    def __iter__(self):
        for fname in glob.glob(self.dirname):
            for line in open(fname):
                yield simple_preprocess(line)

class W2VTransformer():
    """
    The transformer class for word2vec training and transformation
    """

    def fit(self):
        sentences = SentenceGenerator("data/imdb/train/*/*")
        word2vec = Word2Vec(sentences)
        self.wv = word2vec.wv
        del word2vec
        return self


    def transform(self,X):
        X_t = []
        for doc in X:
            tokens = simple_preprocess(doc)
            doc_t = []
            for word in tokens:
                if word not in self.wv.vocab:
                    continue
                doc_t.append(self.wv[word])

            X_t.append(doc_t)
        return X_t

    def transform_word(self,word):
        return self.wv[word]

    def build_wv(self):
        return self.wv



def word2int_from_vocab(word,vocab):
    word2idx = vocab['word2idx']
    if not word2idx.has_key(word):
        print "word not in vocab",word
        return
    return word2idx[word]

def doc2vec_int_from_vocab(doc,vocab):
    tokens = simple_preprocess(doc)
    doc_vec_int = []

    for word in tokens:
        if word in stopwords.words('english'):
            # skip the stopwords
            continue
        int_rep = word2int_from_vocab(word,vocab)
        if int_rep is None:
            continue
        doc_vec_int.append(int_rep)
    return np.array(doc_vec_int)

def get_int_representation_from_vocab(X,vocab,max_words=500,):
    """ get integer representation of words in docs of a dataset"""
    if vocab is None:
        raise AttributeError("vocab is None") # For debugging purposes`
    X_ = []
    for doc in X:
        X_.append(doc2vec_int_from_vocab(doc,vocab))

    if max_words!=None:
        X_ = pad_sequences(X_,maxlen=max_words)
        X_ = np.array(X_)

    return X_



# Uncomment following lines to train and save word2vec model
# if __name__ == "__main__":

    #
    # w2v = W2VTransformer().fit()
    # save_pickle("w2v_transformer.pkl",w2v)
