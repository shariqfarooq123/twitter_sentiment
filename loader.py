from utils import load_pickle
import glob
import numpy as np
from sklearn.utils import shuffle
from preprocess import W2VTransformer
from utils import pad_with_vectors
from keras.utils.np_utils import to_categorical
from preprocess import get_int_representation_from_vocab
from utils import get_imdb_vocab

def load_imdb_tfidf():
    data = load_pickle('data/imdb_tfidf.pkl')
    return data

def load_w2v_transformer(path="w2v_transformer.pkl"):
    return load_pickle(path)

def _file_gen(path):
    for filename in glob.glob(path):
        yield filename

def _imdb_filename_labeller_gen(parent_path,batch_size=32):
    files_pos = list(_file_gen(parent_path+"/pos/*"))
    files_neg = list(_file_gen(parent_path+"/neg/*"))
    # test_files_pos = list(file_gen("data/imdb/test/pos"))
    # test_files_neg = list(file_gen("data/imdb/test/neg"))

    total_files = np.array(files_pos + files_neg)
    labels = np.concatenate((np.ones(len(files_pos)),np.zeros(len(files_neg))))

    inds = shuffle(np.arange(len(labels)))
    total_files = total_files[inds]
    labels = labels[inds]
    total_size = len(labels)
    n_batches = total_size/batch_size
    while(1):
        for i in xrange(n_batches):
            low = i*batch_size
            up = (i+1)*batch_size
            if(up<total_size-total_size%batch_size - batch_size):
                yield (total_files[low:up],labels[low:up])
            else:
                yield (total_files[up:],labels[up:])

def imdb_gen(data='train',batch_size=32,max_words=500,mode='w2v'):
    gen = _imdb_filename_labeller_gen("data/imdb/"+data,batch_size)
    w2v = load_w2v_transformer()
    vocab = get_imdb_vocab()
    while(1):
        x_batch_fnames, y_batch = gen.next()
        x = []


        for fname in x_batch_fnames:
            with open(fname) as file:
                content = file.read()
                x.append(content)

        if mode is 'w2v':
            x_batch = w2v.transform(x)
            if max_words is not None:
                x_batch = pad_with_vectors(x_batch,max_words)
        elif mode is 'int':
            x_batch = get_int_representation_from_vocab(x,max_words=max_words,vocab=vocab)
        else:
            raise AttributeError("attribute mode should be one of 'w2v' or 'int'.")

        y_batch = to_categorical(y_batch,2)
        yield x_batch,y_batch


