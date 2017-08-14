from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
import wordEmbedder as we
import pickle
from pathlib import Path
import numpy as np
import collections
import gensim
import random
import sys
import re
import pickle

from keras.datasets import imdb as imdb
from keras.preprocessing import sequence

def get_input_json(filenames,w2v=None,token=None,feed=None):

    if w2v==None:
        # Load Google's pre-trained Word2Vec model.
        model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  
        w2v   = dict(zip(model.index2word, model.syn0))
    
    #Initialize preprocessor
    preprocessor = we.NLTKPreprocessor()
    vectorizer   = we.TfidfEmbeddingVectorizer(w2v)

    X_tokenized  = preprocessor.transform(filenames)
    vectorizer.fit(X_tokenized)
    X_w2v = vectorizer.transform(X_tokenized)

    data = collections.OrderedDict()
    for i in range(len(filenames)):
        review = collections.OrderedDict()
        for j in range(len(X_tokenized[i])):
            review[X_tokenized[i][j]] = X_w2v[i][j].tolist()
        data[filenames[i]] = review

    if token!=None:
        data_feed = collections.OrderedDict()
        for i in range(len(filenames)):
            data_feed[filenames[i]] = feed[i]

        return data,data_feed
    
    return data



def pad_sequences(trainX, validX, testX, maxlen=None, dtype='int32', padding='post', truncating='post', value=0):

    lengthsTr = np.max([len(s) for s in trainX])
    lengthsVd = np.max([len(s) for s in validX])
    lengthsTe = np.max([len(s) for s in testX])

    if maxlen is None:
            maxlen = np.max(np.array([lengthsTr,lengthsVd,lengthsTe]))

    
    if isinstance(trainX[0][0],np.ndarray):
        elem_length = len(trainX[0][0])
        pads = np.zeros((elem_length,))
    else:
        pads = value

    for s in trainX:
       if len(s) < maxlen:
          while not len(s)==maxlen:
                s.append(pads)
    for s in validX:
       if len(s) < maxlen:
          while not len(s)==maxlen:
                s.append(pads)
    
    for s in testX:
       if len(s) < maxlen:
          while not len(s)==maxlen:
                s.append(pads)           
    
    return trainX,validX,testX,maxlen


"""
def remove_unk(x,n_words):
    return [[1 if w >= n_words else w for w in sen] for sen in x]  
"""


def extract_features_w2v(filenames,seed,test_size=0.05,save_test=None):
    
    random.shuffle(filenames)

    X_train, X_valid, y_train, y_valid = train_test_split(filenames, np.zeros(len(filenames)),test_size=2*test_size,random_state=seed)
    filenames_train = X_train
    filenames_valid = X_valid

    X_valid, X_test, y_valid, y_test = train_test_split(filenames_valid, np.zeros(len(filenames_valid)),test_size=0.5,random_state=seed)
    filenames_valid = X_valid
    filenames_test  = X_test

    
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  
    w2v   = dict(zip(model.index2word, model.syn0))
    
    #Initialize preprocessor
    preprocessor = we.NLTKPreprocessor()
    vectorizer   = we.TfidfEmbeddingVectorizer(w2v)

    #Tokenization - train_X_tokenized is a list containing lists of words for each review in the passed argument
    train_X_tokenized = preprocessor.transform(filenames_train)
    valid_X_tokenized = preprocessor.transform(filenames_valid)
    test_X_tokenized  = preprocessor.transform(filenames_test)
    
    #Get TF-IDF weighting terms
    vectorizer.fit(train_X_tokenized)
    vectorizer.fit(valid_X_tokenized)
    vectorizer.fit(test_X_tokenized)
    
    #Get Word2Vec representation of document weighted with TF-IDF
    train_X_w2v = vectorizer.transform(train_X_tokenized)
    valid_X_w2v = vectorizer.transform(valid_X_tokenized)
    test_X_w2v  = vectorizer.transform(test_X_tokenized)

    test_dict = None
    if save_test!=None:
        test_dict = get_input_json(filenames_test,w2v)

    del w2v
    del model
    
    return train_X_w2v,valid_X_w2v,test_X_w2v,filenames_train,filenames_valid,filenames_test,test_dict


def tokenize_and_remove_unk(X,n_words,dictionary):
#Transform list of lists of tokenized documents to list of lists of tokens, each token being the index of the word in the dictionary. Tokens appear in the same order with corresponding words in documents.
    
    X_tokenized = []
    for doc in X:
        toks = []
        for w in doc:
            if w in list(dictionary.keys()):
                if dictionary[w]<n_words:
                    toks.append(dictionary[w])
                else:
                    toks.append(1)
            else:
                    toks.append(1)
        X_tokenized.append(toks)

    return X_tokenized




def extract_features(filenames_train_valid,filenames_test,seed,test_size,save_test,n_words,dictionary,embedding_dim):
    
    random.shuffle(filenames_train_valid)
    random.shuffle(filenames_test)

    X_train, X_valid, y_train, y_valid = train_test_split(filenames_train_valid, np.zeros(len(filenames_train_valid)),test_size=0.1,random_state=seed)
    filenames_train = X_train
    filenames_valid = X_valid

    """
    X_valid, X_test, y_valid, y_test = train_test_split(filenames_valid, np.zeros(len(filenames_valid)),test_size=0.5,random_state=seed)
    filenames_valid = X_valid
    filenames_test  = X_test
    """
    embedding_initMat = None    
    #embedding_initMat =  get_initial_embeddings_from_dictionary(n_words,embedding_dim,dictionary)
    
    testX = []
    for i in filenames_test:
        testX.append(Path(i).read_text())
    
    trainX = []
    for i in filenames_train:
        trainX.append(Path(i).read_text())
    
    validX = []
    for i in filenames_valid:
        validX.append(Path(i).read_text())

    """
    vectorizer = we.CountVectorizer(vocabulary=d)
    testX_tokenized = vectorizer.transform(testX)
    del testX
    testX = remove_unk(testX_tokenized,n_words)
    trainX_tokenized = vectorizer.transform(trainX)
    del trainX
    trainX = remove_unk(trainX_tokenized,n_words)
    validX_tokenized = vectorizer.transform(validX)
    del validX
    validX = remove_unk(validX_tokenized,n_words)
    """
    preprocessor = we.NLTKPreprocessor()
    #Tokenization - train_X_tokenized is a list containing lists of words for each review in the passed argument
    train_X_tokenized = preprocessor.transform(filenames_train)
    valid_X_tokenized = preprocessor.transform(filenames_valid)
    test_X_tokenized  = preprocessor.transform(filenames_test)
    trainX = tokenize_and_remove_unk(train_X_tokenized,n_words,dictionary)
    testX = tokenize_and_remove_unk(test_X_tokenized,n_words,dictionary)
    validX = tokenize_and_remove_unk(valid_X_tokenized,n_words,dictionary)

    
    test_dict = None
    test_dict_token = None
    if save_test!=None:
        test_dict,test_dict_token = get_input_json(filenames_test,w2v=None,token=1,feed=testX)
        
        
    return trainX,validX,testX,filenames_train,filenames_valid,filenames_test,test_dict,test_dict_token,embedding_initMat



def extract_labels(filenames_train,filenames_valid,filenames_test):

    rating = re.compile("_[0-9]+.")
    #Only safely annotated reviews are used for train/test
    trainY = []
    for doc in filenames_train:
        find_rating = rating.search(doc).group()
        doc_rating = int(list(filter(str.isdigit, find_rating))[0])
        if doc_rating<=4:
            #safely negatively annotated
            doc_rating = 0
        elif doc_rating>=7:
            #safely positively annotated
            doc_rating = 1
        trainY.append(doc_rating)

    testY = []
    for doc in filenames_test:
        find_rating = rating.search(doc).group()
        doc_rating = int(list(filter(str.isdigit, find_rating))[0])
        if doc_rating<=4:
            #safely negatively annotated
            doc_rating = 0
        elif doc_rating>=7:
            #safely positively annotated
            doc_rating = 1
        testY.append(doc_rating)

    validY = []
    for doc in filenames_valid:
        find_rating = rating.search(doc).group()
        doc_rating = int(list(filter(str.isdigit, find_rating))[0])
        if doc_rating<=4:
            #safely negatively annotated
            doc_rating = 0
        elif doc_rating>=7:
            #safely positively annotated
            doc_rating = 1
        validY.append(doc_rating)

    #Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)
    validY = to_categorical(validY, nb_classes=2)

    return trainY,validY,testY


def get_initial_embeddings_from_dictionary(n_words,embedding_dim,dictionary):

    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  
    w2v   = dict(zip(model.index2word, model.syn0))
    inv_dict = {v: k for k, v in dictionary.items()}
    
    ebd_init = np.zeros((n_words,embedding_dim))
    w2v_w = list(w2v.keys())
    for i in range(n_words):        
        if inv_dict[i] in w2v_w:
            ebd_init[i,:] = w2v[inv_dict[i]]
        else:
            ebd_init[i,:] = np.zeros((embedding_dim))
            
    return ebd_init    


def get_review_from_token(rev_matrix,inv_dictionary_w,save_mode,save_dir,n_words,embedding_dim,dictionary_w):

    review_num = rev_matrix.shape[0]
    texts = collections.OrderedDict()
    
    for i in range(review_num):
        x = rev_matrix[i,:].tolist()
        texts[i] =  ' '.join(inv_dictionary_w[id] for id in x)

    if save_mode=="pickle":
        with open(save_dir+"test_data_text.pickle", "wb") as f:  
            pickle.dump(texts,f) 
    else:
        with open(save_dir+"test_data_text.json", "w") as f: 
            json.dump(texts, f)    

    print("Exported test id:review  dictionary...")            
    embedding_initMat = None    
    #embedding_initMat =  get_initial_embeddings_from_dictionary(n_words,embedding_dim,dictionary_w)
    
    print("Got initial word embeddings...")

    return embedding_initMat


def get_ready_features(NUM_WORDS,INDEX_FROM,test_samples_num,save_mode,save_dir,embedding_dim,maxlen=None):

    train,test = imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)
    train_X,train_Y = train
    valid_X,valid_Y = test

    trainY = to_categorical(train_Y,2)
    validY = to_categorical(valid_Y,2)

    dictionary_w = imdb.get_word_index()
    dictionary_w = {k:(v+INDEX_FROM) for k,v in dictionary_w.items()}
    dictionary_w["<PAD>"] = 0
    dictionary_w["<START>"] = 1
    dictionary_w["<UNK>"] = 2
    inv_dictionary_w = {value:key for key,value in dictionary_w.items()}

    lengthsTr = np.max([len(s) for s in train_X])
    lengthsVd = np.max([len(s) for s in valid_X])
    if maxlen is None:
       maxlen = np.max(np.array([lengthsTr,lengthsVd]))

    trainX = sequence.pad_sequences(train_X, maxlen=maxlen)
    validX = sequence.pad_sequences(valid_X, maxlen=maxlen)

    testX = np.zeros((test_samples_num,maxlen))
    testY = np.zeros((test_samples_num,validY.shape[1]))
    test_idx = np.array([random.randrange(0,validX.shape[0],1) for k in range(test_samples_num)])
    testX[:,:] = validX[test_idx,:]
    testY[:,:] = validY[test_idx,:]
    valid_idx = [item for item in [k for k in range(validX.shape[0])] if item not in test_idx.tolist()]
    validdX = np.zeros((len(valid_idx),maxlen))
    validdY = np.zeros((len(valid_idx),validY.shape[1]))
    validdX[:,:] = validX[valid_idx,:]
    validdY[:,:] = validY[valid_idx,:]

    embedding_initMat = get_review_from_token(testX,inv_dictionary_w,save_mode,save_dir,NUM_WORDS,embedding_dim,dictionary_w)


    return trainX,trainY,validdX,validdY,testX,testY,embedding_initMat,dictionary_w,inv_dictionary_w



def preprocess_IMDBdata(n_words=None,INDEX_FROM=3,embedding_dim=300,test_samples_num=100,save_dir="/tmp/",save_mode="pickle"):
     
    """
    trainX,validX,testX,filenames_train,filenames_valid,filenames_test,test_dict,test_dict_token,embedding_initMat = extract_features(filenames_train_valid,filenames_test,seed,test_size,save_test,n_words,dictionary,embedding_dim)
#    extract_features_w2v(filenames,seed,test_size,save_test=None)
    
    trainX,validX,testX,maxlen = pad_sequences(trainX, validX,testX, value=0.)
    trainX = np.array(trainX)
    validX = np.array(validX)
    testX  = np.array(testX)
    
    trainY,validY,testY = extract_labels(filenames_train,filenames_valid,filenames_test)
    """
    trainX,trainY,validdX,validdY,testX,testY,embedding_initMat,dictionary_w,inv_dictionary_w = get_ready_features(n_words,INDEX_FROM,test_samples_num,save_mode,save_dir,embedding_dim,maxlen=None)

    return trainX,trainY,validdX,validdY,testX,testY,embedding_initMat,dictionary_w,inv_dictionary_w


   
