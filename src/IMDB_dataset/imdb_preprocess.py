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


def get_input_json(filenames,w2v=None):

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
        
    return data



def pad_sequences(trainX, validX, testX, maxlen=None, dtype='int32', padding='post', truncating='post', value=0.):

    lengthsTr = np.max([len(s) for s in trainX])
    lengthsVd = np.max([len(s) for s in validX])
    lengthsTe = np.max([len(s) for s in testX])
    elem_length = len(trainX[0][0])
    pads = np.zeros((elem_length,))

    if maxlen is None:
        maxlen = np.max(np.array([lengthsTr,lengthsVd,lengthsTe]))

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



def remove_unk(x,n_words):
    return [[1 if w >= n_words else w for w in sen] for sen in x]



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



def extract_features(filenames,seed,test_size,n_words,dictionary):
    
    random.shuffle(filenames)

    X_train, X_valid, y_train, y_valid = train_test_split(filenames, np.zeros(len(filenames)),test_size=2*test_size,random_state=seed)
    filenames_train = X_train
    filenames_valid = X_valid

    X_valid, X_test, y_valid, y_test = train_test_split(filenames_valid, np.zeros(len(filenames_valid)),test_size=0.5,random_state=seed)
    filenames_valid = X_valid
    filenames_test  = X_test


    with open(dictionary, 'rb') as handle:
         d = pickle.load(handle)

    X_train, X_valid, y_train, y_valid = train_test_split(filenames_train_valid, np.zeros(len(filenames_train_valid)),
                                                          test_size=0.1, random_state=seed)
    filenames_train = X_train
    filenames_valid = X_valid
    
    testX = []
    for i in filenames_test:
        testX.append(Path(i).read_text())
    trainX = []
    for i in filenames_train:
        trainX.append(Path(i).read_text())
    validX = []
    for i in filenames_valid:
        validX.append(Path(i).read_text())

    vectorizer = we.CountVectorizer(vocabulary=d)
    testX_tokenized = vectorizer.fit_transform(testX).toarray()
    del testX
    testX = remove_unk(testX_tokenized,n_words)
    trainX_tokenized = vectorizer.fit_transform(trainX).toarray()
    del trainX
    trainX = remove_unk(trainX_tokenized,n_words)
    validX_tokenized = vectorizer.fit_transform(validX).toarray()
    del validX
    validX = remove_unk(validX_tokenized,n_words)

    return trainX,validX,testX,filenames_train,filenames_valid,filenames_test



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



def preprocess_IMDBdata(seed,filenames,n_words=None,dictionary=None,test_size=0.1,save_test=None):

    trainX,validX,testX,filenames_train,filenames_valid,filenames_test,test_dict = extract_features_w2v(filenames,seed,test_size,save_test)
    trainX,validX,testX,maxlen = pad_sequences(trainX, validX, testX, value=0.)
    trainX = np.array(trainX)
    validX = np.array(validX)
    testX  = np.array(testX)
    
    trainY,validY,testY = extract_labels(filenames_train,filenames_valid,filenames_test)
    
    return trainX,validX,testX,trainY,validY,testY,filenames_train,filenames_valid,filenames_test,maxlen,test_dict
