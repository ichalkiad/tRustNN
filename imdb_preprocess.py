import gensim
import re
import numpy as np
import WordEmbedder as we
from textData import filenames_train, filenames_test
from tflearn.data_utils import to_categorical


# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  
w2v = dict(zip(model.index2word, model.syn0))

#Initialize preprocessor
preprocessor = we.NLTKPreprocessor()
vectorizer   = we.TfidfEmbeddingVectorizer(w2v)


#Tokenization - train_X_tokenized is a list containing lists of words for each review in the passed argument
train_X_tokenized = preprocessor.transform(filenames_train)
test_X_tokenized = preprocessor.transform(filenames_test)
#Get TF-IDF weighting terms
vectorizer.fit(train_X_tokenized)
vectorizer.fit(test_X_tokenized)
#Get Word2Vec representation of document weighted with TF-IDF
train_X_w2v = vectorizer.transform(train_X_tokenized)
test_X_w2v = vectorizer.transform(test_X_tokenized)

trainX = train_X_w2v
testX = test_X_w2v

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


#Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

