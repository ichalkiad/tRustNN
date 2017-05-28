# -*- coding: utf-8 -*-
"""
LSTM to classify IMDB sentiment dataset.
References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).

    - http://ai.stanford.edu/~amaas/data/sentiment/
"""
from __future__ import division, print_function, absolute_import
import sys
import tflearn
import imdb_preprocess as imdb_pre
from sacred import Experiment
from sacred.observers import MongoObserver


trainX = imdb_pre.trainX
trainY = imdb_pre.trainY
testX = imdb_pre.testX
testY = imdb_pre.testY

ex = Experiment('IMDBMovieReview-LSTM')
ex.observers.append(MongoObserver.create())

#embedding_layer = Embedding(input_dim=word_model.syn0.shape[0], output_dim=word_model.syn0.shape[1], weights=[word_model.syn0]

# Network building
net = tflearn.input_data([None, 300])
net = tflearn.embedding(net, input_dim=300, output_dim=128, name="Embedding Layer")
net = tflearn.lstm(net, 128, dropout=None, dynamic=True, name="LSTM Layer")
net = tflearn.fully_connected(net, 2, activation='softmax', name="Fully-Connected Layer")
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name="Output Layer")



@ex.automain
def train():
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=1)
