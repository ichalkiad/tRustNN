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
from tensorflow import float32 as tffloat32
import tflearn
import imdb_preprocess as imdb_pre
from sacred import Experiment
from sacred.observers import MongoObserver
import collections 

ex = Experiment('IMDBMovieReview-LSTM')
ex.observers.append(MongoObserver.create())

#embedding_layer = Embedding(input_dim=word_model.syn0.shape[0], output_dim=word_model.syn0.shape[1], weights=[word_model.syn0]


@ex.config
def config():

    #Train and test sets
    trainX = imdb_pre.trainX
    trainY = imdb_pre.trainY
    testX = imdb_pre.testX
    testY = imdb_pre.testY

    #Dictionary describing the architecture of the network
    net_arch = collections.OrderedDict()
    net_arch['input_size'] = 300
    net_arch['embedding']  = {'input_dim':300, 'output_dim':128, 'validate_indices':False,
                               'weights_init':'truncated_normal', 'trainable':True, 'restore':True,
                               'reuse':False, 'scope':None, 'name':"embed1"}
    net_arch['lstm']       = {'n_units':128, 'activation':'tanh', 'inner_activation':'sigmoid',
                               'dropout':None, 'bias':True, 'weights_init':None, 'forget_bias':1.0,
                               'return_seq':False, 'return_state':False, 'initial_state':None,
                               'dynamic':True, 'trainable':True, 'restore':True, 'reuse':False,
                               'scope':None,'name':"lstm1"}
    net_arch['fc']         = {'n_units':2, 'activation':'softmax', 'bias':True,'weights_init':'truncated_normal',
                               'bias_init':'zeros', 'regularizer':None, 'weight_decay':0.001, 'trainable':True,
                               'restore':True, 'reuse':False, 'scope':None,'name':"fc1"}
    net_arch['output']     = {'optimizer':'adam','loss':'categorical_crossentropy','metric':'default','learning_rate':0.001,
                               'dtype':tffloat32, 'batch_size':64,'shuffle_batches':True,'to_one_hot':False,'n_classes':None,
                               'trainable_vars':None,'restore':True,'op_name':None,'validation_monitors':None,'validation_batch_size':None,
                               'name':"xentr"}
    
    
    tensorboard_verbose = 3
    show_metric = True
    batch_size = 1
    save_path = "/tmp/tflearn_lstm_model"
    

@ex.automain
def train(trainX,trainY,testX,testY,net_arch,tensorboard_verbose,show_metric,batch_size):

    # Network building
    net = tflearn.input_data([None,net_arch['input_size']])
    for key in net_arch.keys():
        value = net_arch[key]
        if key=='embedding':
           net = tflearn.embedding(net, input_dim=value['input_dim'], output_dim=value['output_dim'],
                                   validate_indices=value['validate_indices'],weights_init=value['weights_init'],
                                   trainable=value['trainable'], restore=value['restore'],reuse=value['reuse'],
                                   scope=value['scope'],name=value['name'])

        if key=='lstm':
           net = tflearn.lstm(net,n_units=value['n_units'], activation=value['activation'],inner_activation=value['inner_activation'],
                              dropout=value['dropout'], bias=value['bias'], weights_init=value['weights_init'],
                              forget_bias=value['forget_bias'],return_seq=value['return_seq'], return_state=value['return_state'],
                              initial_state=value['initial_state'],dynamic=value['dynamic'], trainable=value['trainable'],
                              restore=value['restore'], reuse=value['reuse'],scope=value['scope'], name=value['name'])
        if key=='fc':
           net = tflearn.fully_connected(net,n_units=value['n_units'], activation=value['activation'], bias=value['bias'],
                                        weights_init=value['weights_init'],bias_init=value['bias_init'], regularizer=value['regularizer'],
                                        weight_decay=value['weight_decay'],trainable=value['trainable'],restore=value['restore'],
                                        reuse=value['reuse'],scope=value['scope'],name=value['name'])
        if key=='output':
           net = tflearn.regression(net,optimizer=value['optimizer'],loss=value['loss'],metric=value['metric'],
                                    learning_rate=value['learning_rate'],dtype=value['dtype'],batch_size=value['batch_size'],
                                    shuffle_batches=value['shuffle_batches'],to_one_hot=value['to_one_hot'],n_classes=value['n_classes'],
                                    trainable_vars=value['trainable_vars'],restore=value['restore'],op_name=value['op_name'],
                                    validation_monitors=value['validation_monitors'],validation_batch_size=value['validation_batch_size'],
                                    name=value['name']) 

    
    model = tflearn.DNN(net, tensorboard_verbose)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=show_metric, batch_size=batch_size)
    model.save(save_path+model.run+".tfl")
