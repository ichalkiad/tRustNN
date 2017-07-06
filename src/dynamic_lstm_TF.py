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
from IMDB_dataset.textData_cluster import filenames_train_valid,filenames_test
from parameter_persistence import export_serial_model,export_serial_lstm_data
from sacred.observers import FileStorageObserver
import IMDB_dataset.imdb_preprocess as imdb_pre
from sacred.observers import MongoObserver
from scipy.special import expit
from sacred import Experiment
import tensorflow as tf
import collections
import tflearn
import extend_recurrent
import numpy as np
import json
import sys
import lrp
import os
import heatmap
#import _pickle

ex = Experiment('IMDBMovieReview-LSTM')

@ex.config
def config():
    
    db = ""
    if db=="mongo":
        print("Using mongodb for logging")
        ex.observers.append(MongoObserver.create())
    elif db=="file":
        print("Using local file storage for logging")        
        ex.observers.append(FileStorageObserver.create('SacredRunLog'))


    net_arch_layers = ['lstm','fc','output']
    tensorboard_verbose = 3
    show_metric = True
    batch_size = 1
    save_path = "./sacred_models/"
    tensorboard_dir = "./sacred_models/tf_logs/"
    run_id = "runID"
    n_words = 10000
    dictionary = "/home/icha/tRustNN/imdb_dict.pickle"
    embedding_dim = 300
    ckp_path = None #"./sacred_models/ckp/"
    internals = "all"    
    save_mode = "json"
    
    #Dictionary describing the architecture of the network
    net_arch = collections.OrderedDict()
    net_arch['lstm']       = {'n_units':128, 'activation':'tanh', 'inner_activation':'sigmoid',
                              'dropout':None, 'bias':True, 'weights_init':None, 'forget_bias':1.0,
                              'return_seq':True, 'return_state':True, 'initial_state':None,
                              'dynamic':True, 'trainable':True, 'restore':True, 'reuse':False,
                              'scope':None,'name':"lstm"}
    net_arch['fc']         = {'n_units':2, 'activation':'softmax', 'bias':True,'weights_init':'truncated_normal',
                              'bias_init':'zeros', 'regularizer':None, 'weight_decay':0.001, 'trainable':True,
                              'restore':True, 'reuse':False, 'scope':None,'name':"fc"}
    net_arch['output']     = {'optimizer':'adam','loss':'categorical_crossentropy','metric':'default','learning_rate':0.001,
                              'dtype':tf.float32, 'batch_size':batch_size,'shuffle_batches':True,'to_one_hot':False,'n_classes':None,
                              'trainable_vars':None,'restore':True,'op_name':None,'validation_monitors':None,
                              'validation_batch_size':batch_size,'name':"output"}

    
    
def build_network(net_arch,net_arch_layers,tensorboard_verbose,sequence_length,embedding_dim,tensorboard_dir,batch_size,ckp_path=None):

    # Network building
    net = tflearn.input_data([None,sequence_length,embedding_dim])
    layer_outputs = dict()
    prev_incoming = net

    for k in range(len(net_arch_layers)):
        key = net_arch_layers[k]
        value = net_arch[key]
        if 'lstm' in key:
            output, state = extend_recurrent.lstm(prev_incoming,n_units=value['n_units'], activation=value['activation'],
                                         inner_activation=value['inner_activation'],dropout=value['dropout'], bias=value['bias'],
                                         weights_init=value['weights_init'],forget_bias=value['forget_bias'],return_seq=value['return_seq'],
                                         return_state=value['return_state'],initial_state=value['initial_state'],dynamic=value['dynamic'],
                                         trainable=value['trainable'],restore=value['restore'], reuse=value['reuse'],scope=value['scope'],
                                         name=value['name'])
            n = key+"_output"
            layer_outputs[n] = output
            n = key+"_cell_state"
            layer_outputs[n] = state
            if ("lstm" not in net_arch_layers[k+1]) and (value['return_seq']==True):
                prev_incoming = output[-1]
            else:
                prev_incoming = output
        if 'fc' in key:
            fc_output = tflearn.fully_connected(prev_incoming,n_units=value['n_units'], activation=value['activation'], bias=value['bias'],
                                        weights_init=value['weights_init'],bias_init=value['bias_init'], regularizer=value['regularizer'],
                                        weight_decay=value['weight_decay'],trainable=value['trainable'],restore=value['restore'],
                                        reuse=value['reuse'],scope=value['scope'],name=value['name'])
            n = key+"_output"
            layer_outputs[n] = fc_output
            prev_incoming = fc_output
        if key=='output':
           net = tflearn.regression(prev_incoming,optimizer=value['optimizer'],loss=value['loss'],metric=value['metric'],
                                    learning_rate=value['learning_rate'],dtype=value['dtype'],batch_size=batch_size,
                                    shuffle_batches=value['shuffle_batches'],to_one_hot=value['to_one_hot'],n_classes=value['n_classes'],
                                    trainable_vars=value['trainable_vars'],restore=value['restore'],op_name=value['op_name'],
                                    validation_monitors=value['validation_monitors'],validation_batch_size=batch_size,
                                    name=value['name']) 

    model = tflearn.DNN(net, tensorboard_verbose,tensorboard_dir=tensorboard_dir,checkpoint_path=ckp_path)

    return model,layer_outputs


@ex.automain
def train(seed,net_arch,net_arch_layers,save_path,tensorboard_verbose,show_metric,batch_size,run_id,db,n_words,dictionary,embedding_dim,tensorboard_dir,ckp_path,internals,save_mode):
    
    print("Extracting features...")
    #Train, valid and test sets. Have to return filenames_test as we have now shuffled them
    trainX,validX,testX,trainY,validY,testY,filenames_train,filenames_valid,filenames_test_sfd,maxlen = imdb_pre.preprocess_IMDBdata(seed,filenames_train_valid,filenames_test,n_words,dictionary)

    """
    with open('trainValidtest.pickle','rb') as handle:
        (trainX,validX,testX,trainY,validY,testY,filenames_train,filenames_valid,filenames_test_sfd) = _pickle.load(handle)
    """
    
    print("Training model...")
    
    model, layer_outputs = build_network(net_arch,net_arch_layers,tensorboard_verbose,trainX.shape[1],embedding_dim,tensorboard_dir,batch_size,ckp_path)
    model.fit(trainX, trainY, validation_set=(validX, validY), show_metric=show_metric, batch_size=batch_size)

    
    print("Evaluating trained model on test set...")
    score = model.evaluate(testX,testY,batch_size=maxlen)
    print("Accuracy on test set: %0.4f%%" % (score[0] * 100))
    
    
    save_dir = save_path+run_id+"/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    #Save model to json format
    export_serial_model(model,net_arch_layers,save_dir)
    
    #Get model's internals for 'feed' input
    """
    feed = trainX
    input_files = filenames_train
    export_serial_lstm_data(model,layer_outputs,feed,input_files,internals,save_dir+"train_")

    feed = validX
    input_files = filenames_valid
    export_serial_lstm_data(model,layer_outputs,feed,input_files,internals,save_dir+"valid_")
    """
    feed = testX
    input_files = filenames_test_sfd
    
    export_serial_lstm_data(model,layer_outputs,feed,input_files,internals,save_dir+"test_",save_mode=save_mode)
    print("Exported internals...")
    
    d = imdb_pre.get_input_json(input_files)
    with open(save_dir+"test_data_input.json", "w") as f:
        json.dump(d, f)
    print("Exported test data dictionary...")
    
    LRP = lrp.lrp_full(model,input_files,net_arch,net_arch_layers,save_dir+"test_data_input.json",save_dir+"test_model_internals_fc."+save_mode,save_dir+"test_model_internals_lstm_hidden."+save_mode,save_dir+"test_model_internals_lstm_states."+save_mode,eps=0.001,delta=0.0,lstm_actv1=expit,lstm_actv2=np.tanh,debug=False)
    predicted_tgs = model.predict_label(feed)
    with open(save_dir+"LRP.pickle","wb") as handle:
        _pickle.dump((LRP,predicted_tgs),handle)
    print("Finished with LRP...")
    
    #Delete part that creates problem in restoring model - should still be able to evaluate, but tricky for continuing training
    del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
    model.save(save_dir+"tf_model.tfl")
    print("Saved model...now exiting...")

