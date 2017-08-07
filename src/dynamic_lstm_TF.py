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
from IMDB_dataset.textData_cluster import filenames
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
import pickle

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
    run_id = "runID_newOutput"
    n_words = 10000 #89527 
    dictionary = "/home/icha/tRustNN/imdb_dict.pickle"
    embedding_dim = 300
    ckp_path = None #"./sacred_models/ckp/"
    internals = "all"    
    save_mode = "pickle"
    n_epoch = 2
    test_size = 0.05 # -1 for whole test set
    embedding_layer = 1
    
    #Dictionary describing the architecture of the network
    net_arch = collections.OrderedDict()
    net_arch['lstm']       = {'n_units':10, 'activation':'tanh', 'inner_activation':'sigmoid',
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

    
    
def build_network(net_arch,net_arch_layers,tensorboard_verbose,sequence_length,embedding_dim,tensorboard_dir,batch_size,n_words,embedding_layer,ckp_path=None,embedding_initMat=None):

    layer_outputs = dict()
   
    # Network building
    if embedding_layer:
        net = tflearn.input_data([None,sequence_length]) 
        W = tf.constant(embedding_initMat, dtype=np.float32,name="W")
        ebd_output = tflearn.embedding(net, input_dim=n_words, output_dim=embedding_dim,weights_init=W, name='embedding')
        n = "embedding_output"
        layer_outputs[n] = ebd_output
        prev_incoming = ebd_output
    else:
        net = tflearn.input_data([None,sequence_length],embedding_dim)
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
def train(seed,net_arch,net_arch_layers,save_path,n_epoch,tensorboard_verbose,show_metric,batch_size,run_id,db,n_words,dictionary,embedding_dim,tensorboard_dir,ckp_path,embedding_layer,internals,save_mode,test_size):

    save_dir = save_path+run_id+"/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    with open(dictionary, 'rb') as handle:
         dictionary_w = pickle.load(handle)
    inv_dictionary_w = {v: k for k, v in dictionary_w.items()}

    
    print("Extracting features...")
    
    #Train, valid and test sets. Have to return filenames_test as we have now shuffled them
    
    trainX,validX,testX,trainY,validY,testY,filenames_train,filenames_valid,filenames_test_sfd,maxlen,test_dict,test_dict_token,embedding_initMat = imdb_pre.preprocess_IMDBdata(seed=seed,filenames=filenames,n_words=n_words,dictionary=dictionary_w,embedding_dim=embedding_dim,test_size=test_size,save_test="save_test")
    
    """
    with open('trainValidtestNew.pickle','rb') as handle:
        (trainX,validX,testX,trainY,validY,testY,filenames_train,filenames_valid,filenames_test_sfd,maxlen,test_dict,test_dict_token,embedding_initMat) = pickle.load(handle)
    
    """
    d = test_dict        
    if save_mode=="pickle":
        with open(save_dir+"test_data_input.pickle", "wb") as f:
            pickle.dump(d,f)
    else:
        with open(save_dir+"test_data_input.json", "w") as f:
            json.dump(d, f)
    print("Exported test data dictionary...")
    d = test_dict_token
    if save_mode=="pickle":
        with open(save_dir+"test_data_input_token.pickle", "wb") as f:
            pickle.dump(d,f)
    else:
        with open(save_dir+"test_data_input_token.json", "w") as f:
            json.dump(d, f)
    print("Exported test data token dictionary...")
    
    """
    with open('trainValidtestNew.pickle','wb') as handle:
        pickle.dump((trainX,validX,testX,trainY,validY,testY,filenames_train,filenames_valid,filenames_test_sfd,maxlen,test_dict,test_dict_token,embedding_initMat),handle)
    """
    print("Training model...")
    
    model, layer_outputs = build_network(net_arch,net_arch_layers,tensorboard_verbose,trainX.shape[1],embedding_dim,tensorboard_dir,batch_size,n_words,embedding_layer,ckp_path,embedding_initMat)

    model.fit(trainX, trainY, validation_set=(validX, validY), n_epoch=n_epoch, show_metric=show_metric, batch_size=batch_size)
    
    print("Evaluating trained model on test set...")
    score = model.evaluate(testX,testY)
    print("Accuracy on test set: %0.4f%%" % (score[0] * 100))
    
    
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
    """
    export_serial_lstm_data(model,layer_outputs,feed,input_files,internals,save_dir+"test_",save_mode=save_mode)
    
    print("Exported internals...")
    """
    #Delete part that creates problem in restoring model - should still be able to evaluate, but tricky for continuing training
    del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
    model.save(save_dir+"tf_model.tfl")
    print("Saved model...")    
   
    predicted_tgs = model.predict_label(feed)

          
    LRP = lrp.lrp_full(model,embedding_layer,n_words,input_files,net_arch,net_arch_layers,save_dir+"test_data_input_token."+save_mode,save_dir+"test_data_input."+save_mode,save_dir+"test_model_internals_fc."+save_mode,save_dir+"test_model_internals_lstm_hidden."+save_mode,save_dir+"test_model_internals_lstm_states."+save_mode,save_dir+"test_model_internals_ebd."+save_mode,inv_dictionary_w,eps=0.001,delta=0.0,save_dir=save_dir,lstm_actv1=expit,lstm_actv2=np.tanh,topN=5,debug=False,predictions=predicted_tgs)


    
    with open(save_dir+"lstm_predictions.pickle","wb") as handle:
        pickle.dump(predicted_tgs,handle)
    print("Finished with LRP and related data...now exiting...")
    

