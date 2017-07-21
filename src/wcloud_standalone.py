from __future__ import division, print_function, absolute_import
from IMDB_dataset.textData import filenames_test
from sacred.observers import FileStorageObserver
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
from dynamic_lstm_TF import build_network
import pickle
import IMDB_dataset.imdb_preprocess as imdb_pre
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import re
from bokeh.plotting import figure, show, output_file
import wcloud_group_coloring as wcColor


ex = Experiment('IMDBMovieReview-WordCloud')

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
    save_path = "./testLSTM1/"  #"./sacred_models/"
    tensorboard_dir = "./sacred_models/tf_logs/"
    run_id = "runIDstandalone"
    n_words = 10000
    dictionary = "/home/icha/tRustNN/imdb_dict.pickle"
    embedding_dim = 300
    ckp_path = "./sacred_models/" + run_id
    internals = "all"    
    feed_input_json = './bokeh_vis/static/test_data_input.json'
    internal_fc_json = "test_standalone_model_internals_fc.json"
    internal_hidden_json = "test_standalone_model_internals_lstm_hidden.json"
    internal_state_json = "test_standalone_model_internals_lstm_states.json"
    save_mode = "pickle"
    
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



def get_wcloud(LRP,k,save_dir,color_dict=None):

     ws = LRP[k]['words']
     scs = LRP[k]['scores']
     weights=collections.OrderedDict()
     for i in range(len(ws)):
         weights[ws[i]] = scs[i]
     wc = WordCloud(
            background_color="white",
            max_words=2000,
            width = 400,
            height = 400,
            stopwords=stopwords.words("english")
          )
     wc.generate_from_frequencies(weights)
     if color_dict!=None:
        default_color = 'grey'
        grouped_color_func_single = wcColor.SimpleGroupedColorFunc(color_dict, default_color)

        wc.recolor(color_func=grouped_color_func_single)

     
     save_filename = re.sub('/', '_', k[-18:-4])+"_word_cloud.png"
     try:
         os.remove(save_dir+save_filename)
     except OSError:
         pass
     wc.to_file(save_dir+save_filename)
        
     return save_filename,wc.to_image()

@ex.automain
def generate_wcloud(seed,net_arch,net_arch_layers,save_path,tensorboard_verbose,show_metric,batch_size,run_id,db,n_words,dictionary,embedding_dim,tensorboard_dir,ckp_path,internals,feed_input_json,internal_fc_json,internal_hidden_json,internal_state_json):

    save_dir = save_path+run_id+"/"
    if not os.path.exists(save_dir):
       os.makedirs(save_dir)
    if os.path.isfile(save_dir+"LRP.pickle")==False: 

        """
        _,_,testX,_,_,_,_,_,filenames_test_sfd,_ = imdb_pre.preprocess_IMDBdata(seed,filenames_train_valid,filenames_test,n_words,dictionary)
        """
        with open('trainValidtest.pickle','rb') as handle:
            (trainX,validX,testX,trainY,validY,testY,filenames_train,filenames_valid,filenames_test_sfd) = pickle.load(handle)
        
        model, layer_outputs = build_network(net_arch,net_arch_layers,tensorboard_verbose,testX.shape[1],embedding_dim,tensorboard_dir,batch_size,ckp_path)
        sess = model.session 
        saver = model.trainer.saver
        saver.restore(sess,tf.train.latest_checkpoint(ckp_path))

        feed = testX 
        input_files = filenames_test_sfd

        predicted_tgs = model.predict_label(feed)

        LRP = lrp.lrp_full(model,input_files,net_arch,net_arch_layers,feed_input_json,save_dir+internal_fc_json,save_dir+internal_hidden_json,save_dir+internal_state_json,eps=0.001,delta=0.0,save_dir=save_dir,lstm_actv1=expit,lstm_actv2=np.tanh,topN=5,debug=False,predictions=predicted_tgs)

        """
        print(heatmap.html_heatmap(LRP['/home/yannis/Desktop/tRustNN/aclImdb/test/pos/127_10.txt']['words'], LRP['/home/yannis/Desktop/tRustNN/aclImdb/test/pos/127_10.txt']['scores'] ))
        """
    else:
        with open(save_dir+"LRP.pickle","rb") as handle:
            (LRP,predicted_tgs) = pickle.load(handle)
        

    kkeys = list(LRP.keys())
    for k in kkeys:
        _ = get_wcloud(LRP,k,save_dir)
        

