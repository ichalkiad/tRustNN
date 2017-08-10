from scipy.special import expit
import tensorflow as tf
import numpy as np
from numpy import newaxis as na
import data_format
import collections
import tflearn
from lrp_linear import lrp_linear as lrp_linear
import json
import sys
import re
import pickle
import gensim


def get_lrp_timedata(LRP):

    out_reversed = []
    kkeys = list(LRP.keys())
    lens = []
    for i in kkeys:
        lens.append(len(list(LRP[i]['words'])))
    max_len = np.max(lens)
    for i in range(max_len):
        j = 0
        normalize_factor = 0
        lrp_t = 0
        for k in kkeys:
            if lens[j]-1-i>=0:
                normalize_factor = normalize_factor + 1
                lrp = abs(list(LRP[k]['scores'])[lens[j]-1-i])  #abs, since we want the total LRP, either positive or negative
                lrp_t = lrp_t + lrp
            
            j = j + 1
        out_reversed.append(lrp_t/normalize_factor)
    
    return out_reversed[::-1] #reverse for time = 0...T

def get_PosNegNeurons_dict(i,predictions,lrp_neurons):
# Get neurons that trigger exclusively for positive or negative reviews according to the network. Assign them to neutral if activate for both types of reviews.

    reviewLRP_data = {"pos":[],"neg":[],"neutral":[]}

    pred = -1
    if predictions[i,0]==1:
       pred = 0
    elif predictions[i,0]==0:
       pred = 1
    
    
    if pred==0:
       for j in lrp_neurons:
           if reviewLRP_data["neg"]==[]:
              reviewLRP_data["neg"] = [j]
           else:
              if j in reviewLRP_data["pos"]:
                   reviewLRP_data["pos"].remove(j)
                   reviewLRP_data["neutral"].append(j)
              elif j not in reviewLRP_data["neg"]:
                   reviewLRP_data["neg"].append(j)
    elif pred==1:
       for j in lrp_neurons:
           if reviewLRP_data["pos"]==[]:
              reviewLRP_data["pos"] = [j]
           else:
              if j in reviewLRP_data["neg"]:
                 reviewLRP_data["neg"].remove(j)
                 reviewLRP_data["neutral"].append(j)
              elif j not in reviewLRP_data["pos"]:
                 reviewLRP_data["pos"].append(j)
    
    return reviewLRP_data

def get_NeuronType(reviewLRP_data,neuron_num):
# Assign a label to each neuron based on whether it activates on positive-,negative-only or both types of reviews.
    
    posNeg_predictionLabel = np.zeros((neuron_num,))

    for i in range(neuron_num):
        if i in reviewLRP_data["pos"]:
            posNeg_predictionLabel[i] = 1
        elif i in reviewLRP_data["neg"]:
            posNeg_predictionLabel[i] = 2

    return posNeg_predictionLabel



def get_NeuronSimilarity_AllReviews(neuronWords_data_fullTestSet):

    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  
    w2v   = dict(zip(model.index2word, model.syn0))

    keys = list(neuronWords_data_fullTestSet.keys())

    dstMat = np.zeros((len(keys),len(keys)))
    for i in range((len(keys))):
        for j in range(i,len(keys)):
                 dstMat[i,j] = neuron_distance(neuron1=list(neuronWords_data_fullTestSet[keys[i]]),neuron2=list(neuronWords_data_fullTestSet[keys[j]]),w2v=w2v)
    
    return dstMat


def get_MostExcitingWords_allReviews(save_dir,neuronWords_jsons,topN=5):
#Get list of top-N exciting words for each neuron based on the whole dataset of reviews
    
    #neuron-word data dictionary
    nw_data = dict()
    done = []
    for i in neuronWords_jsons:
        keys,data = data_format.get_data(save_dir+i)
        kkeys = list(keys)
        for j in kkeys:
            if j not in done:
                if j in list(nw_data.keys()):
                    vals = list(data[j]) + list(set(list(nw_data[j]))-set(list(data[j])))
                    nw_data[j] = vals
                else:
                    nw_data[j] = data[j]
                if len(list(nw_data[j]))>=topN:
                    done.append(j)

    return nw_data        


def neuron_value(test_data_json,review,neuron,w2v):

    if w2v==None:
        keys_test,data_test = data_format.get_data(test_data_json)
        kkeys = list(data_test[review].keys())
        kdata = data_test[review]
        val = 0.0
        for w in neuron:
            val = val + np.array(list(kdata[w]))
    else:
        val = 0.0
        for w in neuron:
            if w in w2v:
                val = val + np.array(w2v[w])

    return np.array(val).sum()/len(neuron)


def neuron_distance(test_data_json=None,review=None,neuron1=[],neuron2=[],w2v=None):
# neuron1, neuron2 are given as a list of words that trigger them most
    
    n1_n2 = [item for item in neuron1 if item not in neuron2]
    n2_n1 = [item for item in neuron2 if item not in neuron1]

    if set(neuron1) == set(neuron2):
        return 0.0
    if (len(n1_n2)>=1 and len(n2_n1)>=1):
        a = neuron_value(test_data_json,review,neuron1,w2v)
        b = neuron_value(test_data_json,review,neuron2,w2v)
        return abs(a-b)
    else:
    #return arbitrary large value
        return 1000

    
def get_DstMatrix_singleReview(review_MaxAct_json,test_data_json,review):
# Get similarity matrix between neurons based on the custom distance function defined above, calculated based on a single review.
    keys,data = data_format.get_data(review_MaxAct_json)
    kkeys = list(keys)
    dstMat = np.zeros((len(kkeys),len(kkeys)))
    for i in range((len(kkeys))):
        for j in range(i,len(kkeys)):
                 dstMat[i,j] = neuron_distance(test_data_json,review,list(data[kkeys[i]]),list(data[kkeys[j]]))
            
    return dstMat
                

def invert_dict_nonunique(d,topN):
    newdict = {}
    for k in d:
        i = 0
        for v in d[k]:
            if i<topN:
                newdict.setdefault(v, []).append(k)
                i = i + 1
            
    return newdict


def get_NeuronExcitingWords_dict(lstm_hidden_json,kkeys,k,save_dir,topN=5):
# Get the N words that excite each LSTM cell maximum, i.e. neuron of output has maximum value during forward pass
    
     d = collections.OrderedDict()
     keys_hidden,data_hidden = data_format.get_data(lstm_hidden_json)
     kdata = data_hidden[k]

     for i in range(len(kkeys)):
          ord_cells = np.argsort(kdata[i,:],axis=0,kind='quicksort')
          d[kkeys[i]] = ord_cells[-(topN+1):-1].tolist()

     NtoW = invert_dict_nonunique(d,topN)
     NtoW_keys = map(int,list(NtoW.keys()))
     for i in range(kdata.shape[1]):
         if i not in NtoW_keys:
             NtoW[str(i)] = []     

     print(map(int,list(NtoW.keys())))   #########################
             
     with open(save_dir+re.sub('/', '_', k[-18:-4])+"_ActCells.json", 'w') as f:
            json.dump(NtoW, f)

     return re.sub('/', '_', k[-18:-4])+"_ActCells.json",NtoW


 
def get_topLRP_cells(lrp_fc,review,save_dir,topN=5):
# Get the N LSTM cells that have been assigned the maximum LRP value from the fully connected layer.
    
     sorted_LRP = np.argsort(lrp_fc,axis=0,kind='quicksort')
     idx = sorted_LRP[-(topN+1):-1].tolist()
     
     with open(save_dir+re.sub('/', '_', review[-18:-4])+"_lrpCells.json", 'w') as f:
            json.dump({review:idx}, f)

     return idx

 
def get_gate(W,b,in_concat):
# in_concat is concatenation(current_input, input_prev_timestep)
# b should correspond to specific gate bias
     
     return np.dot(W.transpose(),in_concat) + b


def get_gates_out_t(in_concat,b,i_arr,f_arr,g_arr,o_arr,d,lstm_actv1,lstm_actv2):

     i_t = lstm_actv1(get_gate(i_arr,b[0:d],in_concat))
     g_t = lstm_actv2(get_gate(g_arr,b[d:2*d],in_concat))
     f_t = lstm_actv1(get_gate(f_arr,b[2*d:3*d],in_concat))
     o_t = lstm_actv1(get_gate(o_arr,b[3*d:4*d],in_concat))

     return i_t,g_t,f_t


def lrp_embedding(model,emb_name,n_words,feed,lstm_first_input,lrp_lstm,dictionary,eps,delta,debug):
#first_lstm_output : the lstm layer that connects to the embedding
     
        layer = tflearn.variables.get_layer_variables_by_name(emb_name)
        W_ebd = model.get_weights(layer[0])
        z = (np.array(feed).astype(int))
        nonz = np.array(np.nonzero(z)).flatten()
        sequence_len = nonz.shape[0]
        W = W_ebd
        N = n_words
        b = np.zeros((W.shape[1]))
        Rout = lrp_lstm
        ws = []
        scores = []
    
        for t in range(sequence_len):
            zj = lstm_first_input[t,:]
            zi = np.zeros((W.shape[0]))
            zi[z[t]] = 1
            R_t = Rout[t,:]
            lrp_ebd = lrp_linear(zi, W, b, zj, R_t, N, eps, delta, debug)
            ws.append(dictionary[z[t]])
            scores.append(lrp_ebd[z[t]])
        
        LRP = collections.OrderedDict(words=ws,scores=scores)

        return LRP


 

def lrp_fullyConnected(model,fc_name,last_lstm_output,fc_out,lrp_mask,d,T,classes,eps,delta,debug):
#last_lstm_output : the lstm layer that connects to the fully connected
     
        layer = tflearn.variables.get_layer_variables_by_name(fc_name)
        W_fc = model.get_weights(layer[0])
        b_fc = model.get_weights(layer[1])
        zi = last_lstm_output[T[0]-1,:]
        zj = fc_out
        W = W_fc
        b = np.zeros((classes)) 
        Rout = fc_out*lrp_mask
        N = 2*d
        lrp_fc = lrp_linear(zi, W, b, zj, Rout, N, eps, delta, debug)

        return lrp_fc


def lrp_lstm(model,layer_name,feed,T,d,lrp_fc,lstm_hidden,lstm_cell,lstm_actv1,lstm_actv2,eps,delta,debug):
        
        layer = tflearn.variables.get_layer_variables_by_name(layer_name)
        input, new_input, forget, output = tf.split(layer[0], num_or_size_splits=4, axis=1)
        i_arr = model.session.run(input)
        g_arr = model.session.run(new_input) 
        f_arr = model.session.run(forget)
        o_arr = model.session.run(output)
        b_tot = model.get_weights(layer[1])
 
        lstm_lrp_x = np.zeros(feed.shape)
        lstm_lrp_h = np.zeros((T[0]+1, d))
        lstm_lrp_c = np.zeros((T[0]+1, d))
        lstm_lrp_g = np.zeros((T[0], d))
        
        lstm_lrp_h[T[0]-1,:] = lrp_fc
        
        for t in reversed(range(T[0])):

             lstm_lrp_c[t,:] += lstm_lrp_h[t,:]

             x_t = feed[t,:]
             h_t = lstm_hidden[t-1,:]
             in_concat = np.concatenate((x_t,h_t),axis=0)
             i_t,g_t,f_t = get_gates_out_t(in_concat,b_tot,i_arr,f_arr,g_arr,o_arr,d,lstm_actv1,lstm_actv2)
             zi = f_t*lstm_cell[t-1,:]
             zj = lstm_cell[t,:]
             W = np.identity(d)
             b = np.zeros((d))
             Rout = lstm_lrp_c[t,:]
             N = 2*d
             lstm_lrp_c[t-1,:] = lrp_linear(zi, W, b, zj, Rout, N, eps, delta, debug)

             zi = i_t*g_t
             lstm_lrp_g[t,:] = lrp_linear(zi, W, b, zj, Rout, N, eps, delta, debug)

             zi = feed[t,:]
             zj = g_t
             W = g_arr[0:T[1],:]
             b = b_tot[d:2*d]
             Rout = lstm_lrp_g[t,:]
             N = d + T[1]
             lstm_lrp_x[t,:] = lrp_linear(zi, W, b, zj, Rout, N, eps, delta, debug)
             
             zi = lstm_hidden[t-1,:]
             W = g_arr[T[1]:,:]
             lstm_lrp_h[t-1,:] = lrp_linear(zi, W, b, zj, Rout, N, eps, delta, debug)

     
        return lstm_lrp_x,(lstm_lrp_h,lstm_lrp_g,lstm_lrp_c)

   
def load_intermediate_outputs(input_filename,embedding_json,fc_json,lstm_hidden_json,lstm_cell_json,layer_name=None):
#layer_name is currently not needed - for networks with more layers we will need it, as the json structure will change

     
     keys_hidden,data_hidden = data_format.get_data(lstm_hidden_json)
     keys_cell,data_cell = data_format.get_data(lstm_cell_json)
     keys_fc,data_fc = data_format.get_data(fc_json)
     keys_ebd,data_ebd = data_format.get_data(embedding_json)
     
     lstm_hidden = data_hidden[input_filename]
     lstm_cell   = data_cell[input_filename]
     fc_out      = data_fc[input_filename]
     embedding_output_data = data_ebd[input_filename]
     
     d = lstm_cell.shape[1]
     
     return fc_out,lstm_hidden,lstm_cell,embedding_output_data,d



def lrp_single_input(model,embedding_layer,n_words,layer_names,input_filename,single_input_data,data_token,eps,delta,fc_json,lstm_hidden_json,lstm_cell_json,ebd_json,dictionary,target_class,T,classes=2,lstm_actv1=expit,lstm_actv2=np.tanh,debug=False):

        
    with model.session.as_default():

        lrp_mask = np.zeros((classes))
        lrp_mask[target_class] = 1.0

        fc_out,lstm_hidden,lstm_cell,embedding_output_data,d = load_intermediate_outputs(input_filename,ebd_json,fc_json,lstm_hidden_json,lstm_cell_json,layer_name=None)
        
        #LRP through fc layer
        fc_name = "fc"
        lrp_fc = lrp_fullyConnected(model,fc_name,lstm_hidden,fc_out,lrp_mask,d,T,classes,eps,delta,debug)

        #LRP through embedding layer if needed
        if embedding_layer:
            lstm_name = "lstm"        
            feed = embedding_output_data
            lstm_lrp_x,(lstm_lrp_h,lstm_lrp_g,lstm_lrp_c) = lrp_lstm(model,lstm_name,feed,T,d,lrp_fc,lstm_hidden,lstm_cell,lstm_actv1,lstm_actv2,eps,delta,debug)
           
            emb_name = "embedding"
            feed = data_token
            lrp_input = lrp_embedding(model,emb_name,n_words,feed,embedding_output_data,lstm_lrp_x,dictionary,eps,delta,debug)

        else:
            #LRP through lstm layer
            lstm_name = "lstm"        
            feed = single_input_data
            lstm_lrp_x,(lstm_lrp_h,lstm_lrp_g,lstm_lrp_c) = lrp_lstm(model,lstm_name,feed,T,d,lrp_fc,lstm_hidden,lstm_cell,lstm_actv1,lstm_actv2,eps,delta,debug)


    return lrp_input,lrp_fc,lstm_lrp_x,(lstm_lrp_h,lstm_lrp_g,lstm_lrp_c)


        
def lrp_full(model,embedding_layer,n_words,input_filename,net_arch,net_arch_layers,test_data_token_json,test_data_json,fc_out_json,lstm_hidden_json,lstm_cell_json,ebd_json,dictionary,eps,delta,save_dir,lstm_actv1=expit,lstm_actv2=np.tanh,topN=5,debug=False,predictions=None):

    LRP = collections.OrderedDict()
    totalLRP = collections.OrderedDict()

    neuronWords_jsons = []
    similarityMatrix_PerReview = collections.OrderedDict()
   
    
    keys_test,data_test = data_format.get_data(test_data_json)
    _,data_test_token = data_format.get_data(test_data_token_json)
    for i in range(len(list(keys_test))):
         k = list(keys_test)[i]
         kkeys = list(data_test[k].keys())
         kdata = np.array(list(data_test[k].values()))
         T = kdata.shape
         data_token = np.array(data_test_token[k])

         lrp_input,lrp_fc,lstm_lrp_x,(lstm_lrp_h,lstm_lrp_g,lstm_lrp_c) = lrp_single_input(model,embedding_layer,n_words,net_arch_layers,k,kdata,data_token,eps,delta,fc_out_json,lstm_hidden_json,lstm_cell_json,ebd_json,dictionary,target_class=1,T=T,classes=2,lstm_actv1=expit,lstm_actv2=np.tanh,debug=debug)

         lrp_neurons = get_topLRP_cells(lrp_fc,k,save_dir,topN)
         reviewLRP_data = get_PosNegNeurons_dict(i,predictions,lrp_neurons) 
         review_filename, _ = get_NeuronExcitingWords_dict(lstm_hidden_json,kkeys,k,save_dir,topN)
         dstMat = get_DstMatrix_singleReview(save_dir+review_filename,test_data_json,k)
         neuronWords_jsons.append(review_filename)
         similarityMatrix_PerReview[k] = dstMat

         LRP[k] = lrp_input # contains LRP of input words
         totalLRP[k] = collections.OrderedDict(words=kkeys,lrp=lrp_fc) # contains LRP halfway through network, i.e. LRP of LSTM neurons
         
    neuron_types = get_NeuronType(reviewLRP_data,lrp_fc.shape[0])
    excitingWords_fullSet = get_MostExcitingWords_allReviews(save_dir,neuronWords_jsons,topN=5)
    similarityMatrix_AllReviews = get_NeuronSimilarity_AllReviews(excitingWords_fullSet)
    with open(save_dir+"exploratoryDataFull.pickle", 'wb') as f:
        pickle.dump((excitingWords_fullSet,similarityMatrix_AllReviews,similarityMatrix_PerReview,neuron_types,totalLRP,LRP), f)
    print("Saved auxiliary data dictionaries and distance matrices...")

    
    return LRP
         
