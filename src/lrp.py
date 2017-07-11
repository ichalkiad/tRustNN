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


def get_completeNeuronAct_data(save_dir,neuronWords_jsons):

    #neuron-word data dictionary
    nw_data = dict()

    for i in neuronWords_jsons:
        keys,data = data_format.get_data(save_dir+i)
        kkeys = list(keys)
        for j in kkeys:
            if j in list(nw_data.keys()):
               vals = list(data[j]) + list(set(list(nw_data[j]))-set(list(data[j])))
               nw_data[j] = vals
            else:
               nw_data[j] = data[j]

    return nw_data        
        
def neuron_value(test_data_json,review,neuron):
    
    keys_test,data_test = data_format.get_data(test_data_json)
    kkeys = list(data_test[review].keys())
    kdata = data_test[review]
    
    val = 0.0
    for w in neuron:
        val = val + np.array(list(kdata[w]))

    
    return np.array(val).sum()/len(neuron)


def neuron_distance(test_data_json,review,neuron1,neuron2):
# neuron1, neuron2 are given as a list of words that trigger them most
    
    n1_n2 = [item for item in neuron1 if item not in neuron2]
    n2_n1 = [item for item in neuron2 if item not in neuron1]

    if set(neuron1) == set(neuron2):
        return 0.0
    if (len(n1_n2)>=2 and len(n2_n1)>=2):
        a = neuron_value(test_data_json,review,neuron1)
        b = neuron_value(test_data_json,review,neuron2)
        return a-b
    else:
    #return arbitrary large value
        return 1000

    
def get_DstMatrix_singleReview(review_MaxAct_json,test_data_json,review):

    keys,data = data_format.get_data(review_MaxAct_json)
    kkeys = list(keys)
    dstMat = np.zeros((len(kkeys),len(kkeys)))
    for i in range((len(kkeys))):
        for j in range(i,len(kkeys)):
                 dstMat[i,j] = neuron_distance(test_data_json,review,list(data[kkeys[i]]),list(data[kkeys[j]]))
            
    return dstMat
                

def invert_dict_nonunique(d):
    newdict = {}
    for k in d:
        for v in d[k]:
            newdict.setdefault(v, []).append(k)
            
    return newdict


def get_reviewForwardMaxAct_cells(lstm_hidden_json,kkeys,k,save_dir,topN=5):

     d = collections.OrderedDict()
     keys_hidden,data_hidden = data_format.get_data(lstm_hidden_json)
     kdata = data_hidden[k]

     for i in range(len(kkeys)):
          ord_cells = np.argsort(kdata[i,:],axis=0,kind='quicksort')
          d[kkeys[i]] = ord_cells[-(topN+1):-1].tolist()

     NtoW = invert_dict_nonunique(d)
     
     with open(save_dir+re.sub('/', '_', k[37:-4])+"_ActCells.json", 'w') as f:
            json.dump(NtoW, f)

     return re.sub('/', '_', k[37:-4])+"_ActCells.json",NtoW


 
def get_reviewRelevant_cells(lrp_fc,review,save_dir,topN=5):
     
     sorted_LRP = np.argsort(lrp_fc,axis=0,kind='quicksort')
     idx = sorted_LRP[-(topN+1):-1].tolist()
     
     with open(save_dir+re.sub('/', '_', review[37:-4])+"_lrpCells.json", 'w') as f:
            json.dump({review:idx}, f)
            
    
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

   
def load_intermediate_outputs(input_filename,fc_json,lstm_hidden_json,lstm_cell_json,layer_name=None):
#layer_name is currently not needed - for networks with more layers we will need it, as the json structure will change

     
     keys_hidden,data_hidden = data_format.get_data(lstm_hidden_json)
     keys_cell,data_cell = data_format.get_data(lstm_cell_json)
     keys_fc,data_fc = data_format.get_data(fc_json)
          
     lstm_hidden = data_hidden[input_filename]
     lstm_cell   = data_cell[input_filename]
     fc_out      = data_fc[input_filename]
     
     d = lstm_cell.shape[1]
     
     return fc_out,lstm_hidden,lstm_cell,d



def lrp_single_input(model,layer_names,input_filename,single_input_data,eps,delta,fc_json,lstm_hidden_json,lstm_cell_json,target_class,T,classes=2,lstm_actv1=expit,lstm_actv2=np.tanh,debug=False):

        
    with model.session.as_default():

        lrp_mask = np.zeros((classes))
        lrp_mask[target_class] = 1.0

        fc_out,lstm_hidden,lstm_cell,d = load_intermediate_outputs(input_filename,fc_json,lstm_hidden_json,lstm_cell_json,layer_name=None)
        
        #LRP through fc layer
        fc_name = "fc"
        lrp_fc = lrp_fullyConnected(model,fc_name,lstm_hidden,fc_out,lrp_mask,d,T,classes,eps,delta,debug)

        #LRP through lstm layer
        lstm_name = "lstm"        
        feed = single_input_data
        lstm_lrp_x,(lstm_lrp_h,lstm_lrp_g,lstm_lrp_c) = lrp_lstm(model,lstm_name,feed,T,d,lrp_fc,lstm_hidden,lstm_cell,lstm_actv1,lstm_actv2,eps,delta,debug)

        

    return lrp_fc,lstm_lrp_x,(lstm_lrp_h,lstm_lrp_g,lstm_lrp_c)


        
def lrp_full(model,input_filename,net_arch,net_arch_layers,test_data_json,fc_out_json,lstm_hidden_json,lstm_cell_json,eps,delta,save_dir,lstm_actv1=expit,lstm_actv2=np.tanh,topN=5,debug=False):

    LRP = dict()
    neuronWords_jsons = []
    
    keys_test,data_test = data_format.get_data(test_data_json)
    for k in keys_test:
         kkeys = list(data_test[k].keys())
         kdata = np.array(list(data_test[k].values()))
         T = kdata.shape
         
         lrp_fc,lstm_lrp_x,(lstm_lrp_h,lstm_lrp_g,lstm_lrp_c) = lrp_single_input(model,net_arch_layers,k,kdata,eps,delta,fc_out_json,lstm_hidden_json,lstm_cell_json,target_class=1,T=T,classes=2,lstm_actv1=expit,lstm_actv2=np.tanh,debug=debug)

         
         get_reviewRelevant_cells(lrp_fc,k,save_dir,topN)
         review_filename, _ = get_reviewForwardMaxAct_cells(lstm_hidden_json,kkeys,k,save_dir,topN)
         neuronWords_jsons.append(review_filename)
         
         w = dict(words=kkeys,scores=np.sum(lstm_lrp_x,axis=1))
         LRP[k] = w

    neuronWords_data_fullTestSet = get_completeNeuronAct_data(save_dir,neuronWords_jsons)
    with open(save_dir+"neuronWords_data_fullTestSet.pickle", 'wb') as f:
        _pickle.dump((neuronWords_data_fullTestSet,neuronWords_jsons), f)

    
    return LRP
         
