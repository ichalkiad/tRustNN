from scipy.special import expit
import tensorflow as tf
import numpy as np
import collections
import tflearn
import json


def get_gate_out(W,b,in_concat):
# in_concat is concatenation(current_input, input_prev_timestep)
# b should correspond to specific gate bias
     
     return np.dot(W.transpose(),in_concat) + b




def get_cell_state_t(in_concat,b_tot,i_arr,f_arr,g_arr,o_arr,d,lstm_actv1,lstm_actv2,c_prev):

     
     i_t = lstm_actv1(get_gate_out(i_arr,b[0:d],in_concat))
     g_t = lstm_actv2(get_gate_out(g_arr,b[d:2*d],in_concat))
     f_t = lstm_actv1(get_gate_out(f_arr,b[2*d:3*d],in_concat))
     o_t = lstm_actv1(get_gate_out(o_arr,b[3*d:4*d],in_concat))

     c_t = f_t*c_prev + i_t*g_t

     return c_t,i_t,g_t,f_t




def get_intermediate_outputs(model,layer_outputs,feed):

    internals = collections.OrderedDict()
    with model.session.as_default():
            
        keys = [k for k in list(layer_outputs.keys()) if "lstm" in k]
        for k in keys:
            if "output" in k:
                if isinstance(layer_outputs[k],list):
                    currentStepOutput = []
                    for history in layer_outputs[k]:
                        currentStepOutput.append(history.eval(feed_dict={'InputData/X:0':feed}))
                    totalOut = np.array(currentStepOutput)
                    sh = totalOut.shape
                    internals[k] = totalOut.reshape((sh[0],sh[2])) 

            if "cell" in k:
                internals[k] = layer_outputs[k][0].eval(feed_dict={'InputData/X:0':feed})

    return internals





def lrp_rule(zi,zi_size,zj,zj_size,W,b,eps,delta,Rout,N):

     _lrp = []
     for i in range(zi_size):
         lrp_zij = 0
         for j in range(zj_size):
             lrp_zij = lrp_zij + (zi[i]*W[i][j] + (eps*np.sign(zj[j]) + delta*b[j])/N)*Rout[j]/(zj[j] + eps*np.sign(zj[j]))
         _lrp.append(lrp_zij)
     lrp_ = np.array(_lrp)

     return lrp_
     




def get_fwd_pass_single_input(model,layer_outputs,single_input_data):

    # single_input_data : array with the embeddings of 1 review
     
    prediction_score = model.predict(single_input_data)
    predicted_label  = model.predict_label(single_input_data)
    T = shape(single_input_data)
    data = get_intermediate_outputs(model,layer_outputs,single_input_data)
    
    return prediction_score,prediction_label,T,data,single_input_data





def lrp_fullyConnected(fc_name,zi,zi_size,zj,zj_size,eps,delta,Rout,N):

        layer = tflearn.variables.get_layer_variables_by_name(fc_name)
        W = model.get_weights(layer[0])
        b = model.get_weights(layer[1])
        lrp_fc = lrp_rule(zi,zi_size,zj,zj_size,W,b,eps,delta,Rout,N)

        return lrp_fc



   

"""


fwdPass_data = get_fwd_pass_single_input(model,layer_outputs,single_input_data)  #####call before lrp_single_out  !!!!!!!!!
def lrp_single_input(model,layer_outputs,layer_names,layer_sizes,fwdPass_data,input_file,eps,delta,lstm_actv1=expit,lstm_actv2=np.tanh):

#layer_sizes: dict with "layer":size of weight matrix, eg (30,2)

    with model.session.as_default():

        T = fwdPass_data[2]
        lrp_out = np.zeros((2)) #2 classes
        lrp_out[fwdPass_data[1]] = fwdPass_data[0]

        #LRP through fc layer
        fc_name = "fc"
        zi = fwdPass_data[?]  #lstm layer before fc layer  
        zj = fwdPass_data[l]
        zi_size, zj_size = layer_sizes[l] ????? ### Re-examine
        N = ????? ### Re-examine
        lrp_fc = lrp_fullyConnected(fc_name,zi,zi_size,zj,zj_size,eps,delta,Rout,N)


        
        #LRP through lstm layer
        l = "lstm"
        lstm_outputs = fwdPass_data[3]["lstm_output"] 
        lstm_final_cell_state = fwdPass_data[3]["lstm_cell_state"]
        feed = fwdPass_data[-1]
        
        layer = tflearn.variables.get_layer_variables_by_name(l)
        input, new_input, forget, output = tf.split(layer[0], num_or_size_splits=4, axis=1)
        i_arr = model.session.run(input)
        g_arr = model.session.run(new_input) 
        f_arr = model.session.run(forget)
        o_arr = model.session.run(output)
        b_tot = model.get_weights(layer[1])

        
        zi = fwdPass_data[?]  #layer before current lstm layer  
        zj = fwdPass_data[l]  # OUTPUT of current lstm layer

        #hidden output size
        d = i_arr.shape[0] - T[1]
   
        zi_size, zj_size = layer_sizes[l]
        lstm_lrp_x = np.zeros(fwdPass_data)
        lstm_lrp_h = np.zeros((T[0]+1, d))
        lstm_lrp_c = np.zeros((T[0]+1, d))
        lstm_lrp_g = np.zeros((T[0], d))

        
        #verify that initial cell/hidden state is zero if not set by user
        lstm_lrp_h[T[0]-1] = np.sum(lrp_fc)
        c_next = lstm_final_cell_state
        
        for t in reversed(range(T[0])):

             lstm_lrp_c[t] += lstm_lrp_h[t]

             x_t = feed[t]
             h_t = lstm_outputs[t-1]
             in_concat = np.concatenate((x_t,h_t),axis=0)
             c_t,i_t,g_t,f_t,c_prev = get_cell_state_t(in_concat,b_tot,i_arr,f_arr,g_arr,o_arr,d,lstm_actv1,lstm_actv2,c_next) 
             c_next = c_t
             zi = 
             zi_size =
             zj = 
             zj_size = 
             W = np.identity(d)
             b = np.zeros((d))
             Rout = 
             N = 2*d
             lstm_lrp_c[t-1] = lrp_rule()

             zi =
             zi_size = 
             lstm_lrp_g[t-1] = lrp_rule

             zi = 
             zi_size =
             zj = 
             zj_size = 
             W = 
             b =
             Rout = lstm_lrp_g
             N = d + T[1]
             lstm_lrp_x = lrp_rule()
             
             zi =
             zi_size =
             W =
             lstm_lrp_h = lrp_rule()



        



"""
