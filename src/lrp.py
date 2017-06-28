import tensorflow as tf
import numpy as np
import collections
import tflearn
import json


def get_fwd_pass_single_input(model,layer_names,layer_outputs,single_input_data,input_file):


     prediction_score = model.predict(single_input_data)
        predicted_label  = model.predict_label(single_input_data)
        

    return #arrays with layer inputs + model prediction score and label (dict("layer":layer_data), pred_score,pred_label)


def lrp_single_input(model,layer_names,layer_sizes,fwdPass_data,input_file,eps,delta): #layer_sizes: dict with "layer":size of weight matrix, eg (30,2)

    with model.session.as_default():
       
        lrp_out = np.zeros((2)) #2 classes
        lrp_out[predicted_label] = prediction_score

        #LRP through fc layer
        l = "fc"
        layer = tflearn.variables.get_layer_variables_by_name(l)
        W = model.get_weights(layer[0])
        b = model.get_weights(layer[1])
        
        zi = fwdPass_data[?]  #lstm layer before fc layer  
        zj = fwdPass_data[l]
        
        zi_size, zj_size = layer_sizes[l]
        fc_lrp = []
        for i in range(zi_size):
            lrp_zij = 0
            for j in range(zj_size):   #  check division operator !!!
                lrp_zij = lrp_zij + (zi[i]*W[i][j] + (eps*np.sign(zj[j]) + delta*b[j])/zi_size)*lrp_out[j]/(zj[j] + eps*np.sign(zj[j]))
            fc_lrp.append(lrp_zij)
        lrp_fc = np.array(fc_lrp)

        
        #LRP through lstm layer
        l = "lstm"
        layer = tflearn.variables.get_layer_variables_by_name(l)
        input, new_input, forget, output = tf.split(layer[0], num_or_size_splits=4, axis=1)
        input_arr = model.session.run(input)
        new_input_arr = model.session.run(new_input)
        forget_arr    = model.session.run(forget)
        output_arr    = model.session.run(output)

        zi = fwdPass_data[?]  #layer before current lstm layer  
        zj = fwdPass_data[l]  # OUTPUT of current lstm layer
        
        zi_size, zj_size = layer_sizes[l]
        lstm_lrp = []
        
