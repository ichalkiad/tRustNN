import tensorflow as tf
import numpy as np
import collections
import tflearn
import json
#import _pickle

def export_serial_lstm_data(model,layer_outputs,feed,input_files,data="lstm",save_dir="/tmp/",save_mode="json"):
# data="lstm" for LSTM data or "all" for LSTM + FC layer data
# input_files has to be a list even if it is a single file
    
    lstm_outputs = collections.OrderedDict()
    lstm_states  = collections.OrderedDict()
    lstm_hidden  = collections.OrderedDict()  ## REDUNDANT? SAME AS OUTPUTS? yes, outputs includes the unncessary zeros at the end
    fc_outputs   = collections.OrderedDict()
    
    with model.session.as_default():
            
        keys = [k for k in list(layer_outputs.keys()) if "lstm" in k]
        for k in keys:
            """
            if "output" in k:
                if isinstance(layer_outputs[k],list):
                    currentStepOutput = []
                    for history in layer_outputs[k]:
                        currentStepOutput.append(history.eval(feed_dict={'InputData/X:0':feed}))
                    totalDataOutput = np.stack(currentStepOutput,axis=0)
                    for i in range(len(input_files)):
                        lstm_outputs[input_files[i]] = totalDataOutput[:,i,:].tolist()
                else:
                    lstm_outputs[input_files[0]] = layer_outputs[k].eval(feed_dict={'InputData/X:0':feed}).tolist()
            """
            if "cell" in k:
                if isinstance(layer_outputs[k],list):
                    currentStepOutput = []
                    for history_state in layer_outputs[k]:
                        currentStepOutput.append(history_state[0].eval(feed_dict={'InputData/X:0':feed}))
                    totalDataOutput_state = np.stack(currentStepOutput,axis=0)
                    currentStepOutput = []
                    for history_hidden in layer_outputs[k]:
                        currentStepOutput.append(history_hidden[1].eval(feed_dict={'InputData/X:0':feed}))
                    totalDataOutput_hidden = np.stack(currentStepOutput,axis=0)
                    for i in range(len(input_files)):
                        lstm_states[input_files[i]] = totalDataOutput_state[:,i,:].tolist()
                    for i in range(len(input_files)):
                        lstm_hidden[input_files[i]] = totalDataOutput_hidden[:,i,:].tolist()
                else:
                    lstm_states[input_files[0]] = layer_outputs[k][0].eval(feed_dict={'InputData/X:0':feed}).tolist()
                    lstm_hidden[input_files[0]] = layer_outputs[k][1].eval(feed_dict={'InputData/X:0':feed}).tolist()
                    
        if data == "all":
           keys = [k for k in list(layer_outputs.keys()) if "fc" in k]
           for k in keys:
               data_out = layer_outputs[k].eval(feed_dict={'InputData/X:0':feed})
               for i in range(len(input_files)):
                   fc_outputs[input_files[i]] = data_out[i,:].tolist()
               
    if save_mode=="json":
        with open(save_dir+"model_internals_fc.json", 'w') as f:
            json.dump(fc_outputs, f)
        with open(save_dir+"model_internals_lstm_outputs.json", 'w') as f:
            json.dump(lstm_outputs, f)
        with open(save_dir+"model_internals_lstm_hidden.json", 'w') as f:
            json.dump(lstm_hidden, f)
        with open(save_dir+"model_internals_lstm_states.json", 'w') as f:
            json.dump(lstm_states, f)
    elif save_mode=="pickle":
        with open(save_dir+"model_internals_fc.pickle", 'wb') as f:
            _pickle.dump(fc_outputs, f)
        with open(save_dir+"model_internals_lstm_outputs.pickle", 'wb') as f:
            _pickle.dump(lstm_outputs, f)
        with open(save_dir+"model_internals_lstm_hidden.pickle", 'wb') as f:
            _pickle.dump(lstm_hidden, f)
        with open(save_dir+"model_internals_lstm_states.pickle", 'wb') as f:
            _pickle.dump(lstm_states, f)



def export_serial_model(model,layer_names,save_dir):

    network = collections.OrderedDict()
    
    for l in layer_names:
        if "output" in l:
            break
        network[l] = collections.OrderedDict()
        layer = tflearn.variables.get_layer_variables_by_name(l)
        if "lstm" in l:
            data_arr = model.get_weights(layer[0])
            #Each of the following matrices: (embedding_size + hidden_layer_size) x LSTM_cell_number , first part coming from concatenating new_input + output from previous cell
            input, new_input, forget, output = tf.split(layer[0], num_or_size_splits=4, axis=1)
            input_arr     = model.session.run(input).tolist()
            network[l]['input_gate'] = input_arr
            # New_input_arr is needed if we want to restore network's exact state when we stopped training
            new_input_arr = model.session.run(new_input).tolist()
            network[l]['new_input'] = new_input_arr
            forget_arr    = model.session.run(forget).tolist()
            network[l]['forget_gate'] = forget_arr
            output_arr    = model.session.run(output).tolist()
            network[l]['output_gate'] = output_arr
        if "fc" in l:
            data_arr = model.get_weights(layer[0])
            network[l]['W'] = data_arr.tolist()
        
        data_arr = model.get_weights(layer[1])
        network[l]['b'] = data_arr.tolist()


    
    with open(save_dir+"model.json", 'w') as f:
         json.dump(network, f)
   
   
