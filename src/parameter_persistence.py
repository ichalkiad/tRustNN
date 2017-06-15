import tensorflow as tf
import tflearn
import json


def export_serial_lstm_data(model,layer_outputs,feed,data="lstm",save_dir="/tmp/"):
# data="lstm" for LSTM data or "all" for LSTM + FC layer data
    
    internals = dict()

    with model.session.as_default():
            
        keys = [k for k in list(layer_outputs.keys()) if "lstm" in k]
        for k in keys:
            if "output" in k:
                internals[k] = layer_outputs[k].eval(feed_dict={'InputData/X:0':feed}).tolist()
            if "cell" in k:
                internals[k] = {"cell": layer_outputs[k][0].eval(feed_dict={'InputData/X:0':feed}).tolist(), "hidden":layer_outputs[k][1].eval(feed_dict={'InputData/X:0':feed}).tolist()}
                    
            if data == "all":
                keys = [k for k in list(layer_outputs.keys()) if "fc" in k]
                for k in keys:
                    internals[k] = layer_outputs[k].eval(feed_dict={'InputData/X:0':feed}).tolist()
            
    with open(save_dir+"model_internals.json", 'w') as f:
         json.dump(internals, f)


def export_serial_model(model,layer_names,save_dir):

    network = dict()
    
    for l in layer_names:
        if "output" in l:
            break
        network[l] = dict()
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
   
   
