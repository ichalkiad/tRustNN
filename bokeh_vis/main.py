from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook, show, curdoc
from bokeh.layouts import row, widgetbox, column, gridplot
from bokeh.models.widgets import Select, Slider
from sklearn.decomposition import PCA
import dim_reduction
import numpy as np
import clustering
import random
import sys
import os
import _pickle


src_path = os.path.abspath("./src/")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
import data_format
from wcloud_standalone import get_wcloud

def get_wc_colourGroups(rawInput_source):

    words  = rawInput_source.data['w']
    colors = rawInput_source.data['z']
    color_dict = dict()
    
    for color in sorted(data_format.list_duplicates(colors)):
        color_dict[color[0]] = list(words[color[1]])

    return color_dict

def get_selections(keys):
    
    lstm_layers = [l for l in keys if "lstm" in l]
    select_layer = Select(title="LSTM layer", value="lstm", options=lstm_layers)

    gates = ["input_gate","forget_gate","output_gate"]
    select_gate = Select(title="Gate", value="input_gate", options=gates)

    return (select_layer,select_gate)


def get_clustering_selections(algorithms):

    algorithm_select = Select(value='MiniBatchKMeans',title='Select clustering algorithm:',width=200, options=algorithms)
    cluster_slider = Slider(title="Number of clusters",value=2.0,start=2.0,end=10.0,step=1,width=400)

    return (algorithm_select,cluster_slider)

def get_rawInput_selections():

    review = [r for r in keys_raw]
    review.append("All")
    select_rawInput = Select(title="Input text", value="All", options=review)

    return select_rawInput

def get_projection_selections(algorithms):

    algorithm_select = Select(value='PCA',title='Select projection algorithm:',width=200, options=algorithms)
    knn_slider = Slider(title="Number of neighbors",value=5.0,start=5.0,end=30.0,step=1,width=200)
    dim_slider = Slider(title="Number of dimensions",value=2,start=2,end=3,step=1,width=200)
    
    return (algorithm_select,knn_slider,dim_slider)


def get_rawText_data(rawInput_selections,keys_raw,data_raw):

    data = []
    words = []
    if rawInput_selections == "All":
        for k in keys_raw:
            kkeys = data_raw[k].keys()
            for kk in kkeys:
                data.append(data_raw[k][kk])
                words.append(kk)
    else:
        kkeys = data_raw[rawInput_selections].keys()
        for k in kkeys:
            data.append(data_raw[rawInput_selections][k])
            words.append(k)

    return np.transpose(np.array(data)),np.array(words)


def update_source(attrname, old, new):
    #As sources are currenty created, only 2-dim projections are visualized!!!!!!!
    
    layer_value = gate_selections[0].value
    gate_value  = gate_selections[1].value

    x = data[layer_value][gate_value]

    #update dimension reduction source
    algorithm = projection_selections[0].value
    knn = int(projection_selections[1].value)
    dimensions = int(projection_selections[2].value)
    x_pr,performance_metric = dim_reduction.project(x, algorithm, knn, dimensions, labels)

    #update clustering 
    algorithm_cl = clustering_selections[0].value
    n_clusters = int(clustering_selections[1].value)
    cluster_labels, colors, cl_spectral = clustering.apply_cluster(x,algorithm_cl,n_clusters)
    rawInput_plot.title.text = algorithm_cl+" - "+rawInput_selections.value
    
    proj_source.data = dict(x=x_pr[:, 0], y=x_pr[:, 1], z=colors)
    if performance_metric!=(None,None):
        project_plot.title.text = algorithm + performance_metric[0] + performance_metric[1]
    else:
        project_plot.title.text = algorithm

    #update raw input 
    text_data,text_words = get_rawText_data(rawInput_selections.value,keys_raw,data_raw)
    X_w2v, performance_metric_w2v = dim_reduction.project(text_data, algorithm, knn, dimensions, labels=labels)
    w2v_labels, w2v_colors, w2v_cl_spectral = clustering.apply_cluster(text_data,algorithm_cl,n_clusters)
    rawInput_source.data = dict(x=X_w2v[:, 0], y=X_w2v[:, 1], z=w2v_colors, w=text_words)

    if LRP,rawInput_selections.value='All':
        color_dict = get_wc_colourGroups(rawInput_source)
        get_wcloud(LRP,rawInput_selections.value,wc_saveDir,color_dict=color_dict)


    
 
#Get trained model parameters: weights and gate values
keys,data = data_format.get_data("./bokeh_vis/data/model.json")
#Get raw input
keys_raw,data_raw = data_format.get_data("./bokeh_vis/data/test_data_input.json")

#Load LRP dictionary
LRP_pickle = '/home/yannis/Desktop/tRustNN/sacred_models/runIDstandalone/'
wc_saveDir = '/home/yannis/Desktop/'
with open(LRP_pickle+"LRP.pickle","rb") as handle:
    (LRP,predicted_tgs) = _pickle.load(handle)


#LSTM gates
gate_selections = get_selections(keys)
gate_inputs = widgetbox(gate_selections[0],gate_selections[1])
#Dimensionality reduction
projection_selections = get_projection_selections(dim_reduction.get_dimReduction_algorithms())
projection_inputs = widgetbox(projection_selections[0],projection_selections[1],projection_selections[2])
#Clustering
clustering_selections = get_clustering_selections(clustering.get_cluster_algorithms())
clustering_inputs = widgetbox(clustering_selections[0],clustering_selections[1])
#Raw input clustering
rawInput_selections = get_rawInput_selections()
rawInput_inputs = widgetbox(rawInput_selections)

hover = HoverTool()
hover.tooltips = [("Cell", "$index"),("(x,y)", "($x,$y)")]
hover.mode = 'mouse'
tools = "pan,wheel_zoom,box_zoom,reset,hover"

#Dimensionality reduction
labels = None # LOAD GROUND TRUTH OR NET-ASSIGNED LABELS??
data_pr = data[gate_selections[0].value][gate_selections[1].value]
X, performance_metric = dim_reduction.project(data_pr, 'PCA', n_neighbors=10, n_components=2, labels=labels)
cluster_labels, colors, cl_spectral = clustering.apply_cluster(data_pr,'MiniBatchKMeans',2)
proj_source = ColumnDataSource(dict(x=X[:,0],y=X[:,1],z=colors))

project_plot = figure(title=projection_selections[0].value + performance_metric[0] + performance_metric[1],tools=tools)
project_plot.scatter('x', 'y', marker='circle', size=10, fill_color='z', alpha=0.5, source=proj_source, legend=None)
project_plot.xaxis.axis_label = 'Dim 1'
project_plot.yaxis.axis_label = 'Dim 2'

#Input text
text_data,text_words = get_rawText_data(rawInput_selections.value,keys_raw,data_raw)
X_w2v, performance_metric_w2v = dim_reduction.project(text_data, 'PCA', n_neighbors=10, n_components=2, labels=labels)
w2v_labels, w2v_colors, w2v_cl_spectral = clustering.apply_cluster(text_data,'MiniBatchKMeans',2)
rawInput_source = ColumnDataSource(dict(x=X_w2v[:,0],y=X_w2v[:,1],z=w2v_colors,w=text_words))


hover_input = HoverTool()
hover_input.tooltips = [("Cell", "$index"),("(x,y)", "($x,$y)"),("Input word","@w")]
hover_input.mode = 'mouse'
tools_input = "pan,wheel_zoom,box_zoom,reset"

rawInput_plot = figure(title=clustering_selections[0].value+" - "+rawInput_selections.value,tools=tools_input)
rawInput_plot.scatter('x', 'y', marker='circle', size=10, fill_color='z', alpha=0.5, source=rawInput_source, legend=None)
rawInput_plot.add_tools(hover_input)

if LRP,rawInput_selections.value='All':
   color_dict = get_wc_colourGroups(rawInput_source)
   get_wcloud(LRP,rawInput_selections.value,wc_saveDir,color_dict=color_dict)




#Layout
for attr in gate_selections:
    attr.on_change('value', update_source)
for attr in projection_selections:
    attr.on_change('value', update_source)
for attr in clustering_selections:
    attr.on_change('value', update_source)
rawInput_selections.on_change('value', update_source)


#row(gate_inputs, projection_inputs, project_plot, clustering_inputs, cluster_plot, width=400)
gp = gridplot([[project_plot, rawInput_plot],[row(gate_inputs, projection_inputs,clustering_inputs, rawInput_inputs)]], responsive=True)
curdoc().add_root(gp)
curdoc().title = "tRustNN"



