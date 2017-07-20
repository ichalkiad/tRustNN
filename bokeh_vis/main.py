from bokeh.models import ColumnDataSource, HoverTool, Range1d, Plot, LinearAxis, Grid, Paragraph
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
import pickle
from bokeh.models.glyphs import ImageURL

src_path = os.path.abspath("./src/")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
import data_format
from wcloud_standalone import get_wcloud


def get_mostActiveWords(neuronWords_data_fullTestSet):

    words = []
    for k in range(len(list(neuronWords_data_fullTestSet.keys()))):
        if str(k) in list(neuronWords_data_fullTestSet.keys()):
            vals = list(neuronWords_data_fullTestSet[str(k)])
            if len(vals)>5:
                words.append(vals[0]+","+vals[1]+","+vals[2]+","+vals[3]+","+vals[4])
            else:
                for i in range(len(vals)):
                    words.append(vals[i]+",")
                    
    return words

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


def get_clustering_selections(algorithms_neurons,algorithms_data):

    algorithm_select_neuron = Select(value="MiniBatchKMeans - selected gate",title="Select clustering option for neurons:",width=200, options=algorithms_neurons)
    algorithm_select_data = Select(value="MiniBatchKMeans",title="Select clustering option for raw data:",width=200, options=algorithms_data)
    cluster_slider = Slider(title="Number of clusters (use in kmeans,hierarchical clustering)",value=2.0,start=2.0,end=10.0,step=1,width=400)

    return (algorithm_select_neuron,algorithm_select_data,cluster_slider)

def get_rawInput_selections():

    review = [r for r in keys_raw]
    select_rawInput = Select(title="Input text", value=review[0], options=review)

    return select_rawInput

def get_projection_selections(algorithms):

    algorithm_select = Select(value="PCA",title="Select projection algorithm:",width=200, options=algorithms)
    knn_slider = Slider(title="Number of neighbors",value=5.0,start=5.0,end=30.0,step=1,width=200)
    
    return (algorithm_select,knn_slider)


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


"""
-------------------------------------------------------------------------------------------------------
                                  UPDATE SOURCE
-------------------------------------------------------------------------------------------------------
"""


def update_source(attrname, old, new):
    #As sources are currenty created, only 2-dim projections are visualized!!!!!!!

    layer_value = gate_selections[0].value
    gate_value  = gate_selections[1].value
    
    x = data[layer_value][gate_value]

    #update dimension reduction source
    algorithm = projection_selections[0].value
    knn = int(projection_selections[1].value)
    x_pr,performance_metric = dim_reduction.project(x, algorithm, knn, labels)

    
    #update clustering 
    algorithm_cl_neurons = clustering_selections[0].value
    algorithm_cl_data = clustering_selections[1].value
    n_clusters = int(clustering_selections[2].value)

    if algorithm_cl_neurons=="DBSCAN - all reviews" or algorithm_cl_neurons== "AgglomerativeClustering - all reviews":
        neuronData = neuronWords_data_full
    elif algorithm_cl_neurons=="Positive-Negative neuron clustering (LSTM's predictions)":
        neuronData = posNeg_predictionLabel
    elif algorithm_cl_neurons=="Internal state clustering (LSTM's outputs)":
        neuronData = lstm_hidVal
    else:
        neuronData = neuronWords_data
    
    cluster_labels, colors, cl_spectral = clustering.apply_cluster(x,algorithm_cl_neurons,n_clusters,algorithm_data=algorithm_cl_data,review=rawInput_selections.value,neuronData=neuronData)
    
    proj_source.data = dict(x=x_pr[:, 0], y=x_pr[:, 1], z=colors,w=mostActiveWords)
    if performance_metric!=(None,None):
        project_plot.title.text = algorithm + performance_metric[0] + performance_metric[1]
    else:
        project_plot.title.text = algorithm

        
    #update raw input 
    text_data,text_words = get_rawText_data(rawInput_selections.value,keys_raw,data_raw)
    X_w2v, performance_metric_w2v = dim_reduction.project(text_data, algorithm, knn, labels=labels)
    w2v_labels, w2v_colors, w2v_cl_spectral = clustering.apply_cluster(text_data,"MiniBatchKMeans - selected gate",n_clusters,algorithm_data=algorithm_cl_data)
    rawInput_source.data = dict(x=X_w2v[:, 0], y=X_w2v[:, 1], z=w2v_colors, w=text_words)
    color_dict = get_wc_colourGroups(rawInput_source)
    wc_filename,wc_img = get_wcloud(LRP,rawInput_selections.value,load_dir,color_dict=color_dict)
    wc_plot.add_glyph(img_source, ImageURL(url=dict(value=load_dir+wc_filename), x=0, y=0, anchor="bottom_left"))

    text_banner.text = open(rawInput_selections.value,"r").read()
    label_banner.text = "POSITIVE" if predicted_tgs[list(keys_raw).index(rawInput_selections.value)][1] == 1 else "NEGATIVE"
        
"""
------------------------------------------------------------------------------------------------------------------------
                   MAIN APP CODE
------------------------------------------------------------------------------------------------------------------------
"""

load_dir = "./bokeh_vis/static/"

#Get trained model parameters: weights and gate values
keys,data = data_format.get_data(load_dir+"model.json")
#Get raw input
keys_raw,data_raw = data_format.get_data(load_dir+"test_data_input.json")

#Load auxiliary data
LRP=None
with open(load_dir+"LRP.pickle","rb") as handle:
    (LRP,predicted_tgs) = pickle.load(handle)
with open(load_dir+"neuronWords_data_fullTestSet.pickle", 'rb') as f:
    neuronWords_data_fullTestSet,neuronWords_data_full,neuronWords_data,posNeg_predictionLabel = pickle.load(f)
mostActiveWords = get_mostActiveWords(neuronWords_data_fullTestSet)
_,lstm_hidden = data_format.get_data(load_dir+"test_standalone_model_internals_lstm_hidden.json")
lstm_hidVal = np.vstack(np.array(list(lstm_hidden.values())))


#Get preset buttons selections
        
#LSTM gates
gate_selections = get_selections(keys)
gate_inputs = widgetbox(gate_selections[0],gate_selections[1])
#Dimensionality reduction
projection_selections = get_projection_selections(dim_reduction.get_dimReduction_algorithms())
projection_inputs = widgetbox(projection_selections[0],projection_selections[1])
#Clustering
algorithm_neurons,algorithm_data = clustering.get_cluster_algorithms()
clustering_selections = get_clustering_selections(algorithm_neurons,algorithm_data)
clustering_inputs = widgetbox(clustering_selections[0],clustering_selections[1],clustering_selections[2])
#Raw input clustering
rawInput_selections = get_rawInput_selections()
rawInput_inputs = widgetbox(rawInput_selections)


hover_input = HoverTool()
"""
<div>
            <img
                src="@imgs" height="42" alt="@imgs" width="42"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
"""
hover_input.tooltips = """
    <div>
        <div>
            <span style="font-size: 17px; font-weight: bold;">@w</span>
        </div>
        <div>
            <span style="font-size: 15px; color: #966;">Cell No. : $index</span>
        </div>
    </div>
    """

hover_input.mode = 'mouse'
tools = "pan,wheel_zoom,box_zoom,reset"


#Dimensionality reduction
labels = None # LOAD GROUND TRUTH OR NET-ASSIGNED LABELS??
data_pr = data[gate_selections[0].value][gate_selections[1].value]
X, performance_metric = dim_reduction.project(data_pr, "PCA", n_neighbors=10, labels=labels)
X_cluster_labels, X_colors, X_cl_spectral = clustering.apply_cluster(data_pr,algorithm=clustering_selections[0].value,n_clusters=int(clustering_selections[2].value),algorithm_data=clustering_selections[1].value)
proj_source = ColumnDataSource(dict(x=X[:,0],y=X[:,1],z=X_colors,w=mostActiveWords))
project_plot = figure(title=projection_selections[0].value + performance_metric[0] + performance_metric[1],tools=tools)
project_plot.scatter('x', 'y', marker='circle', size=10, fill_color='z', alpha=0.5, source=proj_source, legend=None)
project_plot.xaxis.axis_label = 'Dim 1'
project_plot.yaxis.axis_label = 'Dim 2'
project_plot.add_tools(hover_input)

#Input text
text_data,text_words = get_rawText_data(rawInput_selections.value,keys_raw,data_raw)
X_w2v, performance_metric_w2v = dim_reduction.project(text_data, "PCA", n_neighbors=10, labels=labels)
w2v_labels, w2v_colors, w2v_cl_spectral = clustering.apply_cluster(text_data,algorithm=clustering_selections[0].value,n_clusters=int(clustering_selections[2].value),algorithm_data=clustering_selections[1].value)
rawInput_source = ColumnDataSource(dict(x=X_w2v[:,0],y=X_w2v[:,1],z=w2v_colors,w=text_words))



#WordCloud
color_dict = get_wc_colourGroups(rawInput_source)
wc_filename,wc_img = get_wcloud(LRP,rawInput_selections.value,load_dir,color_dict=color_dict)
img_source = ColumnDataSource(dict(url = [load_dir+wc_filename]))
xdr = Range1d(start=0, end=400)
ydr = Range1d(start=0, end=400)
wc_plot = Plot(title=None, x_range=xdr, y_range=ydr, plot_width=500, plot_height=500, min_border=0)
image = ImageURL(url=dict(value=load_dir+wc_filename), x=0, y=0, anchor="bottom_left")
wc_plot.add_glyph(img_source, image)


text_banner = Paragraph(text=open(rawInput_selections.value,"r").read(), width=1000, height=200)
label_banner = Paragraph(text="POSITIVE" if predicted_tgs[list(keys_raw).index(rawInput_selections.value)][1] == 1 else "NEGATIVE", width=100, height=30)

#Layout
for attr in gate_selections:
    attr.on_change('value', update_source)
for attr in projection_selections:
    attr.on_change('value', update_source)
for attr in clustering_selections:
    attr.on_change('value', update_source)
rawInput_selections.on_change('value', update_source)


gp = gridplot([[project_plot, wc_plot],[row(gate_inputs,projection_inputs,rawInput_inputs)],[row(clustering_inputs,)],[row(text_banner,label_banner)]], responsive=True)
curdoc().add_root(gp)
curdoc().title = "tRustNN"



