from bokeh.models import ColumnDataSource, HoverTool, Range1d, Plot, LinearAxis, Grid, Paragraph,TapTool,Div
from bokeh.plotting import figure, show, output_file
from bokeh.io import curdoc
from bokeh.layouts import widgetbox , layout
from bokeh.models.widgets import Select, Slider
import dim_reduction
import numpy as np
import clustering
import random
import sys
import os
import pickle
from bokeh.models.glyphs import ImageURL
import re
from bokeh.models.callbacks import CustomJS

src_path = os.path.abspath("./src/")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
import data_format
from wcloud_standalone import get_wcloud
import heatmap as hmap



def tap_callback(attrname,old,new):

    print(attrname)
    print(old)   
    #neuronExcitingWords_AllReviews
    #hmap.html_heatmap(ws,scs,text_banner.text)


def get_wc_colourGroups(rawInput_source):

    words  = rawInput_source.data['w']
    colors = rawInput_source.data['z']
    color_dict = dict()
    
    for color in sorted(data_format.list_duplicates(colors)):
        color_dict[color[0]] = list(words[color[1]])

    return color_dict

def get_selections(keys):
    
    gates = ["IN - what to add on","NOT IMPORTANT - what to drop off","IMPORTANT - where to focus on"]
    select_gate = Select(title="Gate", value="IN - what to add on", options=gates)

    if select_gate.value == "IN - what to add on":
        select_gate.value = "input_gate"
    elif select_gate.value == "NOT IMPORTANT - what to drop off":
        select_gate.value = "forget_gate"
    elif select_gate.value == "IMPORTANT - where to focus on":
        select_gate.value = "output_gate"

    return select_gate


def get_clustering_selections(algorithms_neurons):

    algorithm_select_neuron = Select(value="KMeans - selected gate",title="Select clustering option for neurons:",width=250, options=algorithms_neurons)
    cluster_slider = Slider(title="Number of clusters (use in kmeans,hierarchical clustering)",value=2.0,start=2.0,end=4.0,step=1,width=400)

    return (algorithm_select_neuron,cluster_slider)

def get_rawInput_selections():

    review = [r for r in keys_raw]
    select_rawInput = Select(title="Input text", value=review[0], options=review)

    return select_rawInput

def get_projection_selections(algorithms):

    algorithm_select = Select(value="PCA",title="Select projection algorithm:",width=250, options=algorithms)
    
    return algorithm_select


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
   
    gate_value  = gate_selections.value
    if gate_value == "IN - what to add on":
        gate_value = "input_gate"
    elif gate_value == "NOT IMPORTANT - what to drop off":
        gate_value = "forget_gate"
    elif gate_value == "IMPORTANT - where to focus on":
        gate_value = "output_gate"
    
    x = data[lstm_layer_name][gate_value]

    #update dimension reduction source
    algorithm = projection_selections.value
    knn = 10
    x_pr,performance_metric = dim_reduction.project(x, algorithm, knn, labels)

    #update clustering 
    algorithm_cl_neurons = clustering_selections[0].value
    n_clusters = int(clustering_selections[1].value)

    if algorithm_cl_neurons=="Internal state clustering (LSTM's outputs)":
        text_set.text = "Internal state clustering - selected review: Clusters representation of input review at every timestep as learned by the LSTM layer." 
        lstm_hidVal = np.array(lstm_hidden[rawInput_selections.value])
        x_pr,performance_metric = dim_reduction.project(np.transpose(lstm_hidVal), algorithm, knn, labels)
        cluster_labels, colors, cl_spectral = clustering.apply_cluster(data=np.transpose(lstm_hidVal),algorithm=algorithm_cl_neurons,n_clusters=n_clusters,review=None,neuronData=None,mode="nn")
        w = [i for i in range(lstm_hidVal.shape[0])]
    elif algorithm_cl_neurons=="DBSCAN - all reviews" or algorithm_cl_neurons== "AgglomerativeClustering - all reviews":
        if algorithm_cl_neurons=="DBSCAN - all reviews":
            text_set.text = "DBSCAN - all reviews: Clusters neurons based on how related their most activating words are. List of activating words generated from all reviews."
        elif algorithm_cl_neurons== "AgglomerativeClustering - all reviews":
            text_set.text = "AgglomerativeClustering - all reviews: Hierarchical clustering of neurons based on how related their most activating words are. List of activating words generated from all reviews."
        neuronData = similarityMatrix_AllReviews
        cluster_labels, colors, cl_spectral = clustering.apply_cluster(x,algorithm_cl_neurons,n_clusters,review=rawInput_selections.value,neuronData=neuronData,mode="nn")
        w = neuronExcitingWords_AllReviews
    elif algorithm_cl_neurons=="Positive-Negative neuron clustering (LSTM's predictions)":
        text_set.text = "Positive-Negative neuron clustering: Clusters neurons based on how much they contributed to classifying the review as positive or negative."
        neuronData = neuron_types
        cluster_labels, colors, cl_spectral = clustering.apply_cluster(x,algorithm_cl_neurons,n_clusters,review=rawInput_selections.value,neuronData=neuronData,mode="nn")
        w = neuronExcitingWords_AllReviews
    else:
        if algorithm_cl_neurons=="KMeans - selected gate":
            text_set.text = "KMeans: Clusters neurons based on their gate values after training."
        elif algorithm_cl_neurons=="DBSCAN - selected review":
            text_set.text = "DBSCAN - selected review: Clusters neurons based on how related their most activating words are. List of activating words generated from seleceted review."
        neuronData = similarityMatrix_PerReview
        cluster_labels, colors, cl_spectral = clustering.apply_cluster(x,algorithm_cl_neurons,n_clusters,review=rawInput_selections.value,neuronData=neuronData,mode="nn")
        w = neuronExcitingWords_AllReviews
   
    proj_source.data = dict(x=x_pr[:, 0], y=x_pr[:, 1], z=colors)
    if performance_metric!=(None,None):
        project_plot.title.text = algorithm + performance_metric[0] + performance_metric[1]
    else:
        project_plot.title.text = algorithm
       
    #update raw input
    text_src = re.sub('/home/icha/','/home/yannis/Desktop/tRustNN/',rawInput_selections.value)
    text_banner.text = open(text_src,"r").read()
    label_banner.text = "Network decision : POSITIVE" if predicted_tgs[list(keys_raw).index(rawInput_selections.value)][1] == 1 else "Network decision : NEGATIVE"

    text_data,text_words = get_rawText_data(rawInput_selections.value,keys_raw,data_raw)
    w2v_labels, w2v_colors, w2v_cl_spectral = clustering.apply_cluster(text_data,"KMeans - selected gate",n_clusters,mode="wc")
    rawInput_source.data = dict(z=w2v_colors, w=text_words)
    color_dict = get_wc_colourGroups(rawInput_source)
    if gate_value=="input_gate":
        wc_filename,wc_img,wc_words = get_wcloud(LRP,rawInput_selections.value,load_dir,color_dict=color_dict,gate="in",text=text_banner.text)
    elif gate_value=="forget_gate":
        wc_filename,wc_img,wc_words = get_wcloud(LRP,rawInput_selections.value,load_dir,color_dict=color_dict,gate="forget")
    elif gate_value=="output_gate":
        wc_filename,wc_img,wc_words = get_wcloud(LRP,rawInput_selections.value,load_dir,color_dict=color_dict,gate="out")

    wc_plot.add_glyph(img_source, ImageURL(url=dict(value=load_dir+wc_filename), x=0, y=0, anchor="bottom_left"))




    

"""
------------------------------------------------------------------------------------------------------------------------
                   MAIN APP CODE
------------------------------------------------------------------------------------------------------------------------
"""


# Provide data paths and files 
load_dir = "./bokeh_vis/static/"
lstm_layer_name = "lstm"
#Get trained model parameters: weights and gate values
keys,data = data_format.get_data(load_dir+"model.json")
#Get raw input
keys_raw,data_raw = data_format.get_data(load_dir+"test_data_input.pickle")

#Load auxiliary data
with open(load_dir+"lstm_predictions.pickle","rb") as handle:
    predicted_tgs = pickle.load(handle)
with open(load_dir+"exploratoryDataFull.pickle", 'rb') as f:
    excitingWords_fullSet,similarityMatrix_AllReviews,similarityMatrix_PerReview,neuron_types,totalLRP,LRP = pickle.load(f)

neuronExcitingWords_AllReviews = list((excitingWords_fullSet.values()))
_,lstm_hidden = data_format.get_data(load_dir+"test_model_internals_lstm_hidden.pickle")

#Get preset buttons' selections
        
#LSTM gates
gate_selections = get_selections(keys)
#Dimensionality reduction
projection_selections = get_projection_selections(dim_reduction.get_dimReduction_algorithms())
#Clustering
algorithm_neurons = clustering.get_cluster_algorithms()
clustering_selections = get_clustering_selections(algorithm_neurons)
#Raw input clustering
rawInput_selections = get_rawInput_selections()

tools = "pan,wheel_zoom,box_zoom,reset,tap"

#Dimensionality reduction
labels = None 
data_pr = data[lstm_layer_name][gate_selections.value]
X, performance_metric = dim_reduction.project(data_pr, "PCA", n_neighbors=10, labels=labels)
X_cluster_labels, X_colors, X_cl_spectral = clustering.apply_cluster(data_pr,algorithm=clustering_selections[0].value,n_clusters=int(clustering_selections[1].value),mode="nn")
proj_source = ColumnDataSource(dict(x=X[:,0],y=X[:,1],z=X_colors))
project_plot = figure(title=projection_selections.value + performance_metric[0] + performance_metric[1],tools=tools,plot_width=300, plot_height=300)
scatter_tap = project_plot.scatter('x', 'y', marker='circle', size=10, fill_color='z', alpha=0.5, source=proj_source, legend=None)
project_plot.xaxis.axis_label = 'Dim 1'
project_plot.yaxis.axis_label = 'Dim 2'
taptool = project_plot.select(dict(type=TapTool))[0]

#Input text
text_data,text_words = get_rawText_data(rawInput_selections.value,keys_raw,data_raw)
w2v_labels, w2v_colors, w2v_cl_spectral = clustering.apply_cluster(text_data,algorithm="KMeans - selected gate",n_clusters=int(clustering_selections[1].value),mode="wc")
rawInput_source = ColumnDataSource(dict(z=w2v_colors,w=text_words))


text_src = re.sub('/home/icha/','/home/yannis/Desktop/tRustNN/',rawInput_selections.value)
text_banner = Div(text=open(text_src,"r").read(), width=1300, height=100)
label_banner = Paragraph(text="Network decision : POSITIVE" if predicted_tgs[list(keys_raw).index(rawInput_selections.value)][1] == 1 else "Network decision : NEGATIVE", width=200, height=30)


#WordCloud
color_dict = get_wc_colourGroups(rawInput_source)
wc_filename,wc_img,wc_words = get_wcloud(LRP,rawInput_selections.value,load_dir,color_dict=color_dict,gate="in",text=text_banner.text)
#if wc from "out" gate
words_to_be_highlighted = [i for i in wc_words and totalLRP[rawInput_selections.value]['words']]

proj_source.data = dict(x=X[:,0],y=X[:,1],z=X_colors)
img_source = ColumnDataSource(dict(url = [load_dir+wc_filename]))
xdr = Range1d(start=0, end=600)
ydr = Range1d(start=0, end=600)
wc_plot = Plot(title=None, x_range=xdr, y_range=ydr, plot_width=500, plot_height=550, min_border=0)
image = ImageURL(url=dict(value=load_dir+wc_filename), x=0, y=0, anchor="bottom_left", retry_attempts=3, retry_timeout=2500)
wc_plot.add_glyph(img_source, image)


text_0 = Paragraph(text="Clustering option:", width=200, height=20)
text_set = Paragraph(text="KMeans: Clusters neurons based on their gate values after training.", width=250, height=100)


#Layout
gate_selections.on_change('value', update_source)
projection_selections.on_change('value', update_source)
for attr in clustering_selections:
    attr.on_change('value', update_source)
rawInput_selections.on_change('value', update_source)
scatter_tap.data_source.on_change('selected', tap_callback)


gp = layout([project_plot, wc_plot, widgetbox(gate_selections,projection_selections,rawInput_selections,clustering_selections[0],clustering_selections[1],text_0,text_set,label_banner)],
            [text_banner],
            responsive=True)
curdoc().add_root(gp)
curdoc().title = "tRustNN"



