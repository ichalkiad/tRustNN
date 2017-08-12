from bokeh.models import ColumnDataSource, HoverTool, Range1d, Plot, LinearAxis, Grid, Paragraph,TapTool,Div
from bokeh.plotting import figure, show, output_file
from bokeh.io import curdoc
from bokeh.layouts import widgetbox , layout
from bokeh.models.widgets import Select, Slider, Button
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
from sklearn.preprocessing import MinMaxScaler

src_path = os.path.abspath("./src/")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
import data_format
from wcloud_standalone import get_wcloud
import heatmap as hmap
from lrp import get_lrp_timedata


def button_callback():
    text_src = re.sub('/home/icha/','/home/yannis/Desktop/tRustNN/',rawInput_selections.value)
    text_banner.text = open(text_src,"r").read()

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

    #update raw input
    text_src = re.sub('/home/icha/','/home/yannis/Desktop/tRustNN/',rawInput_selections.value)
    text_banner.text = open(text_src,"r").read()
    text_banner2.text = open(text_src,"r").read()
    label_banner.text = "Network decision : POSITIVE" if predicted_tgs[list(keys_raw).index(rawInput_selections.value)][0] == 0 else "Network decision : NEGATIVE"

    #update dimension reduction source
    algorithm = projection_selections.value
    knn = 5
    x_pr,performance_metric = dim_reduction.project(x, algorithm, knn, labels)

    #update clustering 
    algorithm_cl_neurons = clustering_selections[0].value
    n_clusters = int(clustering_selections[1].value)

    if algorithm_cl_neurons=="Internal state clustering (LSTM's outputs)":
        text_set.text = "Internal state clustering - selected review: Clusters representation of input review at every timestep as learned by the LSTM layer." 
        lstm_hidVal = np.array(lstm_hidden[rawInput_selections.value])
        x_pr,performance_metric = dim_reduction.project(np.transpose(lstm_hidVal), algorithm, knn, labels)
        cluster_labels, colors, _ = clustering.apply_cluster(data=np.transpose(lstm_hidVal),algorithm=algorithm_cl_neurons,n_clusters=n_clusters,review=None,neuronData=None,mode="nn")
    
    elif algorithm_cl_neurons=="DBSCAN - all reviews" or algorithm_cl_neurons== "AgglomerativeClustering - all reviews":
        if algorithm_cl_neurons=="DBSCAN - all reviews":
            text_set.text = "DBSCAN - all reviews: Clusters neurons based on how related their most activating words are. List of activating words generated from all reviews."
        elif algorithm_cl_neurons== "AgglomerativeClustering - all reviews":
            text_set.text = "AgglomerativeClustering - all reviews: Hierarchical clustering of neurons based on how related their most activating words are. List of activating words generated from all reviews."
        neuronData = similarityMatrix_AllReviews
        cluster_labels, colors, _ = clustering.apply_cluster(x,algorithm_cl_neurons,n_clusters,review=rawInput_selections.value,neuronData=neuronData,mode="nn")
    
    elif algorithm_cl_neurons=="Positive-Negative neuron clustering (LSTM's predictions)":
        text_set.text = "Positive-Negative neuron clustering: Clusters neurons based on how much they contributed to classifying the review as positive or negative."
        neuronData = neuron_types
        cluster_labels, colors, spectr = clustering.apply_cluster(x,algorithm_cl_neurons,n_clusters,review=rawInput_selections.value,neuronData=neuronData,mode="nn")
        neutral = tuple(int((spectr[0].lstrip('#'))[i:i+2], 16) for i in (0, 2 ,4))
        positive = tuple(int((spectr[1].lstrip('#'))[i:i+2], 16) for i in (0, 2 ,4))
        negative = tuple(int((spectr[2].lstrip('#'))[i:i+2], 16) for i in (0, 2 ,4))
        neu = "<span style='background-color: rgb("+str(neutral[0])+","+str(neutral[1])+","+str(neutral[2])+")'>Neutral</span>"
        pos = "<span style='background-color: rgb("+str(positive[0])+","+str(positive[1])+","+str(positive[2])+")'>Positive</span>"
        neg = "<span style='background-color: rgb("+str(negative[0])+","+str(negative[1])+","+str(negative[2])+")'>Negative</span>"
        text_set.text = "Positive-Negative neuron clustering: Clusters neurons based on how much they contributed to classifying the review as positive or negative:"+neu+" "+pos+" "+neg
    else:
        if algorithm_cl_neurons=="KMeans - selected gate":
            text_set.text = "KMeans: Clusters neurons based on their gate values after training."
        elif algorithm_cl_neurons=="DBSCAN - selected review":
            text_set.text = "DBSCAN - selected review: Clusters neurons based on how related their most activating words are. List of activating words generated from seleceted review."
        neuronData = similarityMatrix_PerReview       
        cluster_labels, colors, _ = clustering.apply_cluster(x,algorithm_cl_neurons,n_clusters,review=rawInput_selections.value,neuronData=neuronData,mode="nn")

        
    proj_source.data = dict(x=x_pr[:, 0], y=x_pr[:, 1], z=colors)
    

    text_data,text_words = get_rawText_data(rawInput_selections.value,keys_raw,data_raw) ###LOADS EMBEDDINGS HERE
    w2v_labels, w2v_colors, _ = clustering.apply_cluster(text_data,"KMeans - selected gate",n_clusters,mode="wc")
    rawInput_source.data = dict(z=w2v_colors, w=text_words)
    color_dict = get_wc_colourGroups(rawInput_source)
    if gate_value=="input_gate":
        wc_filename,wc_img,wc_words = get_wcloud(LRP,rawInput_selections.value,load_dir,color_dict=color_dict,gate="in",text=text_banner.text)
    elif gate_value=="forget_gate":
        wc_filename,wc_img,wc_words = get_wcloud(LRP,rawInput_selections.value,load_dir,color_dict=color_dict,gate="forget")
    elif gate_value=="output_gate":
        wc_filename,wc_img,wc_words = get_wcloud(LRP,rawInput_selections.value,load_dir,color_dict=color_dict,gate="out")

    words_to_be_highlighted = [i for i in wc_words and totalLRP[rawInput_selections.value]['words']]
    lrp_source.data['lrp'] = scaler.fit_transform(np.array(totalLRP[rawInput_selections.value]['lrp'].tolist()).reshape(-1,1))
    tap_source.data['wc_words'] = words_to_be_highlighted
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


#neuronExcitingWords_AllReviews = list((excitingWords_fullSet.values()))
_,lstm_hidden = data_format.get_data(load_dir+"test_model_internals_lstm_hidden.pickle")
_,learned_embeddings = data_format.get_data(load_dir+"test_model_internals_ebd.pickle")

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

tools = "pan,wheel_zoom,box_zoom,reset"

#Dimensionality reduction
labels = None 
data_pr = data[lstm_layer_name][gate_selections.value]
X, performance_metric = dim_reduction.project(data_pr, "PCA", n_neighbors=5, labels=labels)
X_cluster_labels, X_colors, _ = clustering.apply_cluster(data_pr,algorithm=clustering_selections[0].value,n_clusters=int(clustering_selections[1].value),mode="nn")
proj_source = ColumnDataSource(dict(x=X[:,0],y=X[:,1],z=X_colors))
#  + performance_metric[0] + performance_metric[1]
project_plot = figure(title=projection_selections.value,tools=tools,plot_width=300, plot_height=300)
scatter_tap = project_plot.scatter('x', 'y', marker='circle', size=10, fill_color='z', alpha=0.5, source=proj_source, legend=None)
project_plot.xaxis.axis_label = 'Dim 1'
project_plot.yaxis.axis_label = 'Dim 2'
taptool = TapTool()
project_plot.add_tools(taptool)


#Input text
text_data,text_words = get_rawText_data(rawInput_selections.value,keys_raw,data_raw)  ###LOADS EMBEDDINGS HERE
w2v_labels, w2v_colors, _ = clustering.apply_cluster(text_data,algorithm="KMeans - selected gate",n_clusters=int(clustering_selections[1].value),mode="wc")
rawInput_source = ColumnDataSource(dict(z=w2v_colors,w=text_words))


text_src = re.sub('/home/icha/','/home/yannis/Desktop/tRustNN/',rawInput_selections.value)
text_banner = Div(text=open(text_src,"r").read(), width=1300, height=100)
text_banner2 = Div(text=open(text_src,"r").read(), width=1300, height=100)
label_banner = Paragraph(text="Network decision : POSITIVE" if predicted_tgs[list(keys_raw).index(rawInput_selections.value)][0] == 0 else "Network decision : NEGATIVE", width=200, height=30)

button = Button(label="Reset text")
button.on_click(button_callback)

#WordCloud
color_dict = get_wc_colourGroups(rawInput_source) #Colors based on similarity in embedding space
wc_filename,wc_img,wc_words = get_wcloud(LRP,rawInput_selections.value,load_dir,color_dict=color_dict,gate="in",text=text_banner.text)
words_to_be_highlighted = list(set(wc_words).intersection(totalLRP[rawInput_selections.value]['words']))
highlight_source = ColumnDataSource(dict(scores=[]))
tap_source = ColumnDataSource(dict(wc_words=words_to_be_highlighted))
scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
lrp_source = ColumnDataSource(dict(lrp=scaler.fit_transform(np.array(totalLRP[rawInput_selections.value]['lrp'].tolist()).reshape(-1,1))))
#totalLRP : how relevant is each LSTM neuron


taptool.callback = CustomJS(args=dict(source=tap_source,lrp=lrp_source,high=highlight_source,div=text_banner,div_orig=text_banner2),
code="""
     cell = cb_obj.selected['1d']['indices'][0]
     var d = high.data;
     d['scores'] = []
     for(var i=0; i<source.data['wc_words'].length; i++){
        d['scores'].push(lrp.data['lrp'][cell])
     }
     high.change.emit();
     ws = div_orig.text.split(" ");
     ws_out = [];
     for(var j=0; j<ws.length; j++){
        w_idx = source.data['wc_words'].indexOf(ws[j])
        if (w_idx>=0){
           if (d['scores'][w_idx]>0){
                ws_out.push("<span style='background-color: rgba(255,0,0,"+d['scores'][w_idx]+")'>"+ws[j]+"</span>")
           }
           else if (d['scores'][w_idx]<0){
                ws_out.push("<span style='background-color: rgba(0,255,0,"+Math.abs(d['scores'][w_idx])+")'>"+ws[j]+"</span>")
           }
        }  
        else {
           ws_out.push(ws[j])
        }
     }
     div.text = ws_out.join(" ")
     console.log(ws_out)     
     """)
  

img_source = ColumnDataSource(dict(url = [load_dir+wc_filename]))
xdr = Range1d(start=0, end=600)
ydr = Range1d(start=0, end=600)
wc_plot = Plot(title=None, x_range=xdr, y_range=ydr, plot_width=500, plot_height=550, min_border=0)
image = ImageURL(url=dict(value=load_dir+wc_filename), x=0, y=0, anchor="bottom_left", retry_attempts=5, retry_timeout=1500)
wc_plot.add_glyph(img_source, image)


text_0 = Paragraph(text="Clustering option:", width=200, height=20)
text_set = Div(text="KMeans: Clusters neurons based on their gate values after training.", width=250, height=100)


lrp_timedata = get_lrp_timedata(LRP)
time = [i for i in range(len(lrp_timedata))]
lrptime_source = ColumnDataSource(dict(lrptime = lrp_timedata,time=time))
lrp_plot = figure(title="Total normalized LRP per timestep",plot_width=300, plot_height=50)
lrp_plot.scatter('time','lrptime', marker='circle', size=5, alpha=0.5, source=lrptime_source)
lrp_plot.xaxis.axis_label = 'Time'
lrp_plot.yaxis.axis_label = 'Total normalized LRP'


#Layout
gate_selections.on_change('value', update_source)
projection_selections.on_change('value', update_source)
for attr in clustering_selections:
    attr.on_change('value', update_source)
rawInput_selections.on_change('value', update_source)

gp = layout([project_plot, wc_plot, widgetbox(rawInput_selections,gate_selections,projection_selections,clustering_selections[0],clustering_selections[1],text_0,text_set,label_banner,button)],
            [lrp_plot],
            [text_banner],
            responsive=True)
curdoc().add_root(gp)
curdoc().title = "tRustNN"



