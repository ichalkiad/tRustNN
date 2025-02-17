import numpy as np
np.random.seed(0)
import random
from bokeh.io import curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.plotting import figure
from bokeh.palettes import Spectral6,d3
import dim_reduction
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import json
import os
import sys

src_path = os.path.abspath("./src/")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from lrp import get_DstMatrix_singleReview

#X: [n_features,n_samples]

def clustering(X, algorithm, n_clusters=2):

    X = np.transpose(X)
    
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=5, include_self=False)

    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # Generate the new colors:
    if algorithm=='KMeans':
        model = cluster.KMeans(n_clusters=n_clusters,random_state=0)

    elif algorithm=='Birch':
        model = cluster.Birch(n_clusters=n_clusters)

    elif algorithm=='DBSCAN':
        model = cluster.DBSCAN(eps=.2)

    elif algorithm=='AffinityPropagation':
        model = cluster.AffinityPropagation(damping=.9,
                                            preference=-200)

    elif algorithm=='MeanShift':
        model = cluster.MeanShift(bandwidth=bandwidth,
                                  bin_seeding=True)

    elif algorithm=='SpectralClustering':
        model = cluster.SpectralClustering(n_clusters=n_clusters,
                                           eigen_solver='arpack',
                                           affinity="nearest_neighbors")

    elif algorithm=='Ward':
        model = cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                                linkage='ward',
                                                connectivity=connectivity)

    elif algorithm=='AgglomerativeClustering':
        model = cluster.AgglomerativeClustering(linkage="average",
                                                affinity="cityblock",
                                                n_clusters=n_clusters,
                                                connectivity=connectivity)

    model.fit(X)

    if hasattr(model, 'labels_'):
            y_pred = model.labels_.astype(np.int)
    else:
            y_pred = model.predict(X)
    
    return X, y_pred



def apply_cluster(data,algorithm,n_clusters,review=None,neuronData=None,mode="nn"):

    spectral = np.hstack([Spectral6] * 20)
    if mode=="nn":
        spectral = ['#d50000','#ff80ab','#6200ea','#40c4ff','#18ffff']
    elif mode=="wc":
        spectral = ['#aed581','#ff5722','#8d6e63','#006064','#4caf50','#ff6f00','#3e2723']
    
    #keep only review name
    y_pred = None
    colors = None
    
    if ((algorithm == "KMeans - selected gate") or (algorithm == "Internal state clustering (LSTM's outputs)")):
        x_cl, y_pred = clustering(data, "KMeans", n_clusters)
    else:
        if algorithm == "DBSCAN - selected review":
            dstMat = neuronData[review]
            db = cluster.DBSCAN(eps=0.2,metric='precomputed').fit(dstMat)
            y_pred = db.labels_.astype(np.int)
        elif algorithm == "DBSCAN - all reviews":
            dstMat = neuronData
            db = cluster.DBSCAN(eps=0.2,metric='precomputed').fit(dstMat)
            y_pred = db.labels_.astype(np.int)
        elif algorithm == "AgglomerativeClustering - all reviews":
            dstMat = neuronData
            db = cluster.AgglomerativeClustering(n_clusters=n_clusters,affinity="precomputed",linkage="average").fit(dstMat)
            y_pred = db.labels_.astype(np.int)
        elif algorithm == "Positive-Negative neuron clustering (LSTM's predictions)":
            y_pred = [int(i) for i in neuronData.tolist()]
        

    colors = [spectral[i] for i in y_pred]
    

    return y_pred, colors, spectral


def get_cluster_algorithms():

#    return ['KMeans','AffinityPropagation','MeanShift','SpectralClustering','Ward','AgglomerativeClustering','DBSCAN','Birch']
    return (["KMeans - selected gate",
            "DBSCAN - selected review",
            "DBSCAN - all reviews",
            "AgglomerativeClustering - all reviews",
            "Positive-Negative neuron clustering (LSTM's predictions)",
             "Internal state clustering (LSTM's outputs)"]
            )

    

