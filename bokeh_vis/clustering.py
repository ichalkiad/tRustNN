import numpy as np
np.random.seed(0)

from bokeh.io import curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.plotting import figure
from bokeh.palettes import Spectral6
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

def clustering(X, algorithm, n_clusters):

    X = np.transpose(X)
    
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # Generate the new colors:
    if algorithm=='MiniBatchKMeans':
        model = cluster.MiniBatchKMeans(n_clusters=n_clusters)

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



def apply_cluster(data,algorithm,n_clusters,algorithm_data=None,review=None,neuronData_jsons=None,test_data_json=None,load_dir=None):

    spectral = np.hstack([Spectral6] * 20)
    #keep only review name
    if review!=None:
        review_part = review.split('/')[-1][:-4]
    y_pred = None
    colors = None
    
    if algorithm == "MiniBatchKMeans - selected gate":        
        x_cl, y_pred = clustering(data, algorithm_data, n_clusters)
        colors = [spectral[i] for i in y_pred]
    else:
        if algorithm == "DBSCAN - selected review":
            reviewData_name = [s for s in neuronData_jsons if review_part in s][0]
            dstMat = get_DstMatrix_singleReview(load_dir+reviewData_name,load_dir+test_data_json,review)
            db = cluster.DBSCAN(eps=0.2,metric='precomputed').fit(dstMat)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            y_pred = db.labels_.astype(np.int)
            colors = [spectral[i] for i in y_pred]
            
        #elif algorithm == "DBSCAN - all reviews":
        #elif algorithm == "AgglomerativeClustering - all reviews":
    


    return y_pred, colors, spectral


def get_cluster_algorithms():

#    return ['MiniBatchKMeans','AffinityPropagation','MeanShift','SpectralClustering','Ward','AgglomerativeClustering','DBSCAN','Birch']
    return (["MiniBatchKMeans - selected gate",
            "DBSCAN - selected review",
            "DBSCAN - all reviews",
            "AgglomerativeClustering - all reviews"
           ], ["MiniBatchKMeans","SpectralClustering","Ward"])

    

