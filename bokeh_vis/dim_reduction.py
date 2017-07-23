import numpy as np
np.random.seed(0)

from bokeh.io import curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.plotting import figure
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

"""
    elif algorithm=="LDA": # Needs labels
        X.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
        X = discriminant_analysis.LinearDiscriminantAnalysis(n_components).fit_transform(X, labels)
"""

#X: [n_features,n_samples]


def project(X, algorithm, n_neighbors, labels=None):

    n_components = 2
    X = np.transpose(X)
    
    if algorithm=="LSA":
        svd = decomposition.TruncatedSVD(n_components)
        svd.fit_transform(X)
        performance_metric = (" - Total explained variance in data: ",str(np.sum(svd.explained_variance_)))
    elif algorithm=="PCA":
        pca = decomposition.PCA(n_components)
        pca.fit_transform(X)
        performance_metric = (" - Total explained variance in data: ",str(np.sum(pca.explained_variance_)))
    elif algorithm=="ISOMAP":
        isomap = manifold.Isomap(n_neighbors, n_components)
        isomap.fit_transform(X)
        performance_metric = (None,None)
    elif algorithm=="LLE":
        lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,method='standard')
        X = lle.fit_transform(X)
        performance_metric = (" - Reconstruction error: ",str(lle.reconstruction_error_))
    elif algorithm=="MDS":
        mds = manifold.MDS(n_components, n_init=1, max_iter=100)
        X = mds.fit_transform(X)
        performance_metric = (None,None)
    elif algorithm=="tSNE":
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X = tsne.fit_transform(X)
        performance_metric = (" - Kullback-Leibler divergence: ",str(tsne.kl_divergence_))
    
    return X,performance_metric




def get_dimReduction_algorithms():

    """
    'LDA',
    """
    return ['LSA',
            'PCA',
            'ISOMAP',
            'LLE',
            'MDS',
            'tSNE'
           ]


    

