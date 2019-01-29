# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 10:18:03 2017

@author: damodara
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as mpatches
import os




def imshow_grid(images, shape=[2, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)
    n_dim = np.shape(images)
    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        if len(n_dim)<=3:
           grid[i].imshow(images[i], cmap=plt.get_cmap('gray'))  # The AxesGrid object work as a list of axes.
        else:
           grid[i].imshow(images[i]) 
        
        
    plt.show()

def plot_embedding(X, y, d, title=None, save_fig=0, pname=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
#        plot colored number
#        plt.text(X[i, 0], X[i, 1], str(y[i]),
#                 color=plt.cm.bwr(d[i] / 1.),
#                 fontdict={'weight': 'bold', 'size': 9})
        if d[i]==0:
            c = 'red'
        elif d[i]==1:
            c = 'green'
        elif d[i]==2:
            c = 'blue'
            
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color= c,
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    red_patch = mpatches.Patch(color='red', label='Source data')
    green_patch = mpatches.Patch(color='green', label='Target data')
    plt.legend(handles=[red_patch, green_patch])
    plt.show()
    if title is not None:
        plt.title(title)
    if save_fig:
        fname = title+'.png'
        if pname is not None:
            fname = os.path.join(pname, fname) 
        plt.savefig(fname)

def tsne_plot(xs, xt, xs_label, xt_label, map_xs=None, title=None, pname=None):

    num_test=1000
    if map_xs is not None:
        combined_imgs = np.vstack([xs[0:num_test, :], xt[0:num_test, :], map_xs[0:num_test,:]])
        combined_labels = np.vstack([xs_label[0:num_test, :],xt_label[0:num_test, :], xs_label[0:num_test,:]])
        combined_labels = combined_labels.astype('int')
        combined_domain = np.vstack([np.zeros((num_test,1)),np.ones((num_test,1)),np.ones((num_test,1))*2])

    from sklearn.manifold import TSNE

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    source_only_tsne = tsne.fit_transform(combined_imgs)


    plot_embedding(source_only_tsne, combined_labels.argmax(1), combined_domain,
                   title, save_fig=1, pname=pname)
        
 
    
