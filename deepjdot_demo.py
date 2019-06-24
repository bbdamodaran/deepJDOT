# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:26:41 2018

@author: damodara
DeepJDOT: with emd for the sample data
"""

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import dnn
from scipy.spatial.distance import cdist 
import ot
from sklearn.datasets import make_moons, make_blobs

#seed=1985
#np.random.seed(seed)

#%%
source_traindata, source_trainlabel = make_blobs(1200, centers=[[0, 0], [0, 1]], cluster_std=0.2)
target_traindata, target_trainlabel = make_blobs(1200, centers=[[1, 1], [1, 2]], cluster_std=0.2)
plt.figure()
plt.scatter(source_traindata[:,0], source_traindata[:,1], c=source_trainlabel, alpha=0.4)
plt.scatter(target_traindata[:,0], target_traindata[:,1], c=target_trainlabel, marker='x', alpha=0.4)
plt.legend(['source train data', 'target train data'])
plt.title("2D blobs (purple=class_0, yellow=class_1)")

# convert to one hot encoded vector
from keras.utils.np_utils import to_categorical
source_trainlabel_cat = to_categorical(source_trainlabel)
target_trainlabel_cat = to_categorical(target_trainlabel)
#%% optimizer
n_class = len(np.unique(source_trainlabel))
n_dim = np.shape(source_traindata)
optim = dnn.keras.optimizers.SGD(lr=0.001)

#%% feature extraction and classifier function definition

def feat_ext(main_input, l2_weight=0.0):
    net = dnn.Dense(500, activation='relu', name='fe')(main_input)
    net = dnn.Dense(100, activation='relu', name='feat_ext')(net)
    return net
    
def classifier(model_input, nclass, l2_weight=0.0):
    net = dnn.Dense(100, activation='relu', name='cl')(model_input)
    net = dnn.Dense(nclass, activation='softmax', name='cl_output')(net)
    return net
     
#%% Feature extraction as a keras model
main_input = dnn.Input(shape=(n_dim[1],))
fe = feat_ext(main_input)
fe_size=fe.get_shape().as_list()[1]
# feature extraction model
fe_model = dnn.Model(main_input, fe, name= 'fe_model')
# Classifier model as a keras model
cl_input = dnn.Input(shape =(fe.get_shape().as_list()[1],))  # input dim for the classifier 
net = classifier(cl_input , n_class)
# classifier keras model
cl_model = dnn.Model(cl_input, net, name ='classifier')
#%% source model
ms = dnn.Input(shape=(n_dim[1],))
fes = feat_ext(ms)
nets = classifier(fes,n_class)
source_model = dnn.Model(ms, nets)
source_model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
source_model.fit(source_traindata, source_trainlabel_cat, batch_size=128, epochs=75, validation_data=(target_traindata, target_trainlabel_cat))
source_acc = source_model.evaluate(source_traindata, source_trainlabel_cat)
target_acc = source_model.evaluate(target_traindata, target_trainlabel_cat)
print("source acc using source model", source_acc)
print("target acc using source model", target_acc)

#%% Target model
main_input = dnn.Input(shape=(n_dim[1],))
# feature extraction model
ffe=fe_model(main_input)
# classifier model
net = cl_model(ffe)
#con_cat = dnn.concatenate([net, ffe ], axis=1)
# target model with two outputs: predicted class prob, and intermediate layers
model = dnn.Model(inputs=main_input, outputs=[net, ffe])
model.set_weights(source_model.get_weights())
#%% deepjdot model and training
from Deepjdot import Deepjdot

batch_size=128
sample_size=50
sloss = 2.0; tloss=1.0; int_lr=0.002; jdot_alpha=5.0
# DeepJDOT model initalization
al_model = Deepjdot(model, batch_size, n_class, optim,allign_loss=1.0,
                      sloss=sloss,tloss=tloss,int_lr=int_lr,jdot_alpha=jdot_alpha,
                      lr_decay=True,verbose=1)
# DeepJDOT model fit
h,t_loss,tacc = al_model.fit(source_traindata, source_trainlabel_cat, target_traindata,
                            n_iter=1500,cal_bal=False)


#%% accuracy assesment
tarmodel_sacc = al_model.evaluate(source_traindata, source_trainlabel_cat)    
acc = al_model.evaluate(target_traindata, target_trainlabel_cat)
print("source acc using source+target model", tarmodel_sacc)
print("target acc using source+target model", acc)
#%% Intermediate layers features extraction from the pre-trained model
def feature_extraction(model, data, out_layer_num=-2, out_layer_name=None):
    '''
    extract the features from the pre-trained model
    model - keras model from the features to be extracted
    data  - input to the keras model   
    inp_layer_num - input layer
    out_layer_num -- from which layer to extract the features
    out_layer_name -- name of the layer to extract the features
    '''
    if out_layer_name is None:
        # define the model
        intermediate_layer_model = dnn.Model(inputs=model.layers[0].input,
                             outputs=model.layers[out_layer_num].output)
        # extract the features of the intermediate layer
        intermediate_output = intermediate_layer_model.predict(data)
    else:
        intermediate_layer_model = dnn.Model(inputs=model.layers[0].input,
                             outputs=model.get_layer(out_layer_name).output)
        intermediate_output = intermediate_layer_model.predict(data)
        
    
    return intermediate_output
#%%  source model intermediate layer values  
smodel_source_feat = feature_extraction(source_model, source_traindata[:100,],
                                        out_layer_name='feat_ext')
smodel_target_feat  = feature_extraction(source_model, target_traindata[:100,],
                                        out_layer_name='feat_ext')

#%% intermediate layers of source and target domain for TSNE plot of target (DeepJDOT) model
subset = 100
al_sourcedata = model.predict(source_traindata[:subset,])[1]
al_targetdata = model.predict(target_traindata[:subset,])[1]

#%% function for TSNE plot (source and target are combined)
def tsne_plot(xs, xt, xs_label, xt_label, subset=True, title=None, pname=None):

    num_test=100
    if subset:
        combined_imgs = np.vstack([xs[0:num_test, :], xt[0:num_test, :]])
        combined_labels = np.vstack([xs_label[0:num_test, :],xt_label[0:num_test, :]])
        combined_labels = combined_labels.astype('int')
            
    from sklearn.manifold import TSNE
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    source_only_tsne = tsne.fit_transform(combined_imgs)
    plt.figure(figsize=(10, 10))
    plt.scatter(source_only_tsne[:num_test,0], source_only_tsne[:num_test,1],
                c=combined_labels[:num_test].argmax(1), s=75, marker='o', alpha=0.5, label='source train data')
    plt.scatter(source_only_tsne[num_test:,0], source_only_tsne[num_test:,1], 
                c=combined_labels[num_test:].argmax(1),s=50,marker='x',alpha=0.5,label='target train data')
    plt.legend(loc='best')
    plt.title(title)

#%% TSNE plots of source model and target model
title = 'tsne plot of source and target data with source model\n(purple=class_0, yellow=class_1)'
tsne_plot(smodel_source_feat, smodel_target_feat, source_trainlabel_cat, target_trainlabel_cat, title=title)

title = 'tsne plot of source and target data with source+target model\n(purple=class_0, yellow=class_1)'
tsne_plot(al_sourcedata, al_targetdata, source_trainlabel_cat, target_trainlabel_cat, title=title)
    

