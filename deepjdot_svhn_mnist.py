# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:21:05 2018

@author: damodara
"""

import numpy as np

import matplotlib.pylab as plt
import dnn
import ot
import os
import json
import copy
import h5py
import importlib
from scipy.spatial.distance import cdist 
import matplotlib as mpl
#mpl.use('Agg')
#plt.switch_backend('agg')
#from sklearn import datasets

from matplotlib.colors import ListedColormap

#%% SVHN - MNIST
from da_dataload import svhnn_to_mnist
(source_traindata, source_trainlabel, source_testdata, source_testlabel),\
(target_traindata, target_trainlabel,target_testdata, target_testlabel)=svhnn_to_mnist('min_max', lowerbound_zero=True)
data_name = 'svhnn_mnist'

#%%
do_reg = True

if do_reg:
    source_trainlabel_cat = source_trainlabel / 10
    source_testlabel_cat = source_testlabel / 10
    target_trainlabel_cat = (target_trainlabel / 10)[..., None]
    target_testlabel_cat = (target_testlabel / 10)[..., None]
else:
    from keras.utils.np_utils import to_categorical
    source_trainlabel_cat = to_categorical(source_trainlabel)
    source_testlabel_cat = to_categorical(source_testlabel)
    #test_label_cat = to_categorical(y_test)
    #
    target_trainlabel_cat = to_categorical(target_trainlabel)
    target_testlabel_cat = to_categorical(target_testlabel)
    #target_label_cat = to_categorical(target_label)
#%%
n_class = source_trainlabel_cat.shape[-1]
n_dim = np.shape(source_traindata)

#%%
pathname ='results/'
filesave = True
 #%%
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val 
#%%
def feature_extraction(model, data, out_layer_num=-2, out_layer_name=None):
    '''
    extract the features from the pre-trained model
    inp_layer_num - input layer
    out_layer_num -- from which layer to extract the features
    out_layer_name -- name of the layer to extract the features
    '''
    if out_layer_name is None:
        intermediate_layer_model = dnn.Model(inputs=model.layers[0].input,
                             outputs=model.layers[out_layer_num].output)
        intermediate_output = intermediate_layer_model.predict(data)
    else:
        intermediate_layer_model = dnn.Model(inputs=model.layers[0].input,
                             outputs=model.get_layer(out_layer_name).output)
        intermediate_output = intermediate_layer_model.predict(data)
        
    
    return intermediate_output
    
#%%
def tsne_plot(xs, xt, xs_label, xt_label, subset=True, title=None, pname=None):
    num_test=1000
    import matplotlib.cm as cm
    if subset:
        combined_imgs = np.vstack([xs[0:num_test, :], xt[0:num_test, :]])
        combined_labels = np.vstack([xs_label[0:num_test, :],xt_label[0:num_test, :]])
        combined_labels = combined_labels.astype('int')
        combined_domain = np.vstack([np.zeros((num_test,1)),np.ones((num_test,1))])
    
    from sklearn.manifold import TSNE
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    source_only_tsne = tsne.fit_transform(combined_imgs)
    plt.figure(figsize=(15,15))
    plt.scatter(source_only_tsne[:num_test,0], source_only_tsne[:num_test,1], c=combined_labels[:num_test].argmax(1),
                s=50, alpha=0.5,marker='o', cmap=cm.jet, label='source')
    plt.scatter(source_only_tsne[num_test:,0], source_only_tsne[num_test:,1], c=combined_labels[num_test:].argmax(1),
                s=50, alpha=0.5,marker='+',cmap=cm.jet,label='target')
    plt.axis('off')
    plt.legend(loc='best')
    plt.title(title)
    if filesave:
        plt.savefig(os.path.join(pname,title+'.png'),bbox_inches='tight', pad_inches = 0,
                    format='png')
    else:
        plt.savefig(title+'.png')
    plt.close() 


#%% source model
from architectures import assda_feat_ext, classifier, regressor, res_net50_fe 
ms = dnn.Input(shape=(n_dim[1],n_dim[2],n_dim[3]))
fes = assda_feat_ext(ms, small_model=True)
output_layer = regressor if do_reg else classifier
nets = output_layer(fes, n_class)
source_model = dnn.Model(ms, nets)
#%%
optim = dnn.keras.optimizers.Adam(lr=0.0002)#,beta_1=0.999, beta_2=0.999)
if do_reg:
    metrics = ['mae']
    loss = 'binary_crossentropy'
else:
    metrics = ['acc', 'mae']
    loss = 'categorical_crossentropy'
source_model.compile(optimizer=optim, loss=loss, metrics=metrics)
checkpoint = dnn.keras.callbacks.ModelCheckpoint('bst.hdf5', monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto')
early_stop = dnn.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, 
                                                       patience=5, verbose=0, mode='auto')
callbacks_list = [early_stop, checkpoint]
#%%
source_model.fit(source_traindata, source_trainlabel_cat, batch_size=128, epochs=10,
                  validation_split=0.16, callbacks=callbacks_list)

# pp='/home/damodara/OT/DA/ALJDOT/codes/results/adaa_source/mnist_usps'
# source_model = dnn.keras.models.load_model(os.path.join(pp, 'mnist_usps_sourcemodel.h5'))

# source_model.load_weights('bst.hdf5')
smodel_train_acc = source_model.evaluate(source_traindata, source_trainlabel_cat)
smodel_test_acc = source_model.evaluate(source_testdata, source_testlabel_cat)
smodel_target_trainacc = source_model.evaluate(target_traindata, target_trainlabel_cat)
smodel_target_testacc = source_model.evaluate(target_testdata, target_testlabel_cat)
print(source_model.metrics_names)
print("source train eval using source model", smodel_train_acc)
print("target train eval using source model", smodel_target_trainacc)
print("source test eval using source model", smodel_test_acc)
print("target test eval using source model", smodel_target_testacc)


#%%
if filesave:
    source_model.save(os.path.join(pathname,data_name+'_Sourcemodel.h5'))
# source_model = dnn.keras.models.load_model(os.path.join(pathname, 'mnist_usps_Sourcemodel.h5'))

#%%
if not do_reg:
    sd = feature_extraction(source_model, source_testdata[:5000,:], out_layer_num=-2)
    td = feature_extraction(source_model, target_testdata[:5000,:], out_layer_num=-2)
    # td = feature_extraction(source_model, target_testdata, out_layer_num=-2)
    title = data_name+'_smodel'
    tsne_plot(sd, td, source_testlabel_cat, target_testlabel_cat, title=title, pname=pathname)
#%% Creating components of target model

main_input = dnn.Input(shape=(n_dim[1],n_dim[2],n_dim[3]))
fe = assda_feat_ext(main_input, small_model=True)
fe_size=fe.get_shape().as_list()[1]
fe_model = dnn.Model(main_input, fe, name= 'fe_model')
#
cl_input = dnn.Input(shape=(fe.get_shape().as_list()[1],))
net = output_layer(cl_input , n_class,l2_weight=0.0)
cl_model = dnn.Model(cl_input, net, name ='classifier')
#fe_size = 768
#%% aljdot model
main_input = dnn.Input(shape=(n_dim[1],n_dim[2],n_dim[3]))
ffe=fe_model(main_input)
net = cl_model(ffe)
#con_cat = dnn.concatenate([net, ffe ], axis=1)
model = dnn.Model(inputs=main_input, outputs=[net, ffe])
#model.set_weights(source_model.get_weights())

#%% Target model loss and fit function
optim = dnn.keras.optimizers.Adam(lr=0.0001)#,beta_1=0.999, beta_2=0.999)
#sample_size=50
from sklearn.metrics import accuracy_score, mean_absolute_error

class jdot_align(object):
    def __init__(self, model, batch_size, n_class, optim, allign_loss=1.0, tar_cl_loss=1.0, 
                 sloss=0.0,tloss=1.0,int_lr=0.01, ot_method='emd',
                 jdot_alpha=0.01, lr_decay=True, verbose=1):
        self.model = model
        self.batch_size = batch_size
        self.sbatch_size = batch_size
        self.n_class= n_class
        self.optimizer= optim
        self.gamma=dnn.K.zeros(shape=(self.batch_size, self.batch_size))
        self.tgamma = dnn.K.zeros(shape=(self.batch_size, self.batch_size))
        self.train_cl =dnn.K.variable(tar_cl_loss)
        self.train_algn=dnn.K.variable(allign_loss)
        self.sloss = dnn.K.variable(sloss)
        self.tloss = dnn.K.variable(tloss)
        self.verbose = verbose
        self.int_lr =int_lr
        self.lr_decay= lr_decay
        #
        self.ot_method = ot_method
        self.jdot_alpha=jdot_alpha
        # target classification L2 loss       
        def classifier_cat_loss(y_true, y_pred):
            '''
            sourceloss + target classification loss
            classifier loss based on categorical cross entropy in the target domain
            1:batch_size - is source samples
            batch_size:end - is target samples
            self.gamma - is the optimal transport plan
            '''
            # source true labels
            ys = y_true[:self.batch_size,:]
            # target prediction
            ypred_t = y_pred[self.batch_size:2*self.batch_size,:]
            source_ypred = y_pred[:self.batch_size,:]
            if do_reg:
                source_loss = dnn.K.mean(dnn.K.binary_crossentropy(ys, source_ypred))
            else:
                source_loss = dnn.K.mean(dnn.K.categorical_crossentropy(ys, source_ypred))
            # categorical cross entropy loss
            ypred_t = dnn.K.log(ypred_t)
            # loss calculation based on double sum (sum_ij (ys^i, ypred_t^j))
            loss = -dnn.K.dot(ys, dnn.K.transpose(ypred_t))
            return self.train_cl*(self.tloss*dnn.K.sum(self.gamma * loss) + self.sloss*source_loss)

        self.classifier_cat_loss = classifier_cat_loss
        
#        def source_classifier_cat_loss(y_true, y_pred):
#            '''
#            classifier loss based on categorical cross entropy in the source domain
#            1:batch_size - is source samples
#            batch_size:end - is target samples
#            '''
#            # source true labels
#            ys = y_true[:self.batch_size,:]
#            source_ypred = y_pred[:self.batch_size,:]
#            source_loss = dnn.K.mean(dnn.K.categorical_crossentropy(ys, source_ypred))
#             
#            return self.sloss*source_loss
#        self.source_classifier_cat_loss = source_classifier_cat_loss
        
        def L2_dist(x,y):
            '''
            compute the squared L2 distance between two matrics
            '''
            dist = dnn.K.reshape(dnn.K.sum(dnn.K.square(x),1), (-1,1))
            dist += dnn.K.reshape(dnn.K.sum(dnn.K.square(y),1), (1,-1))
            dist -= 2.0*dnn.K.dot(x, dnn.K.transpose(y))  
            return dist
 
        def align_loss(y_true, y_pred):
            '''
            source and target alignment loss in the intermediate layers of the target model
            allignment is performed in the target model (both source and target features are from targte model)
            y-true - is dummy value( that is full of zeros)
            y-pred - is the value of intermediate layers in the target model
            1:batch_size - is source samples
            batch_size:end - is target samples            
            '''
            # source domain features            
            gs = y_pred[:self.batch_size,:]
            # target domain features
            gt = y_pred[self.batch_size:2*self.batch_size,:]
            gdist = L2_dist(gs,gt)  
            loss = self.jdot_alpha*dnn.K.sum(self.gamma*gdist)
            return self.train_algn*loss
        self.align_loss= align_loss
 

 
    def fit(self, source_traindata, ys_label, target_traindata, n_iter=5000, cal_bal=True):
        '''
        ys_label - source data true labels
        '''
        if do_reg:
            print("Regression mode, cal_bal will be set to False")
            cal_bal = False
        ns = source_traindata.shape[0]
        nt= target_traindata.shape[0]
        method=self.ot_method # for optimal transport
        alpha=self.jdot_alpha
        t_acc = []
        t_loss =[]
        tloss = dnn.K.eval(self.tloss)
        g_metric ='deep'
        def mini_batch_class_balanced(label, sample_size=20, shuffle=False):
            ''' sample the mini-batch with class balanced
            '''
            label = np.argmax(label, axis=1)
            if shuffle:
                rindex = np.random.permutation(len(label))
                label = label[rindex]

            n_class = len(np.unique(label))
            index = []
            for i in range(n_class):
                s_index = np.nonzero(label == i)
                s_ind = np.random.permutation(s_index[0])
                index = np.append(index, s_ind[0:sample_size])
                #          print(index)
            index = np.array(index, dtype=int)
            return index

        self.model.compile(optimizer= optim, loss =[self.classifier_cat_loss, self.align_loss])
        dnn.K.set_value(self.model.optimizer.lr, self.int_lr)        
        for i in range(n_iter):
            if self.lr_decay and i%10000 ==0:
                # p = float(i) / n_iter
                # lr = self.int_lr / (1. + 10 * p)**0.9
                lr = dnn.K.get_value(self.model.optimizer.lr)
                dnn.K.set_value(self.model.optimizer.lr, lr*0.1)
            # fixing f and g, and computing optimal transport plan (gamma)
            if cal_bal:
                s_ind = mini_batch_class_balanced(ys_label, sample_size=sample_size)
                self.sbatch_size = len(s_ind)
            else:
                s_ind = np.random.choice(ns, self.batch_size)
                self.sbatch_size = self.batch_size

            t_ind = np.random.choice(nt, self.batch_size)

            
            xs_batch, ys = source_traindata[s_ind], ys_label[s_ind]
            xt_batch = target_traindata[t_ind]


            l_dummy = np.zeros_like(ys)
            g_dummy = np.zeros((2*self.batch_size, fe_size))
            s = xs_batch.shape
            
            # concat of source and target samples and prediction
            modelpred = self.model.predict(np.vstack((xs_batch, xt_batch)))
            # intermediate features
            gs_batch = modelpred[1][:self.batch_size, :]
            gt_batch = modelpred[1][self.batch_size:, :]
            # softmax prediction of target samples
            ft_pred = modelpred[0][self.batch_size:,:]

            if g_metric=='orginal':
                # compution distance metric
                if len(s) == 3:  # when the input is image, convert into 2D matrix
                    C0 = cdist(xs_batch.reshape(-1, s[1] * s[2]), xt_batch.reshape(-1,
                                                                                   s[1] * s[2]), metric='sqeuclidean')

                elif len(s) == 4:
                    C0 = cdist(xs_batch.reshape(-1, s[1] * s[2] * s[3]), xt_batch.reshape(-1,
                                                                                          s[1] * s[2] * s[3]),metric='sqeuclidean')

            else:
                # distance computation between source and target
                C0 = cdist(gs_batch, gt_batch, metric='sqeuclidean')
            
           #  if i==0:
           #      scale = np.max(C0)
           #  C0/=scale
            C1 = cdist(ys, ft_pred, metric='sqeuclidean')
            
            C= alpha*C0+tloss*C1
                             
            # transportation metric
            
            if method == 'emd':
                 gamma=ot.emd(ot.unif(gs_batch.shape[0]),ot.unif(gt_batch.shape[0]),C)
            elif method =='sinkhorn':
                 gamma=ot.sinkhorn(ot.unif(gs_batch.shape[0]),ot.unif(gt_batch.shape[0]),C,reg)
            # update the computed gamma                      
            dnn.K.set_value(self.gamma, gamma)

            
            data = np.vstack((xs_batch, xt_batch))
            hist = self.model.train_on_batch([data], [np.vstack((ys, l_dummy)), g_dummy])
            t_loss.append(hist[0])
            if self.verbose:
                if i%50==0:
                   print ('iter =', i)
                   print(list(zip(self.model.metrics_names, hist)))
                   evaluation = self.evaluate(target_testdata, target_testlabel_cat)
#                   tpred = self.model.predict(target_testdata)[0]
                   if do_reg:
                       mae = evaluation
                       print('Target mae:', mae)
                       t_acc.append(mae)
                   else:
                       acc, mae = evaluation
                       t_acc.append(acc)
                       print('Target acc:', t_acc[-1])
                       print('Target mae:', mae)
        return hist, t_loss, t_acc
            
        

    def predict(self, data):
        ypred = self.model.predict(data)[1]
        return ypred

    def evaluate(self, data, label):
        ypred = self.model.predict(data)[0]
        if not do_reg:
            acc = accuracy_score(label.argmax(1), ypred.argmax(1))
        mae = mean_absolute_error(label, ypred)
        return mae if do_reg else (acc, mae)
    
    
#%%
model.set_weights(source_model.get_weights())
#model.set_weights(allweights)
batch_size=500
sample_size=50
sloss = 1.0; tloss=0.0001; int_lr=0.001; jdot_alpha=0.001
al_model = jdot_align(model, batch_size, n_class, optim,allign_loss=1.0,
                      sloss=sloss,tloss=tloss,int_lr=int_lr,jdot_alpha=jdot_alpha,lr_decay=True)
h,t_loss,tacc = al_model.fit(source_traindata, source_trainlabel_cat, target_traindata,
                            n_iter=1000,cal_bal=True)
#%%
print(metrics)
tmodel_source_train_acc = al_model.evaluate(source_traindata, source_trainlabel_cat)
print("source train eval using source+target model", tmodel_source_train_acc)
tmodel_tar_train_acc = al_model.evaluate(target_traindata, target_trainlabel_cat)
print("target train eval using source+target model", tmodel_tar_train_acc)
tmodel_source_test_acc = al_model.evaluate(source_testdata, source_testlabel_cat)
print("source test eval using source+target model", tmodel_source_test_acc)
tmodel_tar_test_acc = al_model.evaluate(target_testdata, target_testlabel_cat)
print("target test eval using source+target model", tmodel_tar_test_acc)

#print("target domain acc", tmodel_tar_test_acc)
#print("trained on target, source acc", tmodel_source_test_acc)
#print("maximum target domain acc", np.max(tacc))

allweights = model.get_weights()
#%% deepjdot model save

if filesave:
    al_model.model.save(os.path.join(pathname, data_name+'tmodel_tloss_'+str(tloss)+'_jalpa_'+str(jdot_alpha)+'.h5'))
    al_model.model.save_weights(os.path.join(pathname, data_name+'t_weights_tloss_'+str(tloss)+'_jalpa_'+str(jdot_alpha)+'.h5'))
    sss=al_model.model.to_json()
    # np.savez(os.path.join(pathname, data_name+'_DeepJDOT_parameter.npz'), allign_loss = 1.0, sloss=1.0, t_loss=1.0, int_lr=0.0001,
    #          jdot_alpha=0.001, lr_decay=True)
    #
    # #%% save results in txt file
    fn = os.path.join(pathname, data_name+'_deepjdot_eval.txt')
    fb = open(fn,'w')
    fb.write(" data name = %s DeepJDOT\n" %(data_name))
    fb.write("DeepJDOT param, sloss =%f, tloss=%f,jdot_alpha=%f, int_lr=%f\n" %(sloss, tloss, jdot_alpha, int_lr))
    fb.write("=============================\n")
    fb.write("Trained in source domain, source data train eval =%s\n" %(smodel_train_acc))
    fb.write("Trained in source domain, source data test eval=%s\n" %(smodel_test_acc))
    fb.write("Trained in source domain, target data train eval=%s\n" %(smodel_target_trainacc))
    fb.write("Trained in source domain, target data test eval=%s\n" %(smodel_target_testacc))
    fb.write("=======DeepJDOT Results====================\n")
    fb.write("Trained with DeepJDOT model, source data train eval=%s\n" %(tmodel_source_train_acc))
    fb.write("Trained with DeepJDOT model, source data test eval=%s\n" %(tmodel_source_test_acc))
    fb.write("Trained with DeepJDOT model, target data train eval=%s\n" %(tmodel_tar_train_acc))
    fb.write("Trained with DeepJDOT model, target data test eval=%s\n" %(tmodel_tar_test_acc))
    # fb.write("Target domain DeepJDOT model, target data max acc = %f\n" %(np.max(tacc)))
    fb.close()

#    np.savez(os.path.join(pathname, data_name+'deepjdot_objvalues.npz'), hist_loss = h, total_loss=t_loss, target_acc=tacc)
#%%
if not do_reg:
    al_sourcedata = model.predict(source_traindata[:2000,:])[1]
    al_targetdata = model.predict(target_traindata[:2000,:])[1]
    
    title = data_name+'_DeepJDOT'
    tsne_plot(al_sourcedata, al_targetdata, source_trainlabel_cat, target_trainlabel_cat,
              title=title, pname=os.path.join(pathname))
