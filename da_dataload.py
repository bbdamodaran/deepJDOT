# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:30:35 2017

@author: damodara
"""
import DatasetLoad
import numpy as np

def mnist_to_usps():
    from keras.datasets import mnist
    (source_traindata, source_trainlabel), (source_testdata, source_testlabel) = mnist.load_data()
    source_size = source_traindata.shape
    
    resize = False
    resize_size =16
    from preprocess import zero_mean_unitvarince,resize_data
    if resize == True:
       source_traindata = resize_data(source_traindata, resize_size=resize_size)
       source_testdata = resize_data(source_testdata, resize_size=resize_size)
    
    source_size = source_traindata.shape
    
    source_traindata = zero_mean_unitvarince(source_traindata,scaling=True)
    source_testdata = zero_mean_unitvarince(source_testdata,scaling=True)
    
    
    source_traindata = source_traindata.reshape(-1,source_size[1],source_size[2],1)
    source_testdata =source_testdata.reshape(-1,source_size[1],source_size[2],1)
    
    #%%
    from DatasetLoad import usps_digit_dataload
    target_traindata, target_trainlabel, target_testdata, target_testlabel = usps_digit_dataload()
    target_trainlabel =target_trainlabel-1
    target_testlabel =target_testlabel-1

    target_traindata = target_traindata.reshape(-1, 16, 16,1)
    target_testdata = target_testdata.reshape(-1, 16, 16,1)
    print(target_traindata.shape)
    resize =True
    resize_size =28
    if resize:
        npad = ((0,0),(6,6),(6,6),(0,0))
        target_traindata = np.pad(target_traindata,pad_width=npad, mode='constant')
        target_testdata = np.pad(target_testdata, pad_width=npad, mode='constant')
        # target_traindata = resize_data(target_traindata, resize_size=resize_size)
        # target_testdata = resize_data(target_testdata, resize_size=resize_size)
        target_traindata = target_traindata.reshape(-1, 28, 28, 1)
        target_testdata = target_testdata.reshape(-1, 28, 28, 1)



    target_traindata = zero_mean_unitvarince(target_traindata, scaling=True)
    target_testdata = zero_mean_unitvarince(target_testdata, scaling=True)

    return (source_traindata, source_trainlabel, source_testdata, source_testlabel), (target_traindata, target_trainlabel, target_testdata, target_testlabel)


#%%

def usps_to_mnist():
    from DatasetLoad import usps_digit_dataload
    source_traindata, source_trainlabel, source_testdata, source_testlabel = usps_digit_dataload()
    source_trainlabel =source_trainlabel-1
    source_testlabel =source_testlabel-1

    # 2d to 3d for CNN
    source_traindata = source_traindata.reshape(-1, 16, 16,1)
    source_testdata = source_testdata.reshape(-1,16, 16,1)

    from preprocess import zero_mean_unitvarince, resize_data

    source_traindata = zero_mean_unitvarince(source_traindata, scaling=True)
    source_testdata = zero_mean_unitvarince(source_testdata, scaling=True)

    
    #
    from keras.datasets import mnist
    (target_traindata, target_trainlabel), (target_testdata, target_testlabel) = mnist.load_data()
    target_size = target_traindata.shape
    
    resize = True
    resize_size =16

    if resize == True:
       target_traindata = resize_data(target_traindata, resize_size=resize_size)
       target_testdata = resize_data(target_testdata, resize_size=resize_size)
    
    target_size = target_traindata.shape
    
    target_traindata = zero_mean_unitvarince(target_traindata,scaling=True)
    target_testdata = zero_mean_unitvarince(target_testdata,scaling=True)
    
    
    target_traindata = target_traindata.reshape(-1,target_size[1],target_size[2],1)
    target_testdata =target_testdata.reshape(-1,target_size[1],target_size[2],1)
    
    return (source_traindata, source_trainlabel, source_testdata, source_testlabel), (target_traindata, target_trainlabel, target_testdata, target_testlabel)
    

#%% MNIST MNISTM
def mnist_to_mnistm():
    from keras.datasets import mnist
    (source_traindata, source_trainlabel), (source_testdata, source_testlabel) = mnist.load_data()
    
    source_size = source_traindata.shape
    resize = False
    resize_size =32
    from preprocess import zero_mean_unitvarince,resize_data
    if resize == True:
       source_traindata = resize_data(source_traindata, resize_size=resize_size)
       source_testdata = resize_data(source_testdata, resize_size=resize_size)
    
    source_size = source_traindata.shape
    
    source_traindata = zero_mean_unitvarince(source_traindata,scaling=True)
    source_testdata = zero_mean_unitvarince(source_testdata,scaling=True)
    
    convert_rgb=1
    if convert_rgb:
        source_traindata = np.stack((source_traindata,source_traindata,source_traindata), axis=3)
        source_testdata = np.stack((source_testdata,source_testdata,source_testdata), axis=3)
        
    from DatasetLoad import mnist_m_dataload
    from skimage.color import rgb2gray
    target_traindata, target_trainlabel, target_testdata, target_testlabel= mnist_m_dataload()
    target_size = target_traindata.shape
    resize = False
    resize_size =28
    
    if resize == True:
       target_traindata = resize_data(target_traindata, resize_size=resize_size)
       target_testdata = resize_data(target_testdata, resize_size=resize_size)
    
    target_size = target_traindata.shape
    
    target_traindata = zero_mean_unitvarince(target_traindata,scaling=True)
    target_testdata = zero_mean_unitvarince(target_testdata,scaling=True)
    
    return (source_traindata, source_trainlabel, source_testdata, source_testlabel), (target_traindata, target_trainlabel, target_testdata, target_testlabel)
    
#%%
def mnistm_to_mnist():
    from DatasetLoad import mnist_m_dataload
    from skimage.color import rgb2gray
    source_traindata, source_trainlabel, source_testdata, source_testlabel= mnist_m_dataload()
    source_size = source_traindata.shape
    resize = True
    resize_size =28
    from preprocess import zero_mean_unitvarince,resize_data
    if resize == True:
       source_traindata = resize_data(source_traindata, resize_size=resize_size)
       source_testdata = resize_data(source_testdata, resize_size=resize_size)
    
    source_size = source_traindata.shape
    
    source_traindata = zero_mean_unitvarince(source_traindata,scaling=True)
    source_testdata = zero_mean_unitvarince(source_testdata,scaling=True)
    
    
    from keras.datasets import mnist
    (target_traindata, target_trainlabel), (target_testdata, target_testlabel) = mnist.load_data()
    
    target_size = target_traindata.shape
    resize = False
    resize_size =32
    from preprocess import zero_mean_unitvarince,resize_data
    if resize == True:
       target_traindata = resize_data(target_traindata, resize_size=resize_size)
       target_testdata = resize_data(target_testdata, resize_size=resize_size)
    
    target_size = target_traindata.shape
    
    target_traindata = zero_mean_unitvarince(target_traindata,scaling=True)
    target_testdata = zero_mean_unitvarince(target_testdata,scaling=True)
    
    convert_rgb=1
    if convert_rgb:
        target_traindata = np.stack((target_traindata,target_traindata,target_traindata), axis=3)
        target_testdata = np.stack((target_testdata,target_testdata,target_testdata), axis=3)
        

    
    return source_traindata, source_trainlabel, source_testdata, source_testlabel, target_traindata, target_trainlabel, target_testdata, target_testlabel


#%% SVHNN MNIST
def svhnn_to_mnist(method = 'zero_mean_unitvarince', **params):
    from skimage.color import rgb2gray
    from scipy.misc import imresize
    from DatasetLoad import SVHN_dataload
    source_traindata, source_trainlabel, source_testdata, source_testlabel = SVHN_dataload()   
    source_size = source_traindata.shape
    
    from preprocess import zero_mean_unitvarince, instance_zero_mean_unitvar, min_max_scaling
    if method =='instance_zero_mean_unitvar':
        source_traindata = instance_zero_mean_unitvar(source_traindata, scaling=True)
        source_testdata = instance_zero_mean_unitvar(source_testdata, scaling=True)
    elif method =='min_max':
        source_traindata = min_max_scaling(source_traindata, **params)
        source_testdata = min_max_scaling(source_testdata, **params)
    else:
        source_traindata = zero_mean_unitvarince(source_traindata, scaling=True)
        source_testdata = zero_mean_unitvarince(source_testdata, scaling=True)


    
    source_trainlabel = source_trainlabel*(source_trainlabel!=10)
    source_testlabel = source_testlabel*(source_testlabel!=10)
 
    
    from keras.datasets import mnist
    (target_traindata, target_trainlabel), (target_testdata, target_testlabel) = mnist.load_data()
    target_size = target_traindata.shape
    
    resize = True
    resize_size =32
    from preprocess import zero_mean_unitvarince,resize_data
    if resize == True:
       target_traindata = resize_data(target_traindata, resize_size=resize_size)
       target_testdata = resize_data(target_testdata, resize_size=resize_size)

    if method =='instance_zero_mean_unitvar':
        target_traindata = instance_zero_mean_unitvar(target_traindata, scaling=True)
        target_testdata = instance_zero_mean_unitvar(target_testdata, scaling=True)
    elif method =='min_max':
        target_traindata = min_max_scaling(target_traindata, **params)
        target_testdata = min_max_scaling(target_testdata, **params)
    else:
        target_traindata = zero_mean_unitvarince(target_traindata,scaling=True)
        target_testdata = zero_mean_unitvarince(target_testdata,scaling=True)
       
    convert_rgb=1
    if convert_rgb:
        target_traindata = np.stack((target_traindata,target_traindata,target_traindata), axis=3)
        target_testdata = np.stack((target_testdata,target_testdata,target_testdata), axis=3)
        
    return (source_traindata, source_trainlabel,source_testdata, source_testlabel), (target_traindata, target_trainlabel, target_testdata, target_testlabel)


def mnist_to_svhnn():
    from keras.datasets import mnist
    (source_traindata, source_trainlabel), (source_testdata, source_testlabel) = mnist.load_data()
    
    source_size = source_traindata.shape
    resize = False
    resize_size =32
    from preprocess import zero_mean_unitvarince,resize_data
    if resize == True:
       source_traindata = resize_data(source_traindata, resize_size=resize_size)
       source_testdata = resize_data(source_testdata, resize_size=resize_size)
    
    source_size = source_traindata.shape
    
    source_traindata = zero_mean_unitvarince(source_traindata,scaling=True)
    source_testdata = zero_mean_unitvarince(source_testdata,scaling=True)
    
    convert_rgb=1
    if convert_rgb:
        source_traindata = np.stack((source_traindata,source_traindata,source_traindata), axis=3)
        source_testdata = np.stack((source_testdata,source_testdata,source_testdata), axis=3)
        
    #########################################
    from skimage.color import rgb2gray
    from scipy.misc import imresize
    from DatasetLoad import SVHN_dataload
    target_traindata, label = SVHN_dataload()   
    target_size = target_traindata.shape
    
    from preprocess import zero_mean_unitvarince
    target_traindata = zero_mean_unitvarince(target_traindata, scaling=True)
    
    target_trainlabel = label*(label!=10)
    target_size = target_traindata.shape
    
    return source_traindata, source_trainlabel, source_testdata, source_testlabel, target_traindata, target_trainlabel
    

def syndigit_to_svhn(method = 'zero_mean_unitvarince'):
    from DatasetLoad import synthetic_digits_dataload
    source_traindata, source_trainlabel, source_testdata, source_testlabel = synthetic_digits_dataload()

    from preprocess import zero_mean_unitvarince, instance_zero_mean_unitvar, min_max_scaling
    if method == 'instance_zero_mean_unitvar':
        source_traindata = instance_zero_mean_unitvar(source_traindata, scaling=True)
        source_testdata = instance_zero_mean_unitvar(source_testdata, scaling=True)
    elif method == 'min_max':
        source_traindata = min_max_scaling(source_traindata)
        source_testdata = min_max_scaling(source_testdata)
    else:
        source_traindata = zero_mean_unitvarince(source_traindata, scaling=True)
        source_testdata = zero_mean_unitvarince(source_testdata, scaling=True)
    
    from DatasetLoad import SVHN_dataload
    target_traindata, target_trainlabel, target_testdata, target_testlabel = SVHN_dataload()   
    target_size = target_traindata.shape

    from preprocess import zero_mean_unitvarince, instance_zero_mean_unitvar, min_max_scaling
    if method == 'instance_zero_mean_unitvar':
        source_traindata = instance_zero_mean_unitvar(source_traindata, scaling=True)
        source_testdata = instance_zero_mean_unitvar(source_testdata, scaling=True)
    elif method == 'min_max':
        source_traindata = min_max_scaling(source_traindata)
        source_testdata = min_max_scaling(source_testdata)
    else:
        source_traindata = zero_mean_unitvarince(source_traindata, scaling=True)
        source_testdata = zero_mean_unitvarince(source_testdata, scaling=True)
    
    target_trainlabel = target_trainlabel*(target_trainlabel!=10)
    target_testlabel = target_testlabel*(target_testlabel!=10)
    
    return (source_traindata, source_trainlabel,source_testdata, source_testlabel), (target_traindata, target_trainlabel, target_testdata, target_testlabel)


# %% stl10 to cifar10
def cifar_to_stl(resize_mode='i',normalize=True):
    import numpy as np
    from keras.datasets import cifar10
    (source_traindata, source_trainlabel), (source_testdata, source_testlabel) = cifar10.load_data()

    # remove the class 'frog' label = '6'
    def remove(data, label, lind):
        ind1 = (label < lind) + (label > lind)
        ind1 = ind1.ravel()
        data = data[ind1]
        label = label[ind1]
        ind2 = label > lind
        label[ind2] = label[ind2] - 1
        return data, label

    source_traindata, source_trainlabel = remove(source_traindata, source_trainlabel, 6)
    source_testdata, source_testlabel = remove(source_testdata, source_testlabel, 6)

    source_size = source_traindata.shape

    if resize_mode=='imagenet':
        resize =True
        resize_size = 224
    else:
        resize =False
        resize_size =32
    from preprocess import zero_mean_unitvarince, resize_data
    if resize == True:
        source_traindata = resize_data(source_traindata, resize_size=resize_size)
        source_testdata = resize_data(source_testdata, resize_size=resize_size)

    source_size = source_traindata.shape
    if normalize == True:
        source_traindata = zero_mean_unitvarince(source_traindata, scaling=True)
        source_testdata = zero_mean_unitvarince(source_testdata, scaling=True)

    from DatasetLoad import stl10_dataload
    target_traindata, target_trainlabel, target_testdata, target_testlabel = stl10_dataload()

    # remove the class name 'monkey', label = '7'
    target_traindata, target_trainlabel = remove(target_traindata, target_trainlabel, 7)
    target_testdata, target_testlabel = remove(target_testdata, target_testlabel, 7)

    if resize_mode=='imagenet':
        resize =True
        resize_size = 224
    else:
        resize =True
        resize_size =32

    from preprocess import zero_mean_unitvarince, resize_data
    if resize == True:
        target_traindata = resize_data(target_traindata, resize_size=resize_size)
        target_testdata = resize_data(target_testdata, resize_size=resize_size)

    from preprocess import zero_mean_unitvarince
    target_traindata = zero_mean_unitvarince(target_traindata, scaling=True)
    target_testdata = zero_mean_unitvarince(target_testdata, scaling=True)

    return (source_traindata, source_trainlabel,source_testdata, source_testlabel), (target_traindata, target_trainlabel, target_testdata, target_testlabel)






    
    
