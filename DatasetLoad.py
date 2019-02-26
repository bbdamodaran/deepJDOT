# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:17:23 2016

@author: damodara
"""
#%% 
"""
function loads the data
MNIST, forestcover, digits, iris datasets are loaded from sklearn.datasets
"""
#%%
def adult_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:/PostDocWork/LSML/RandomFourierFeatures/Datasets/adult/adult/adult123.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['XTrain']
    TrainData=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    TrainData=TrainData.T
    train_label=np.squeeze(adult['yTrain'])
    Dummy1=adult['XTest']
    TestData = csc_matrix(Dummy1,shape=Dummy1.shape).toarray()
    TestData=TestData.T
    test_label = np.squeeze(adult['yTest'])
    del Dummy, Dummy1
    return TrainData, train_label, TestData, test_label
    

    
def cifar10_dataload():
    import numpy as np
    import scipy.io as sio
    import os
    filename='D:/PostDocWork/LSML/RandomFourierFeatures/Datasets/cifar-10-matlab/cifar-10-batches-mat/CIFAR-10-TrainCombined-Test.mat'
    cifar=sio.loadmat(filename)
    TrainData=cifar['TrainData']
    train_label=cifar['Trainlabel']
    TestData=cifar['TestData']
    test_label=cifar['Testlabel']
    
    return TrainData, train_label, TestData, test_label


def cifar10_deepfeat_dataload():
    import numpy as np
    import scipy.io as sio
    import os
    filename='D:\PostDocWork\LSML\RandomFourierFeatures\Datasets\cifar10-alexnet\cifar10-alexnet-fc7.mat'
    cifar=sio.loadmat(filename)
    TrainData=cifar['TrainData']
    train_label=cifar['Trainlabel']
    TestData=cifar['TestData']
    test_label=cifar['Testlabel']
    
    return TrainData, train_label, TestData, test_label
    
def MNIST_dataload():
    from sklearn.datasets import fetch_mldata
    import numpy as np
    mnist = fetch_mldata('MNIST original')
    Data = mnist.data
    label = mnist.target
    return Data,label
    
def MNIST_official_split_dataload():
    import os
    import numpy as np
    pname ='D:\PostDocWork\LSML\RandomFourierFeatures\Datasets\mnist_official'
    fname ='MNIST_OfficialSplit.npz'
    mnist = np.load(os.path.join(pname,fname))
    TrainData = mnist['TrainData']
    train_label = mnist['Trainlabel']
    TestData = mnist['TestData']
    test_label = mnist['Testlabel']
    return TrainData, train_label, TestData, test_label
    
def forest_dataload():
    from sklearn.datasets import fetch_covtype
    import numpy as np
    forest = fetch_covtype()
    Data= forest['data']
    label = forest['target']
    return Data, label
    
def digits_dataload():
    from sklearn import datasets
    Digits=datasets.load_digits()
    Data=Digits.data/16.
    label=Digits.target
    return Data,label
    
def iris_dataload():
    from sklearn import datasets
    iris=datasets.load_iris()
    Data=iris.data
    label=iris.target
    return Data,label
    
def covtype_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:\PostDocWork\LSML\RandomFourierFeatures\Datasets\covtype\covtype\covtype.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['Data']
    Data=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    label=np.squeeze(adult['label'])    
    return Data, label

def ijcnn1_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:\PostDocWork\LSML\RandomFourierFeatures\Datasets\ijcnn1\ijcnn1_combined.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['Xtrain']
    TrainData=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    train_label=np.squeeze(adult['ytrain'])
    Dummy1=adult['Xtest']
    TestData = csc_matrix(Dummy1,shape=Dummy1.shape).toarray()
    test_label = np.squeeze(adult['ytest'])
    Dummy2=adult['Xval']
    ValData = csc_matrix(Dummy2,shape=Dummy2.shape).toarray()
    val_label = np.squeeze(adult['yval'])
    return TrainData, train_label, TestData, test_label, ValData, val_label
    
    
def usps_digit_dataload():
    import numpy as np
    import scipy.io as sio
    import os
    filename='/home/damodara/OT/DA/datasets/usps_digits/usps.mat'
    usps=sio.loadmat(filename)
    TrainData = usps['TrainData']
    TrainData = ((TrainData + 1) / 2.0) * 255.0
    train_label = usps['trainlabel']
    TestData = usps['TestData']
    TestData = ((TestData + 1) / 2.0) * 255.0
    test_label=usps['testlabel']
    
    return TrainData, train_label, TestData, test_label
    
#def rcv1_dataload():
#%% domain adaptaion datasets
# cal tech
def caltec_decaf_dataload():
    import numpy as np
    import scipy.io as sio
    import os
    pathname = 'D:\Datasets\DomianAdaptation\Office_CalTech\decaf6'
    fn = 'caltech_decaf.mat'
    caltech = sio.loadmat(os.path.join(pathname, fn))
    Data = caltech['feas']
    label = caltech['labels']
    return Data, label
    
# amazon 
def amazon_decaf_dataload():
    import numpy as np
    import scipy.io as sio
    import os
    pathname = 'D:\Datasets\DomianAdaptation\Office_CalTech\decaf6'
    fn = 'amazon_decaf.mat'
    loaddata = sio.loadmat(os.path.join(pathname, fn))
    Data = loaddata['feas']
    label = loaddata['labels']
    return Data, label  
    
def dslr_decaf_dataload():
    import numpy as np
    import scipy.io as sio
    import os
    pathname = 'D:\Datasets\DomianAdaptation\Office_CalTech\decaf6'
    fn = 'dslr_decaf.mat'
    loaddata = sio.loadmat(os.path.join(pathname, fn))
    Data = loaddata['feas']
    label = loaddata['labels']
    return Data, label 
    
def webcam_decaf_dataload():
    import numpy as np
    import scipy.io as sio
    import os
    pathname = 'D:\Datasets\DomianAdaptation\Office_CalTech\decaf6'
    fn = 'webcam_decaf.mat'
    loaddata = sio.loadmat(os.path.join(pathname, fn))
    Data = loaddata['feas']
    label = loaddata['labels']
    return Data, label 
    
def mnist_usps_decaf_dataload():
    import numpy as np
    import scipy.io as sio
    import os
    pathname = 'D:\Datasets\DomianAdaptation\Digits\MNIST_USPS\mnist+usps\mnist+usps'
    fn = 'MNIST_vs_USPS.mat'
    loaddata = sio.loadmat(os.path.join(pathname, fn))
    Data_source = loaddata['X_src']
    Data_target = loaddata['X_tar']
    label_source = loaddata['Y_src']
    label_target =loaddata['Y_tar']
    return Data_source, label_source, Data_target, label_target    
    
def usps_mnist_dataload():
    import numpy as np
    import scipy.io as sio
    import os
    pathname = '/home/damodara/OT/DA/datasets/small_mnist_usps'
    fn = 'USPS_vs_MNIST.mat'
    loaddata = sio.loadmat(os.path.join(pathname, fn))
    Data_source = loaddata['X_src']
    Data_target = loaddata['X_tar']
    label_source = loaddata['Y_src']
    label_target =loaddata['Y_tar']
    return Data_source, label_source, Data_target, label_target 
def mnist_usps_dataload(): 
    import numpy as np
    import scipy.io as sio
    import os
    pathname = '/home/damodara/OT/DA/datasets/small_mnist_usps'
    fn = 'MNIST_vs_USPS.mat'
    loaddata = sio.loadmat(os.path.join(pathname, fn))
    Data_source = loaddata['X_src']
    Data_target = loaddata['X_tar']
    label_source = loaddata['Y_src']
    label_target =loaddata['Y_tar']
    return Data_source, label_source, Data_target, label_target 
    
def SVHN_dataload():
    import numpy as np
    import scipy.io as sio
    import os
    pathname ='/home/damodara/OT/DA/datasets/SVHN'
    fn = 'train_32x32.mat'
    loaddata = sio.loadmat(os.path.join(pathname, fn))
    Traindata = loaddata['X']
    trainlabel = loaddata['y']
    Traindata = np.rollaxis(Traindata, 3, 0)
    fn = 'test_32x32.mat'
    loadtdata = sio.loadmat(os.path.join(pathname, fn))
    Testdata = loadtdata['X']
    testlabel = loadtdata['y']
    Testdata = np.rollaxis(Testdata, 3, 0)
    return Traindata, trainlabel, Testdata, testlabel


# %% MNIST-M load
def mnist_m_dataload():
    import pickle as pkl
    import numpy as np
    import os
    from scipy.misc import imread

    img_path = '/home/damodara/OT/DA/datasets/mnist-m'
    train_path = os.path.join(img_path, 'mnistm_data_keras.pkl')
    mnist_m_train = pkl.load(open(train_path, 'rb'))
    Traindata = mnist_m_train['train']
    train_label = mnist_m_train['trainlabel']

    Testdata = mnist_m_train['test']
    test_label = mnist_m_train['testlabel']

    # Testdata = mnist_m_train['test']
    # test_label = mnist_m_train['testlabel']
    # train_path = os.path.join(img_path, 'mnist-m_train.pkl')
    # mnist_m_train = pkl.load(open(train_path, 'rb'))
    # Traindata = mnist_m_train['Traindata']
    # train_label = mnist_m_train['train_label']
    #
    # test_path = os.path.join(img_path, 'mnist-m_test.pkl')
    # mnist_m_test = pkl.load(open(test_path, 'rb'))
    # Testdata = mnist_m_test['Testdata']
    # Testdata = np.array(Testdata)
    # test_label = mnist_m_test['test_label']

    return Traindata, train_label, Testdata, test_label


# %% Synthetic digits
def synthetic_digits_small_dataload():
    import os
    import scipy.io as sio
    import numpy as np

    filepath = '/home/damodara/OT/DA/datasets/SynthDigits'
    train_fname = os.path.join(filepath, 'synth_train_32x32_small.mat')
    loaddata = sio.loadmat(train_fname)
    Traindata = loaddata['X']
    train_label = loaddata['y']
    #
    test_fname = os.path.join(filepath, 'synth_test_32x32_small.mat')
    loaddata = sio.loadmat(test_fname)
    Testdata = loaddata['X']
    test_label = loaddata['y']
    Traindata = np.rollaxis(Traindata, 3, 0)
    Testdata = np.rollaxis(Testdata, 3, 0)

    return Traindata, train_label, Testdata, test_label


def synthetic_digits_dataload():
    import os
    import scipy.io as sio
    import numpy as np

    filepath = '/home/damodara/OT/DA/datasets/SynthDigits'
    train_fname = os.path.join(filepath, 'synth_train_32x32.mat')
    loaddata = sio.loadmat(train_fname)
    Traindata = loaddata['X']
    train_label = loaddata['y']
    #
    test_fname = os.path.join(filepath, 'synth_test_32x32.mat')
    loaddata = sio.loadmat(test_fname)
    Testdata = loaddata['X']
    test_label = loaddata['y']

    Traindata = np.rollaxis(Traindata, 3, 0)
    Testdata = np.rollaxis(Testdata, 3, 0)
    return Traindata, train_label, Testdata, test_label


# %% stl9
def stl10_dataload():
    import os
    import scipy.io as sio
    import numpy as np

    filepath = '/home/damodara/OT/DA/datasets/stl10'
    train_fname = os.path.join(filepath, 'stl10_train.mat')
    loaddata = sio.loadmat(train_fname)
    Traindata = loaddata['X']
    train_label = loaddata['y']
    #
    test_fname = os.path.join(filepath, 'stl10_test.mat')
    loaddata = sio.loadmat(test_fname)
    Testdata = loaddata['X']
    test_label = loaddata['y']

    return Traindata, train_label, Testdata, test_label

# %% Office 31 original datasets
def office_31_dataload(dataname='amazon'):
    from scipy.misc import imread, imresize
    import matplotlib.pylab as plt
    import os
    import numpy as np

    pathname = os.path.join('/home/damodara/OT/DA/datasets/office31',
                            dataname)

    images = []
    label = []
    count = -1
    l = -1

    files_path = os.path.join(pathname, 'images')

    img_files = os.listdir(files_path)
    for imgf in img_files:
        l = l + 1
        img_names = os.listdir(os.path.join(files_path, imgf))

        for i in img_names:
            count = count + 1
            tmp = imread(os.path.join(files_path, imgf, i))
            if tmp.shape[1] != 300:
                tmp = imresize(tmp, (300, 300, 3))

            images.append(tmp)
            label.append(l)

    return np.array(images), label
#%% Regression datasets
    
def census_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:/PostDocWork/LSML/RandomFourierFeatures/Datasets/census/census/census.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['Xtrain']
    TrainData=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    TrainData=TrainData.T
    train_label=np.squeeze(adult['ytrain'])
    Dummy1=adult['Xtest']
    TestData = csc_matrix(Dummy1,shape=Dummy1.shape).toarray()
    TestData=TestData.T
    test_label = np.squeeze(adult['ytest'])
    del Dummy, Dummy1
    return TrainData, train_label, TestData, test_label

def cpu_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:/PostDocWork/LSML/RandomFourierFeatures/Datasets/cpu/cpu/cpu.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['Xtrain']
    TrainData=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    TrainData=TrainData.T
    train_label=np.squeeze(adult['ytrain'])
    Dummy1=adult['Xtest']
    TestData = csc_matrix(Dummy1,shape=Dummy1.shape).toarray()
    TestData=TestData.T
    test_label = np.squeeze(adult['ytest'])
    del Dummy, Dummy1
    return TrainData, train_label, TestData, test_label
    
def YearPredictionMSD_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:\PostDocWork\LSML\RandomFourierFeatures\Datasets\YearMSD\YearPredictionMSD.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['Xtrain']
    TrainData=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    train_label=np.squeeze(adult['ytrain'])
    Dummy1=adult['Xtest']
    TestData = csc_matrix(Dummy1,shape=Dummy1.shape).toarray()
    test_label = np.squeeze(adult['ytest'])
    return TrainData, train_label, TestData, test_label
    
def cpusmall_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:\PostDocWork\LSML\RandomFourierFeatures\Datasets\cpusmall\cpusmall.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['Data']
    Data=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    label=np.squeeze(adult['label'])    
    return Data, label
    
def cadata_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:\PostDocWork\LSML\RandomFourierFeatures\Datasets\cadata\cadata.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['Data']
    Data=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    label=np.squeeze(adult['label'])    
    return Data, label    