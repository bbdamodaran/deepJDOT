# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 12:58:39 2018

@author: damodara
Deepjdot  - class file
"""
import dnn
import numpy as np
import ot
from scipy.spatial.distance import cdist 

class Deepjdot(object):
    def __init__(self, model, batch_size, n_class, optim, allign_loss=1.0, tar_cl_loss=1.0, 
                 sloss=0.0,tloss=1.0,int_lr=0.01, ot_method='emd',
                 jdot_alpha=0.01, lr_decay=True, verbose=1):
                     
        self.model = model   # target model
        self.batch_size = batch_size
        self.n_class= n_class
        self.optimizer= optim
        # initialize the gamma (coupling in OT) with zeros
        self.gamma=dnn.K.zeros(shape=(self.batch_size, self.batch_size))
        # whether to minimize with classification loss
        self.train_cl =dnn.K.variable(tar_cl_loss)
        # whether to minimize with the allignment loss 
        self.train_algn=dnn.K.variable(allign_loss)
        self.sloss = dnn.K.variable(sloss) # weight for source classification
        self.tloss = dnn.K.variable(tloss) # weight for target classification
        self.verbose = verbose
        self.int_lr =int_lr  # initial learning rate
        self.lr_decay= lr_decay
        #
        self.ot_method = ot_method
        self.jdot_alpha=jdot_alpha  # weight for the alpha term
        
        
        # target classification cross ent loss and source cross entropy
        def classifier_cat_loss(y_true, y_pred):
            '''
            classifier loss based on categorical cross entropy in the target domain
            1:batch_size - is source samples
            batch_size:end - is target samples
            self.gamma - is the optimal transport plan
            '''
            # source cross entropy loss
            ys = y_true[:batch_size,:] # source true labels
            ypred_t = y_pred[batch_size:,:] # target prediction
            source_ypred = y_pred[:batch_size,:]   # source prediction          
            source_loss = dnn.K.mean(dnn.K.categorical_crossentropy(ys, source_ypred))
            
            # categorical cross entropy loss
            ypred_t = dnn.K.log(ypred_t)
            # loss calculation based on double sum (sum_ij (ys^i, ypred_t^j))
            loss = -dnn.K.dot(ys, dnn.K.transpose(ypred_t))
            # returns source loss + target loss
            return self.train_cl*(self.tloss*dnn.K.sum(self.gamma * loss) + self.sloss*source_loss)
        self.classifier_cat_loss = classifier_cat_loss
        
        # L2 distance
        def L2_dist(x,y):
            '''
            compute the squared L2 distance between two matrics
            '''
            dist = dnn.K.reshape(dnn.K.sum(dnn.K.square(x),1), (-1,1))
            dist += dnn.K.reshape(dnn.K.sum(dnn.K.square(y),1), (1,-1))
            dist -= 2.0*dnn.K.dot(x, dnn.K.transpose(y))  
            return dist
            
       # feature allignment loss
        def align_loss(y_true, y_pred):
            '''
            source and target alignment loss in the intermediate layers of the target model
            allignment is performed in the target model (both source and target features are from target model)
            y-true - is dummy value( that is full of zeros)
            y-pred - is the value of intermediate layers in the target model
            1:batch_size - is source samples
            batch_size:end - is target samples            
            '''
            # source domain features            
            gs = y_pred[:batch_size,:]
            # target domain features
            gt = y_pred[batch_size:,:]
            gdist = L2_dist(gs,gt)  
            return self.jdot_alpha * dnn.K.sum(self.gamma * (gdist))
        self.align_loss= align_loss
        
        def feature_extraction(model, data, out_layer_num=-2):
            '''
            extract the features from the pre-trained model
            inp_layer_num - input layer
            out_layer_num -- from which layer to extract the features
            '''
            intermediate_layer_model = dnn.Model(inputs=model.layers[1].layers[1].input,
                             outputs=model.layers[1].layers[out_layer_num].output)
            intermediate_output = intermediate_layer_model.predict(data)
            return intermediate_output
        self.feature_extraction = feature_extraction
 

 
    def fit(self, source_traindata, ys_label, target_traindata, target_label = None,
            n_iter=5000, cal_bal=True, sample_size=None):
        '''
        source_traindata - source domain training data
        ys_label - source data true labels
        target_traindata - target domain training data
        cal_bal - True: source domain samples are equally represented from
                        all the classes in the mini-batch (that is, n samples from each class)
                - False: source domain samples are randomly sampled
        target_label - is not None  : compute the target accuracy over the iterations
        '''
      
        ns = source_traindata.shape[0]
        nt= target_traindata.shape[0]
        method=self.ot_method # for optimal transport
        alpha=self.jdot_alpha
        fe_size = self.model.output_shape[1][1]
        t_acc = []
        t_loss =[]
        tloss = dnn.K.eval(self.tloss)
        g_metric ='deep' # to allign in intermediate layers, when g_metric='original', the
         # alignment loss is performed wrt original input features  (StochJDOT)
        
        # function to sample n samples from each class
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
            
         # target model compliation and optimizer
        self.model.compile(optimizer= self.optimizer, loss =[self.classifier_cat_loss, self.align_loss])
        # set the learning rate
        dnn.K.set_value(self.model.optimizer.lr, self.int_lr) 
        
        for i in range(n_iter):
            
            if self.lr_decay and i%5000 ==0:
                # p = float(i) / n_iter
                # lr = self.int_lr / (1. + 10 * p)**0.9
                lr = dnn.K.get_value(self.model.optimizer.lr)
                dnn.K.set_value(self.model.optimizer.lr, lr*0.1)
             
            # source domain mini-batch indexes
            if cal_bal:
                s_ind = mini_batch_class_balanced(ys_label, sample_size=sample_size)
                self.sbatch_size = len(s_ind)
            else:
                s_ind = np.random.choice(ns, self.batch_size)
                self.sbatch_size = self.batch_size
                # target domain mini-batch indexes
            t_ind = np.random.choice(nt, self.batch_size)

            # source and target domain mini-batch samples 
            xs_batch, ys = source_traindata[s_ind], ys_label[s_ind]
            xt_batch = target_traindata[t_ind]

             # dummy target outputs for the keras model
            l_dummy = np.zeros_like(ys)  # for target samples
               # for intermediate layer feature values in the target model
            g_dummy = np.zeros((2*self.batch_size, fe_size)) 
            s = xs_batch.shape
            
            # concat of source and target samples and prediction
            modelpred = self.model.predict(np.vstack((xs_batch, xt_batch)))
           
            # modelpred[0] - is softmax prob, and modelpred[1] - is intermediate layer
            gs_batch = modelpred[1][:self.batch_size, :]
            gt_batch = modelpred[1][self.batch_size:, :]
            # softmax prediction of target samples
            ft_pred = modelpred[0][self.batch_size:,:]
            
            
            if g_metric=='orginal':
                # compution distance metric in the image space
                if len(s) == 3:  # when the input is image, convert into 2D matrix
                    C0 = cdist(xs_batch.reshape(-1, s[1] * s[2]), xt_batch.reshape(-1,
                                                                                   s[1] * s[2]), metric='sqeuclidean')

                elif len(s) == 4:
                    C0 = cdist(xs_batch.reshape(-1, s[1] * s[2] * s[3]), xt_batch.reshape(-1,                                                                                          s[1] * s[2] * s[3]),metric='sqeuclidean')
            else:
                # distance computation between source and target in deep layer
                C0 = cdist(gs_batch, gt_batch, metric='sqeuclidean')

            # ground metric for the target classification loss
            C1 = cdist(ys, ft_pred, metric='sqeuclidean')
            
            # JDOT ground metric
            C= alpha*C0+C1
                             
            # JDOT optimal coupling (gamma)
            
            if method == 'emd':
                 gamma=ot.emd(ot.unif(gs_batch.shape[0]),ot.unif(gt_batch.shape[0]),C)
            
            # update the computed gamma                      
            dnn.K.set_value(self.gamma, gamma)
            
            # train the keras model on batch
            data = np.vstack((xs_batch, xt_batch))    
            hist= self.model.train_on_batch([data], [np.vstack((ys,l_dummy)), g_dummy])
            
            t_loss.append(hist[0])
            if self.verbose:
                if i%10==0:
                   print ('tl_loss ={:f}, fe_loss ={:f},  tot_loss={:f}'.format(hist[1],
                          hist[2], hist[0]))
                   if target_label is not None:
                       tpred = self.model.predict(target_traindata)[0]
                       t_acc.append(np.mean(np.argmax(target_label,1)==np.argmax(tpred,1)))
                       print('Target acc\n', t_acc[-1])
        return hist, t_loss, t_acc
            
        

    def predict(self, data):
        ypred = self.model.predict(data)
        return ypred

    def evaluate(self, data, label):
        ypred = self.model.predict(data)
        score = np.mean(np.argmax(label,1)==np.argmax(ypred[0],1))
        return score