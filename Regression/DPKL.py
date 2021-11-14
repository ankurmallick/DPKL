import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
import numpy as np
import copy
from utils import *
from computations import *

gamma = 0.5
D = 100

class DPKL(object):
    
    def __init__(self, hp_dict):
        self.gp_reg = hp_dict['GP_reg']
        self.ss_reg = hp_dict['SS_reg']
        self.lr = hp_dict['lr']
        self.latent_dim = hp_dict['latent_dim']
        self.num_iters = hp_dict['num_iters']
        self.num_samples = hp_dict['num_samples']
        self.bw = 2.0
        self.latent_weights = np.sqrt(2*gamma)*np.random.normal(size=(D,self.latent_dim))
        self.latent_weights = self.latent_weights.astype('float32')
#         self.kernel_batch_size = 10000
        self.norm_params = None
    
    def validate(self,X_val,y_val):
    #Computes RMSE on Validation data
        val_mean, Kmat_val = self.predict(X_val)
        val_rmse = np.sqrt(np.mean(np.square(y_val - val_mean)))
        return val_rmse

    def fit(self, features, labels, unlab_features=None):
        num_val = int(0.1*features.shape[0])
        X_val, y_val, features, labels = data_splitter(features, labels, num_val)
        self.labels, self.norm_params = normalize(labels)
        embedders_list = get_embedders(self.num_samples,self.latent_dim)
        optimizer = tf.keras.optimizers.Nadam(learning_rate = self.lr, name='Nadam')
        reg_mat = (self.gp_reg**2)*tf.eye(tf.shape(self.labels)[0])
        val_rmse_check = float('Inf')
        for itr in range(self.num_iters):
            with tf.GradientTape(persistent=True) as tape:
                embeddings = project([embedder(features) for embedder in embedders_list],self.latent_weights)
                Kmat_reg = tf.matmul(embeddings,embeddings,transpose_b=True) + reg_mat
                if unlab_features is None:
                    loss = get_negll(Kmat_reg,self.labels)
                else:
                    unlab_embeddings = project([embedder(unlab_features) for embedder in embedders_list],self.latent_weights)
                    lab_unlab_kern = tf.matmul(embeddings,unlab_embeddings,transpose_b=True)
                    negll, pred_var = get_negll(Kmat_reg,self.labels,lab_unlab_kern)
                    lab_loss = negll/features.shape[0]
                    unlab_loss = tf.reduce_sum(pred_var)/unlab_features.shape[0]
                    loss =  lab_loss + self.ss_reg*unlab_loss
            params_list = [embedder.variables for embedder in embedders_list]
            grads_list = [tape.gradient(loss,embedder.variables) for embedder in embedders_list]
            if self.num_samples>1:
            #Update gradients
                grads_list = grad_update(grads_list,params_list,self.num_samples)
            #Apply gradients
            for (grads,params) in zip(grads_list,params_list):
                optimizer.apply_gradients(zip(grads,params))
            if itr % 10 == 0 or itr == self.num_iters-1:
                temp = copy.deepcopy(self)
                temp.embedders = [embedder.copy(features) for embedder in embedders_list]
                temp.train_embeddings = project([embedder(features) for embedder in temp.embedders],temp.latent_weights)
                temp.Kmat_reg = tf.matmul(temp.train_embeddings,temp.train_embeddings,transpose_b=True) + reg_mat
                val_rmse = temp.validate(X_val,y_val)
                if val_rmse > val_rmse_check:
                    break
                else:
                    self.embedders = [embedder.copy(features) for embedder in embedders_list]
                    self.train_embeddings = temp.train_embeddings
                    self.Kmat_reg = temp.Kmat_reg
                    val_rmse_check = val_rmse
    
    def predict(self, X):
        test_embeddings = project([embedder(X) for embedder in self.embedders],self.latent_weights)
        train_test_kern = tf.matmul(self.train_embeddings,test_embeddings,transpose_b=True) #<Train, Test>
        test_test_kern = tf.matmul(test_embeddings,test_embeddings,transpose_b=True) #<Test, Test>
        pred_mean_norm, Kmat_test = GP_out(self.Kmat_reg,train_test_kern,test_test_kern,self.gp_reg,self.labels)
        pred_mean = unnormalise(self.norm_params,pred_mean_norm)
        return pred_mean, Kmat_test