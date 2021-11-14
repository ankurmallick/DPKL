import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
import numpy as np
from EmbedModel import EmbedModel
from utils import *
from computations import *

class DKL(object):
    
    def __init__(self, hp_dict):
        self.gp_reg = hp_dict['GP_reg']
        self.ss_reg = hp_dict['SS_reg']
        self.lr = hp_dict['lr']
        self.latent_dim = hp_dict['latent_dim']
        self.num_iters = hp_dict['num_iters']
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
        embedder = EmbedModel(self.latent_dim)
        optimizer = tf.keras.optimizers.Nadam(learning_rate = self.lr, name='Nadam')
        reg_mat = (self.gp_reg**2)*tf.eye(tf.shape(self.labels)[0])
        val_rmse_check = float('Inf')
        for itr in range(self.num_iters):
            with tf.GradientTape(persistent=True) as tape:
                embeddings = embedder(features)
                Kmat_reg = tf.exp(-0.5*scaled_sqdist(embeddings)) + reg_mat
                if unlab_features is None:
                    loss = get_negll(Kmat_reg,self.labels)
                else:
                    unlab_embeddings = embedder(unlab_features)
                    lab_unlab_kern = tf.exp(-0.5*scaled_sqdist(embeddings,unlab_embeddings))
                    negll, pred_var = get_negll(Kmat_reg,self.labels,lab_unlab_kern)
                    lab_loss = negll/features.shape[0]
                    unlab_loss = tf.reduce_sum(pred_var)/unlab_features.shape[0]
                    loss =  lab_loss + self.ss_reg*unlab_loss
            params = embedder.variables
            grads = tape.gradient(loss,embedder.variables)
            optimizer.apply_gradients(zip(grads,params))
            if itr % 10 == 0 or itr == self.num_iters-1:
                temp = copy.deepcopy(self)
                temp.embedder = embedder.copy(features)
                temp.train_embeddings = embedder(features)
                temp.Kmat_reg = tf.exp(-0.5*scaled_sqdist(temp.train_embeddings)) + reg_mat
                val_rmse = temp.validate(X_val,y_val)
                if val_rmse > val_rmse_check:
                    break
                else:
                    self.embedder = embedder.copy(features)
                    self.train_embeddings = temp.train_embeddings
                    self.Kmat_reg = temp.Kmat_reg
                    val_rmse_check = val_rmse
        
    def predict(self, X):
        test_embeddings = self.embedder(X)
        train_test_kern = tf.exp(-0.5*scaled_sqdist(self.train_embeddings,test_embeddings)) #<Train, Test>
        test_test_kern = tf.exp(-0.5*scaled_sqdist(test_embeddings,test_embeddings)) #<Test, Test>
        pred_mean_norm, Kmat_test = GP_out(self.Kmat_reg,train_test_kern,test_test_kern,self.gp_reg,self.labels)
        pred_mean = unnormalise(self.norm_params,pred_mean_norm)
        return pred_mean, Kmat_test
