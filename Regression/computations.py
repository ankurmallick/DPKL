import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import copy
from EmbedModel import EmbedModel
from scipy.stats import norm
from scipy.special import logsumexp

def get_dist(params_0,params_1,num_params):
    dist = 0
    for i in range(num_params):
        dist += tf.norm(params_0[i] - params_1[i])
    return dist

def create_tensor(X, datatype=None):
    if datatype is None:
        datatype = tf.float32
    return tf.convert_to_tensor(X, dtype=datatype)
    
def scaled_sqdist(X,X2=None,lengthscales=None):
    #Returns scaled squared distance between X and X2
    if lengthscales is None:
        lengthscales = tf.ones([tf.shape(X)[1],1])
    lengthscales_X = tf.matmul(tf.ones([tf.shape(X)[0],1]),lengthscales,transpose_b=True)
    X = tf.divide(X,lengthscales_X)
    Xs = tf.reduce_sum(tf.square(X), axis=1)
    if X2 is None:
        dist = -2 * tf.matmul(X, X, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1))  + tf.reshape(Xs, (1, -1))
        return dist
    lengthscales_X2 = tf.matmul(tf.ones([tf.shape(X2)[0],1]),lengthscales,transpose_b=True)
    X2 = tf.divide(X2,lengthscales_X2)
    X2s = tf.reduce_sum(tf.square(X2), axis=1)
    dist = -2 * tf.matmul(X, X2, transpose_b=True)
    dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
    dist = tf.clip_by_value(dist,1e-5,float('Inf'))
    return dist

def get_sq_median(v):
    #Returns median for computing model kernel bandwidth
    v = tf.reshape(v, [-1])
    m_ = tf.shape(v)[0]//2
    return tf.nn.top_k(v, m_).values[m_-1]

def get_model_kernel(model_features,num_models):
    #Calculates kernel matrix for models stored in model_list
    dist_mat = scaled_sqdist(model_features)
    sq_med = get_sq_median(dist_mat) #Median distance
    h = (sq_med)/tf.math.log(tf.cast(num_models,tf.float32))
    return tf.exp((0-dist_mat)/h)

def get_func_grads(grads_tensor,embeddings_tensor):
    data_len = embeddings_tensor.shape[0]
    num_samples = embeddings_tensor.shape[-1]
    embeddings_list = tf.split(embeddings_tensor,data_len,axis=0)
    grads_list = tf.split(grads_tensor,data_len,axis=0)
    kerns_list = [get_model_kernel(tf.transpose(tf.squeeze(emb)),num_samples) for emb in embeddings_list]
    func_grads_tensor = tf.stack([tf.matmul(tf.squeeze(gm),km) for (gm,km) in zip(grads_list,kerns_list)],axis=0)
    func_grads_list = tf.split(func_grads_tensor,num_samples,axis=2)
    return [tf.squeeze(fg) for fg in func_grads_list]

def get_shapes_list(params):
    #Returns shape of each tensor in params
    shapes_list = []
    for var_tensor in params:
        shapes_list.append(tf.shape(var_tensor))
    return shapes_list

def params_to_vect(params):
    #Converts a list of tensors (model parameters) to a 1D vector
    vect_list = []
    for var_tensor in params:   
        vect_list.append(tf.reshape(var_tensor,[-1,1]))
    return tf.concat(vect_list,0)

def vect_to_params(vect,shapes_list):
    #Converts a 1D vector of model parameters to the corresponding list of tensors
    start = 0
    params_list = []
    for shape in shapes_list:
        size = tf.reduce_prod(shape)
        var_vect = vect[start:start+size]
        params_list.append(tf.reshape(var_vect,shape))
        start+=size
    return params_list

def grad_update(grads_list,params_list,num_samples):
    #Returns the kernelized average gradient for each model parameter
    shapes_list = get_shapes_list(params_list[0]) #Shape of each variable in embedder
    params_to_vect_list = [params_to_vect(params) for params in params_list]
    features = tf.transpose(tf.concat(params_to_vect_list,1))
    kernel = get_model_kernel(features,num_samples)
    grads_mat = tf.concat([params_to_vect(grads) for grads in grads_list],1)
    newgrads_mat = tf.matmul(grads_mat,kernel)
    newgrads_list = [vect_to_params(vect,shapes_list) for vect in tf.split(newgrads_mat,int(tf.shape(newgrads_mat)[1].numpy()),1)]
    return newgrads_list

def get_rffweights(latent_dim,num_blocks,stddev):
    latent_weights=[]
    for block in range(num_blocks):
        mat = np.random.normal(size=(latent_dim,latent_dim))
        Q, R = np.linalg.qr(mat)
        S = np.diag(np.sqrt(np.random.chisquare(latent_dim,latent_dim)))
        latent_weights.append(np.matmul(S,Q)/stddev)
    return np.concatenate(latent_weights,axis=0)
    
def project(embeddings_list,latent_weights):
    avg_embedding = 0
    D = latent_weights.shape[0]
    num_samples = len(embeddings_list)
    for embedding in embeddings_list:
        projections = tf.matmul(embedding,latent_weights,transpose_b=True)
        avg_embedding += tf.concat([tf.cos(projections),tf.sin(projections)],axis=1)/tf.sqrt(float(D))
    avg_embedding /= num_samples
    return avg_embedding

def get_embedders(num_samples,latent_dim):
    embedders_list = []
    for _ in range(num_samples):
        embedder = EmbedModel(latent_dim)
        embedders_list.append(embedder)
    return embedders_list

def get_negll(Kmat_reg,Y,lab_unlab_kern=None):
    Kmat_reg = (Kmat_reg + tf.transpose(Kmat_reg))/2.0
    s,U,V = tf.linalg.svd(Kmat_reg, full_matrices=True)
    Kmat_reg_det_log = tf.reduce_sum(tf.math.log(s)) #Direct determinant blows up for large matrix
    negll_vect = tf.matmul(U,Y,transpose_a=True)
    negll_t1 = 0.5*tf.reduce_sum(tf.square(negll_vect)/tf.reshape(s,[-1,1]))
    negll_t2 = 0.5*Kmat_reg_det_log
    negll = negll_t1 + negll_t2
    if lab_unlab_kern is None:
        return negll
    else:
        pred_var_t1 = 1 #1 since we use the SE kernel 
        pred_var_t2_part = tf.matmul(lab_unlab_kern,U,transpose_a=True)
        pred_var_t2 = tf.reduce_sum(tf.square(pred_var_t2_part)/tf.reshape(s,[1,-1]),axis=1)
        pred_var = pred_var_t1 - pred_var_t2
        return negll, pred_var

def get_covg(vect,var):
    covg_ind = 0.5*(np.log(2*np.pi) + np.log(var) + (vect**2)/var)
    return np.mean(covg_ind)
    
def GP_out(Kmat_reg,train_test_kern,test_test_kern,GP_reg,train_target=None):
    Kmat_reg = (Kmat_reg + tf.transpose(Kmat_reg))/2.0
    s,U,V = tf.linalg.svd(Kmat_reg, full_matrices=True)
    Kmat_test_part = tf.matmul(train_test_kern,U,transpose_a=True)
    Kmat_test_part_rs = tf.matmul(Kmat_test_part,tf.linalg.diag(1.0/s))
    Kmat_test = test_test_kern + (GP_reg**2)*tf.eye(tf.shape(test_test_kern)[0]) - tf.matmul(Kmat_test_part_rs,Kmat_test_part,transpose_b=True)
    pred_proj = tf.matmul(U,train_target,transpose_a=True)
    pred_mean = tf.matmul(Kmat_test_part_rs,pred_proj)
    return pred_mean, Kmat_test

def unnormalise(norm_params, y):
    return ((y * norm_params[1]) + norm_params[0]).numpy()

def predict(test_embeddings, train_embeddings, train_target, GP_reg):
    Kmat_reg = tf.exp(-0.5*scaled_sqdist(train_embeddings)) + (GP_reg**2)*tf.eye(tf.shape(train_embeddings)[0])
    train_test_kern = tf.exp(-0.5*scaled_sqdist(train_embeddings,test_embeddings)) #<Train, Test>
    test_test_kern = tf.exp(-0.5*scaled_sqdist(test_embeddings,test_embeddings)) #<Test, Test>
    return GP_out(Kmat_reg,train_test_kern,test_test_kern,GP_reg,train_target)  

def prob_predict(test_embeddings, train_embeddings, train_target, GP_reg):
    Kmat_reg = tf.matmul(train_embeddings,train_embeddings,transpose_b=True) + (GP_reg**2)*tf.eye(tf.shape(train_embeddings)[0])
    train_test_kern = tf.matmul(train_embeddings,test_embeddings,transpose_b=True) #<Train, Test>
    test_test_kern = tf.matmul(test_embeddings,test_embeddings,transpose_b=True) #<Test, Test>
    return GP_out(Kmat_reg,train_test_kern,test_test_kern,GP_reg,train_target)