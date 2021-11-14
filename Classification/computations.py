import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import copy
from EmbedModel import EmbedModel

def get_dist(params_0,params_1,num_params):
    dist = 0
    for i in range(num_params):
        dist += tf.norm(params_0[i] - params_1[i])
    return dist

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
    m = tf.shape(v)[0]//2
    return tf.nn.top_k(v, m).values[m-1]

def get_model_kernel(model_features,num_models):
    #Calculates kernel matrix for models stored in model_list
    dist_mat = scaled_sqdist(model_features)
    sq_med = get_sq_median(dist_mat) #Median distance
    h = (sq_med)/tf.math.log(tf.cast(num_models,tf.float32))
    return tf.exp((0-dist_mat)/h), h

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
    kernel, h = get_model_kernel(features,num_samples)
    grads_mat = tf.concat([params_to_vect(grads) for grads in grads_list],1)
    newgrads_mat = tf.matmul(grads_mat,kernel)
    newgrads_list = [vect_to_params(vect,shapes_list) for vect in tf.split(newgrads_mat,int(tf.shape(newgrads_mat)[1].numpy()),1)]
    return newgrads_list

def svgd_update(grads_list,params_list,num_samples):
    #Returns the kernelized average gradient for each model parameter
    shapes_list = get_shapes_list(params_list[0]) #Shape of each variable in embedder
    params_to_vect_list = [params_to_vect(params) for params in params_list]
    features = tf.concat(params_to_vect_list,1)
    kernel, h = get_model_kernel(tf.transpose(features),num_samples)
    grads_mat = tf.concat([params_to_vect(grads) for grads in grads_list],1)
    newgrads_mat = tf.matmul(grads_mat,kernel) - (2.0/h)*(tf.matmul(features,kernel) - features*tf.reduce_sum(kernel,axis=1))
    newgrads_mat = newgrads_mat/num_samples
    newgrads_list = [vect_to_params(vect,shapes_list) for vect in tf.split(newgrads_mat,int(tf.shape(newgrads_mat)[1].numpy()),1)]
    return newgrads_list

def get_accuracy(y_pred,y_true):
    prediction = tf.argmax(y_pred, 1)
    labels = tf.argmax(y_true, 1)
    return tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))

def get_rmse(y_pred,y_true):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def get_rff(points_list,latent_weights):
    num_samples = len(points_list)
    D = tf.shape(latent_weights)[0].numpy()
    avg_embedding = 0
    for points in points_list:
        projections = tf.matmul(tf.math.l2_normalize(points, axis=1),latent_weights,transpose_b=True)
        avg_embedding += tf.concat([tf.cos(projections),tf.sin(projections)],axis=1)/tf.sqrt(float(D))
    avg_embedding /= num_samples
    return avg_embedding

def get_embedders(num_samples,latent_dim):
    embedders_list = []
    for _ in range(num_samples):
        embedder = EmbedModel(latent_dim)
        embedders_list.append(embedder)
    return embedders_list

def get_logits(embeddings,weights,norm_flag=True):
    if norm_flag:
        #Returns normalized logits by default
        norm_embeddings = tf.math.l2_normalize(embeddings, axis=1) 
        norm_weights = tf.math.l2_normalize(weights, axis=1)
        return tf.matmul(norm_embeddings,norm_weights,transpose_b=True)
    else:
        return tf.matmul(embeddings,weights,transpose_b=True)

def get_probs(X, embedders_list, weights_list, latent_weights):
    embeddings = [embedder(X) for embedder in embedders_list]
    embeddings_rff = get_rff(embeddings,latent_weights)
    weights_rff = get_rff(weights_list,latent_weights)
    probs = tf.matmul(embeddings_rff,weights_rff,transpose_b=True)
    return tf.transpose(tf.transpose(probs) / tf.reduce_sum(probs, axis=1))

def get_rffweights(latent_dim,num_blocks,stddev):
    latent_weights=[]
    for block in range(num_blocks):
        mat = np.random.normal(size=(latent_dim,latent_dim))
        Q, R = np.linalg.qr(mat)
        S = np.diag(np.sqrt(np.random.chisquare(latent_dim,latent_dim)))
        latent_weights.append(np.matmul(S,Q)/stddev)
    return np.concatenate(latent_weights,axis=0)
        