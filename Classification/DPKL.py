import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import copy
from utils import data_splitter, normalize
from computations import get_logits,get_accuracy,get_embedders,get_rff,get_rffweights,svgd_update,grad_update

num_classes = 10
num_epochs = 100
latent_dim = 64
num_samples = 10
batch_size = 32
norm_flag = True #Whether to normalize embeddings and weights for computing logits
alpha = 1.0 #Scaling factor
reg = 1e-3
D = 1000
num_blocks = 10
# latent_weights = tf.constant(tf.random.normal(mean=0.0,stddev=tf.sqrt(alpha),shape=[D,latent_dim]))

def train_step(X,y,embedders_list,weights_list,optimizer,num_lab,latent_weights):
    with tf.GradientTape(persistent=True) as tape:
        cce = tf.losses.CategoricalCrossentropy(from_logits=False)
        # Forward pass
        embeddings = [embedder(X) for embedder in embedders_list]
        embeddings_rff = get_rff(embeddings,latent_weights) 
        losses_list = [] #Modelwise loss
        reg_list = []
        probs_list = []
        for weights in weights_list:
            weights_rff = get_rff([weights],latent_weights)
            probs_num = tf.matmul(embeddings_rff,weights_rff,transpose_b=True)
            probs = tf.transpose(tf.transpose(probs_num) / tf.reduce_sum(probs_num, axis=1))
            probs_list.append(probs)
            losses_list.append(cce(y, probs))
            reg_list.append(reg*tf.reduce_sum(tf.square(weights)))
        #Compute Loss
        train_probs = tf.reduce_mean(tf.stack(probs_list),axis=0)
        negll = cce(y, train_probs) #Overall loss
    #embedder_update
    params_list = [embedder.trainable_variables for embedder in embedders_list]
    grads_list = [tape.gradient(negll,embedder.variables) for embedder in embedders_list]
    if num_samples>1:
    # Update gradients
        grads_list = grad_update(grads_list,params_list,num_samples)
    #Apply gradients
    for (grads,params) in zip(grads_list,params_list):
        optimizer.apply_gradients(zip(grads,params))
    #Weights update
    params_list = [[weights] for weights in weights_list]
    losses_grads_list = [tape.gradient(loss,weights) for (loss,weights) in zip(losses_list,weights_list)]
    reg_grads_list = [tape.gradient(reg,weights) for (reg,weights) in zip(reg_list,weights_list)]
    sf = float(num_lab/batch_size)
    grads_list = [[reg_grad + sf*loss_grad] for (loss_grad,reg_grad) in zip(losses_grads_list,reg_grads_list)] #Total gradient
    if num_samples>1:
    #Update gradients
        grads_list = svgd_update(grads_list,params_list,num_samples)
    # Apply gradients
    for (grads,params) in zip(grads_list,params_list):
        optimizer.apply_gradients(zip(grads,params))
    return embedders_list, weights_list, optimizer

def val_step(X_val,y_val,embedders_list,weights_list,latent_weights):
    embeddings = [embedder(X_val) for embedder in embedders_list]
    embeddings_rff = get_rff(embeddings,latent_weights) 
    probs_list = []
    for weights in weights_list:
        weights_rff = get_rff([weights],latent_weights)
        probs_num = tf.matmul(embeddings_rff,weights_rff,transpose_b=True)
        probs = tf.transpose(tf.transpose(probs_num) / tf.reduce_sum(probs_num, axis=1))
        probs_list.append(probs)
    val_probs = tf.reduce_mean(tf.stack(probs_list),axis=0)
    val_acc_check = get_accuracy(val_probs,y_val).numpy()
    return val_acc_check

def train(X,y,X_val,y_val,latent_weights):
    #Trains the model on data X and labels y
    num_lab = X.shape[0]
    val_acc = 0.0
    dataset = tf.data.Dataset.from_tensor_slices((X,y))
    dataset = dataset.shuffle(1000).batch(batch_size)
    optimizer = tf.keras.optimizers.Nadam(learning_rate = 1e-3, name='Nadam')
    embedders_list = get_embedders(num_samples,latent_dim)
    weights_list = [tf.Variable(tf.random.normal((num_classes,latent_dim))) for _ in range(num_samples)]
    for epoch in range(num_epochs):
        #Training
        for (batch, (points, targets)) in enumerate(dataset):
            embedders_list, weights_list, optimizer = train_step(points,targets,embedders_list,weights_list,optimizer,num_lab,latent_weights)
        #Validation
        if epoch%10 == 0 or epoch == num_epochs-1:
            val_acc_check = val_step(X_val,y_val,embedders_list,weights_list,latent_weights)
            # print ([epoch, val_acc_check])
            if val_acc_check >= val_acc:
                val_acc = val_acc_check
                best_embedders = [embedder.copy(X_val) for embedder in embedders_list]
                best_weights = [weights.numpy() for weights in weights_list]
            else:
                continue #Don't save if performance worsens
    return best_embedders, best_weights

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train = tf.one_hot(y_train,depth=num_classes,axis=-1).numpy()
y_test = tf.one_hot(y_test,depth=num_classes,axis=-1).numpy()
X_train = X_train.reshape((X_train.shape[0],-1))/255.0
X_train = X_train.astype('float32')
X_test = X_test.reshape((X_test.shape[0],-1))/255.0
X_test = X_test.astype('float32')

num_trials = 10
num_lab_max = 1000
num_acq = 100
acc = np.zeros((10,num_trials))
negll = np.zeros((10,num_trials))
    
for trial in range(num_trials):
    print ('Trial = '+str(trial))
    num_lab = 100
    ctr = 0
    while num_lab<=num_lab_max:
        num_val = int(0.1*num_lab)
        X_lab, y_lab, X_unlab, y_unlab = data_splitter(X_train, y_train, num_lab) #Splitting into labeled and unlabeled data
        X_val, y_val, X, y = data_splitter(X_lab,y_lab, num_val)
        latent_weights = tf.convert_to_tensor(get_rffweights(latent_dim,num_blocks,np.sqrt(alpha)),dtype=tf.float32)
        embedders_list, weights_list = train(X,y,X_val,y_val,latent_weights)  
        #Checking accuracy on test data
        embeddings = [embedder(X_test) for embedder in embedders_list]
        embeddings_rff = get_rff(embeddings,latent_weights) 
        probs_list = []
        for weights in weights_list:
            weights_rff = get_rff([weights],latent_weights)
            probs_num = tf.matmul(embeddings_rff,weights_rff,transpose_b=True)
            probs = tf.transpose(tf.transpose(probs_num) / tf.reduce_sum(probs_num, axis=1))
            probs_list.append(probs)
        test_probs = tf.reduce_mean(tf.stack(probs_list),axis=0)
        test_acc = get_accuracy(test_probs,y_test).numpy()
        cce = tf.losses.CategoricalCrossentropy(from_logits=False)
        test_negll = cce(y_test,test_probs).numpy()
        print ("Num_lab = "+str(num_lab)+" Test accuracy = "+str(test_acc))
        acc[ctr][trial] = test_acc
        negll[ctr][trial] = test_negll
        num_lab += num_acq
        ctr += 1
print (np.mean(acc,axis=1))
print (np.mean(negll,axis=1))
acc_filename = 'Results/acc_dpkl.npy'
np.save(acc_filename,acc)
negll_filename = 'Results/negll_dpkl.npy'
np.save(negll_filename,negll)
print ('Results Saved')