import tensorflow as tf

class EmbedModel(tf.keras.Model):
    def __init__(self,latent_dim):
        super(EmbedModel, self).__init__()
        self.latent_dim = latent_dim
        self.enc_fc1 = tf.keras.layers.Dense(100)
        self.enc_fc2 = tf.keras.layers.Dense(50)
        self.enc_fc3 = tf.keras.layers.Dense(50)
        self.enc_fc4 = tf.keras.layers.Dense(self.latent_dim)
        
    def embed(self, x):
        h = tf.nn.relu(self.enc_fc1(x))
        h = tf.nn.relu(self.enc_fc2(h))
        h = tf.nn.relu(self.enc_fc3(h))
        return self.enc_fc4(h)

    def call(self, inputs):
        latent = self.embed(inputs)
        return latent

    def copy(self,data):
        new = EmbedModel(self.latent_dim)
        embedding = new(data)
        new.enc_fc1.set_weights(self.enc_fc1.get_weights())
        new.enc_fc2.set_weights(self.enc_fc2.get_weights())
        new.enc_fc3.set_weights(self.enc_fc3.get_weights())
        new.enc_fc4.set_weights(self.enc_fc4.get_weights())
        return new