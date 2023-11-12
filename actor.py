import tensorflow as tf
import numpy as np

class ActorNetwork(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim):
        super(ActorNetwork, self).__init__()
        self.inputs = tf.keras.layers.InputLayer(name='input_layer', input_shape=(3*embedding_dim,))
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim, activation='tanh') # embedding_dim = 100
        ])
        
    def call(self, x):
        x = self.inputs(x)
        return self.fc(x)

class Actor(object):
    
    def __init__(self, embedding_dim, hidden_dim, learning_rate, state_size, tau):
        
        self.embedding_dim = embedding_dim
        self.state_size = state_size
        
        # actor network /  target network
        self.network = ActorNetwork(embedding_dim, hidden_dim)
        self.target_network = ActorNetwork(embedding_dim, hidden_dim)
        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        # soft target network update hyperparameter
        self.tau = tau
        
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.call(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs    
    
    def build_networks(self):
        # Build networks
        self.network(np.zeros((1, 3*self.embedding_dim)))
        self.target_network(np.zeros((1, 3*self.embedding_dim)))
    
    def update_target_network(self):
        # soft target network update
        c_theta, t_theta = self.network.get_weights(), self.target_network.get_weights()
        for i in range(len(c_theta)):
            t_theta[i] = self.tau * c_theta[i] + (1 - self.tau) * t_theta[i]
        self.target_network.set_weights(t_theta)
        
    def train(self, states, dq_das):
        with tf.GradientTape() as g:
            outputs = self.network(states)
            # loss = outputs*dq_das
        dj_dtheta = g.gradient(outputs, self.network.trainable_weights, -dq_das)
        grads = zip(dj_dtheta, self.network.trainable_weights)
        self.optimizer.apply_gradients(grads)
        
    def save_weights(self, path):
        self.network.save_weights(path)
        
    def load_weights(self, path):
        self.network.load_weights(path)