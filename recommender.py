from replay_buffer import PriorityExperienceReplay 
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_math_ops import Exp

from actor import Actor
from actor2 import Actor2
from critic import Critic
from critic2 import Critic2
from replay_memory import ReplayMemory       # Getting sample batch data for training
from embedding import MovieGenreEmbedding, UserMovieEmbedding
from state_representation import DRRAveStateRepresentation

import matplotlib.pyplot as plt

import wandb
import torch

class GaussianNoise:
    """Gaussian Noise.
    Taken from https://github.com/vitchyr/rlkit
    """

    def __init__(
        self,
        action_dim= 50,
        min_sigma = 0.2,
        max_sigma = 0.2,
        decay_period = 1000000,
    ):
        """Initialize."""
        self.action_dim = action_dim
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def sample(self, t=0):
        """Get an action with gaussian noise."""
        sigma = self.max_sigma[0] - np.diff(int(self.max_sigma[0]) , int(self.min_sigma[0])) * min(1.0, t / self.decay_period)
        return np.random.normal(0, sigma, size=self.action_dim)

class DRRAgent:
    
    def __init__(self, env, users_num, items_num, state_size, is_test=False, use_wandb=False):
        # users_num -> Total users + 1
        # items_num -> Total movies + 1
        self.env = env

        self.users_num = users_num
        self.items_num = items_num
        
        self.embedding_dim = 50
        self.actor_hidden_dim = 128
        self.actor_learning_rate = 0.0001
        
        self.critic_hidden_dim = 128
        self.critic_learning_rate = 0.0001
        
        self.discount_factor = 0.9
        self.tau = 0.001

        self.replay_memory_size = 1000000
        self.batch_size = 32
        self.itr = 0
        
        self.actor = Actor(self.embedding_dim, self.actor_hidden_dim, self.actor_learning_rate, state_size, self.tau)
        self.actor2 = Actor2(self.embedding_dim, self.actor_hidden_dim, self.actor_learning_rate, state_size, self.tau)
        self.critic = Critic(self.critic_hidden_dim, self.critic_learning_rate, self.embedding_dim, self.tau)
        self.critic2 = Critic2(self.critic_hidden_dim, self.critic_learning_rate, self.embedding_dim, self.tau)

        self.embedding_network = UserMovieEmbedding(users_num, items_num, self.embedding_dim)
        self.embedding_network([np.zeros((1,)),np.zeros((1,))])
        self.embedding_network.load_weights('./save_weights/user_movie_at_once.h5')
        

        self.srm_ave = DRRAveStateRepresentation(self.embedding_dim)
        self.srm_ave([np.zeros((1, 100,)),np.zeros((1,state_size, 100))])

        # PER
        self.buffer = PriorityExperienceReplay(self.replay_memory_size, self.embedding_dim) # Initialize replay buffer memory
        self.epsilon_for_priority = 1e-6

        # ε-greedy exploration hyperparameter
        self.epsilon = 1.
        self.epsilon_decay = (self.epsilon - 0.1)/500000
        self.std = 1.0
        self.std_new = 0.1

        self.is_test = is_test
        
        target_policy_noise = 0.2,
        target_policy_noise_clip = 0.5,
        exploration_noise = 0.1,
        
        self.exploration_noise = GaussianNoise(self.embedding_dim, exploration_noise, exploration_noise)
        self.target_policy_noise = GaussianNoise(self.embedding_dim, target_policy_noise, target_policy_noise)
        self.target_policy_noise_clip = 0.5

        # wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="drr", 
            entity="diominor",
            config={'users_num':users_num,
            'items_num' : items_num,
            'state_size' : state_size,
            'embedding_dim' : self.embedding_dim,
            'actor_hidden_dim' : self.actor_hidden_dim,
            'actor_learning_rate' : self.actor_learning_rate,
            'critic_hidden_dim' : self.critic_hidden_dim,
            'critic_learning_rate' : self.critic_learning_rate,
            'discount_factor' : self.discount_factor,
            'tau' : self.tau,
            'replay_memory_size' : self.replay_memory_size,
            'batch_size' : self.batch_size,
            'std_for_exploration': self.std})
            
    # ========================== Function declaration ======================================================        

    def calculate_td_target(self, rewards, q_values, dones):
        y_t = np.copy(q_values)
        for i in range(q_values.shape[0]):
            y_t[i] = rewards[i] + (1 - dones[i])*(self.discount_factor * q_values[i])
        return y_t

    def recommend_item(self, action, recommended_items, top_k=False, items_ids=None):
        if items_ids == None:
            items_ids = np.array(list(set(i for i in range(self.items_num)) - recommended_items)) 
                        
        items_ebs = self.embedding_network.get_layer('movie_embedding')(items_ids)

        action = tf.transpose(action, perm=(1,0))
        if top_k:
            item_indice = np.argsort(tf.transpose(tf.keras.backend.dot(items_ebs, action), perm=(1,0)))[0][-top_k:]
            return items_ids[item_indice]
        else:
            item_idx = np.argmax(tf.keras.backend.dot(items_ebs, action)) 
            return items_ids[item_idx]                                
        
    def train(self, max_episode_num, top_k=False, load_model=False):
        # Initialize target networks
        self.actor.update_target_network()
        self.actor2.update_target_network()
        self.critic.update_target_network()
        self.critic2.update_target_network()

        episodic_precision_history = []
        t_reward=0

        for episode in range(max_episode_num):
            # episodic reward 
            episode_reward = 0
            correct_count = 0
            steps = 0
            q_loss1 = 0
            q_loss2 = 0
            mean_action = 0
            # Environment 
            user_id, items_ids, done = self.env.reset() # Returns userID, Already recommended items, and Done.

            while not done:
                user_eb = self.embedding_network.get_layer('user_embedding')(np.array(user_id))
                items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(items_ids))
                
                ## State output to SRM(State Representation Module)
                state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)]) 
                
                action1 = self.actor.network(state)
                action2 = self.actor2.network(state)

#                 q11 = self.critic.target_network([action1, state])
#                 q12 = self.critic2.target_network([action1, state])
                
#                 q21 = self.critic.target_network([action2, state])
#                 q22 = self.critic2.target_network([action2, state])
                


#                 a = np.argmax([q11 ,q12 , q21, q22])


#                 if a == 0 or a == 1:
#                     action = action1
#                 elif a == 2 or a == 3:
#                     action = action2

                q11 = self.critic.network([action1, state])
                q12 = self.critic2.network([action2, state])
                
                action = action1 if q11 >= q12 else action2

                ## ε-greedy exploration
                if self.epsilon > np.random.uniform() and not self.is_test:
                    self.epsilon -= self.epsilon_decay                    
                    action += np.random.normal(0,self.std,size=action.shape)  
                    
                recommended_item = self.recommend_item(action, self.env.recommended_items, top_k=top_k)
                
                next_items_ids, reward, done, _ = self.env.step(recommended_item, top_k=top_k)
                if top_k:
                    reward = np.sum(reward)

                # get next_state
                next_items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(next_items_ids))
                next_state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(next_items_eb, axis=0)]) 

                self.buffer.append(state, action, reward, next_state, done)
                
                
                if self.buffer.crt_idx > 1 or self.buffer.is_full:
                    self.itr += 1
                    
                    # Sample a minibatch of batch_size=32
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, weight_batch, index_batch = self.buffer.sample(self.batch_size)
                    
                    
                    # get actions with noise

                    noise = torch.FloatTensor(self.target_policy_noise.sample())
                    clipped_noise = torch.clamp(noise, -self.target_policy_noise_clip, self.target_policy_noise_clip)
                    
                    target_next_action =self.actor.target_network(batch_next_states) + clipped_noise
                    target_next_action2 =self.actor2.target_network(batch_next_states) + clipped_noise


                    q1_a1 = self.critic.target_network([target_next_action, batch_next_states])
                    q2_a1 = self.critic2.target_network([target_next_action, batch_next_states])
                    
                    q1_a2 = self.critic.target_network([target_next_action2, batch_next_states])
                    q2_a2 = self.critic2.target_network([target_next_action2, batch_next_states])   
                    
#                     print(q1_a1, q2_a1)
#                     print(type(q1_a1),type(q2_a1))
                    
                    next_Q1 = tf.math.minimum(q1_a1, q2_a1)
                    next_Q2 = tf.math.minimum(q1_a2, q2_a2)
                    
#                     print(next_Q1, next_Q2)
                    
                    next_Q = tf.math.maximum(next_Q1, next_Q2)
#                     print("next_Q :",next_Q)
                    
                    td_targets = self.calculate_td_target(batch_rewards, next_Q, batch_dones)
                    
                    
#                     target_qs = self.critic.target_network([target_next_action, batch_next_states])
#                     target_qs2 = self.critic2.target_network([target_next_action, batch_next_states])
                
#                     min_qs = tf.raw_ops.Min(input=tf.concat([target_qs, target_qs2], axis=1), axis=1, keep_dims=True) 
                    
#                     td_targets = self.calculate_td_target(batch_rewards, min_qs, batch_dones)
                    
                    for (p, i) in zip(td_targets, index_batch):
                        self.buffer.update_priority(abs(p[0]) + self.epsilon_for_priority, i)

                    
                    
                    q_loss1 += self.critic.train([batch_actions, batch_states], td_targets, weight_batch)
                    q_loss2 += self.critic2.train([batch_actions, batch_states], td_targets, weight_batch)
                   
                    

                    if self.itr % 5 == 0:
                        s_grads1 = self.critic.dq_da([self.actor.network(batch_states), batch_states]) # returns q_grads
                        s_grads2 = self.critic2.dq_da([self.actor2.network(batch_states), batch_states])
                        
                        #s_grads = (s_grads1 + s_grads2)

                        self.actor.train(batch_states, s_grads1)
                        self.actor2.train(batch_states, s_grads2)

                        #Update TARGET networks line 17 of paper
                        self.actor.update_target_network()  # Using the actor network
                        self.actor2.update_target_network()
                        self.critic.update_target_network() # Using the critic network
                        self.critic2.update_target_network()

                items_ids = next_items_ids
                episode_reward += reward
                mean_action += np.sum(action[0])/(len(action[0]))
                steps += 1

                if reward > 0:
                    correct_count += 1
                
                print(f'recommended items : {len(self.env.recommended_items)},  epsilon : {self.epsilon:0.3f}, reward : {reward:+}', end='\r')
             
                # ==================After the end of each sequence of steps===================================                 
                if done:
                    print()
                    precision = int(correct_count/steps * 100)
                    t_reward = t_reward + episode_reward
                    print(f'{episode}/{max_episode_num}, precision : {precision:2}%, total_reward:{t_reward}, q_loss1 : {q_loss1/steps}, q_loss2 : {q_loss2/steps}')
                    episodic_precision_history.append(precision)
                           
            
            if (episode+1)%500 == 0:
                self.save_model(f'./save_weights/actor_{episode+1}_fixed.h5',
                                f'./save_weights/actor2_{episode+1}_fixed.h5',
                                f'./save_weights/critic_{episode+1}_fixed.h5',
                                f'./save_weights/critic2_{episode+1}_fixed.h5')
             # ================================================================================================   
                

    def save_model(self, actor_path,actor2_path, critic_path, critic2_path):
        self.actor.save_weights(actor_path)
        self.actor2.save_weights(actor2_path)
        self.critic.save_weights(critic_path)
        self.critic2.save_weights(critic2_path)
        
    def load_model(self, actor_path,actor2_path, critic_path, critic_path2):
        self.actor.load_weights(actor_path)
        self.actor2.load_weights(actor2_path)
        self.critic.load_weights(critic_path)
        self.critic2.load_weights(critic_path2)