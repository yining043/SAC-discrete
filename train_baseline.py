#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 00:16:51 2020

@author: yiningma
"""

import os
import numpy as np
import cv2
import time
import gym

class myenv:

    def __init__(env, replayBuffer):
        self.env = env
        self.replayBuffer = replayBuffer
        
        
    class ReplayBuffer:
        """
        A simple FIFO experience replay buffer.
        """
    
        def __init__(self, obs_dim, act_dim, size):
            self.obs_dim = obs_dim
            self.obs1_buf = np.zeros([size, *obs_dim], dtype=np.uint8)
            self.obs2_buf = np.zeros([size, *obs_dim], dtype=np.uint8)
            self.acts_buf = np.zeros([size, act_dim], dtype=np.uint8)
            self.rews_buf = np.zeros(size, dtype=np.float32)
            self.done_buf = np.zeros(size, dtype=np.float32)
            self.ptr, self.size, self.max_size = 0, 0, size
    
        def store(self, obs, act, rew, next_obs, done):
            self.obs1_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.acts_buf[self.ptr] = act
            self.rews_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
            self.ptr = (self.ptr+1) % self.max_size
            self.size = min(self.size+1, self.max_size)
    
        def sample_batch(self, batch_size=32):
            idxs = np.random.randint(0, self.size, size=batch_size)
            # return normalised float observations
            return dict(obs1=self.obs1_buf[idxs].astype('float32') / 255.,
                        obs2=self.obs2_buf[idxs].astype('float32') / 255.,
                        acts=self.acts_buf[idxs],
                        rews=self.rews_buf[idxs],
                        done=self.done_buf[idxs])
    
    """
    Store the observations in ring buffer type array of size m
    """
    class StateBuffer:
        def __init__(self,m):
            self.m = m
    
        def init_state(self, init_obs):
            self.current_state = np.stack([init_obs]*self.m, axis=2)
            return self.current_state
    
        def append_state(self, obs):
            new_state = np.concatenate( (self.current_state, obs[...,np.newaxis]), axis=2)
            self.current_state = new_state[:,:,1:]
            return self.current_state
    
    def process_image_observation(self, observation, obs_dim=[84,84,4]):
        if list(observation.shape) != obs_dim:
            # Convert to gray scale
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            # Add channel axis
            observation = observation[..., np.newaxis]
            # Resize to specified dims
            observation = cv2.resize(observation, (obs_dim[0], obs_dim[1]), interpolation=cv2.INTER_AREA)
            # Add channel axis
            observation = observation[..., np.newaxis]
    
            observation = observation.squeeze() # remove channel axis
    
        return observation

    def reset(self):
        o, r, d, ep_ret, ep_len = self.env.reset(), 0, 0, False, 0, 0
    
        # fire to start game and perform no-op for some frames to randomise start
        o, _, _, _ = selfenv.step(1) # Fire action to start game
        for _ in range(np.random.randint(1, 10)):
                o, _, _, _ = env.step(0) # Action 'NOOP'
    
        o = process_image_observation(o)
        old_lives = env.ale.lives()
        state = state_buffer.init_state(init_obs=o)
        return o, r, d, ep_ret, ep_len, old_lives, state


    def steps(self, action):
        old_lives = env.ale.lives()
        o, r, d, _ = env.step(action)
        o = process_image_observation(o)
        state = state_buffer.append_state(o)
        
        if()
        
        return state, r, d, _

def main():
    # parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', '-game', type=str, 
                        help='choices:SpaceInvaders, Breakout, SpaceInvaders, BeamRider, Qbert, Enduro, Assault, Jamesbond, Berzerk'
                        , required=True)
    parser.add_argument('--total_timesteps','-n_itr', default=int(2500000), type=int)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--gpu','-gpu', type=str, default='2')
    parser.add_argument('--log_interval', '-log', default=10000, type=int)
    parser.add_argument('--log_dir','-log_dir',type=str, default='./logger/')
    parser.add_argument('--save_freq','-save_freq>>', default=int(1e5), type=int)
    parser.add_argument('--seed', '-seed', type=int, default=6)
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)
    
    # set GPU if avialable
    if(params['use_gpu']):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu']
    
    # import Tensorflow
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    print('Tensorflow imported, version',tf.__version__)
    if(params['use_gpu']):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    
    # set random seed
    seed = params['seed']
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
    # build enviriment
#    from stable_baselines.common.cmd_util import make_atari_env
#    from stable_baselines.common.vec_env import VecFrameStack
    env = gym.make('{}-v4'.format(params['game_name']))
    test_state_buffer = StateBuffer(m=4)
    reset(test_env, test_state_buffer)
    ！！！！！！

    
    # build model
    from stable_baselines.deepq.policies import CnnPolicy,MlpPolicy
    if(params['agent_name']=='dqn'):
        from stable_baselines import DQN
        model = DQN(CnnPolicy, env,
                    learning_starts = 50000,
                    buffer_size = int(1e7),
                    target_network_update_freq=10000,
                    verbose=1,
                    exploration_final_eps=0.01,
                    #parameters
                    tensorboard_log = params['log_dir'], double_q=False)
    else: 
        if(params['agent_name']=='soft_dqn'):
            from Soft_DQN import soft_DQN
            model = soft_DQN(CnnPolicy, env,
                    learning_starts = 50000,
                    buffer_size = int(1e7),
                    target_network_update_freq=10000,
                    verbose=1,
                    exploration_final_eps=0.01,
                    #parameters
                    tensorboard_log = params['log_dir'], double_q=False)
        else:
            raise(NotImplementedError)
            
        
    # train model
    from stable_baselines.common.callbacks import CheckpointCallback
    checkpoint_callback = CheckpointCallback(save_freq=params['save_freq'], save_path=params['log_dir'],
                                         name_prefix='checkpoint_'+params['agent_name'])
    model.learn(total_timesteps=params['total_timesteps'], 
                log_interval=params['log_interval'], 
                tb_log_name=params['game_name']+'_'+params['agent_name']+'_'+ str(np.datetime64('now')),
                callback = checkpoint_callback)
    
    # save trained model to pkl file
    if(not os.path.exists('./trained_agents/')):
        os.mkdir('./trained_agents/')
    model.save('./trained_agents/'+params['game_name']+'_'+params['agent_name']+'_'+
               str(np.datetime64('now')),cloudpickle=True)
    env.close()
    
    

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()