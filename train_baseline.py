#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 00:16:51 2020

@author: yiningma
"""

import os
import numpy as np
import gym
from Utils.envGym import envGym

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
# !!!!

    
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
    #main()
    
    # init a gym env
    _env = gym.make('{}-v4'.format('Breakout'))
    
    # init our env with 4 frame stack
    env = envGym(_env,4)
    
    # reset env
    obs = env.reset()
    done = False
    ep_len = 0
    
    
    while not(done or (ep_len == 1000)):
        
        # step an action
        action = int(input())
        print(action)
        new_obs, rew, done, info = env.step(action)
        ep_len += 1
        print(rew)
        
        #render
        env.render()
        