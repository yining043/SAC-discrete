#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:23:20 2020

@author: yiningma
"""

import numpy as np
import cv2
import gym


class envGym:

    def __init__(self,env, stack):
        gym.logger.set_level(40)
        self.env = env
        self.state_buffer = self.StateBuffer(m=stack)
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.observation_space = gym.spaces.Box(
            low = np.min(self.env.observation_space.low)*1.0, 
            high = np.max(self.env.observation_space.high)*1.0, 
            shape = (84,84,4), dtype=np.float32)
        self.reward_range = self.env.reward_range
        
        self.ep_len = 0
        self.rp_rew = 0
        
        
        
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
        o = self.env.reset()
        self.ep_len = 0
        self.rp_rew = 0
    
        # fire to start game and perform no-op for some frames to randomise start
        o, _, _, _ = self.env.step(1) # Fire action to start game
        for _ in range(np.random.randint(1, 10)):
            o, _, _, _ = self.env.step(0) # Action 'NOOP'
    
        o = self.process_image_observation(o)
        state = self.state_buffer.init_state(init_obs=o)
        return state


    def step(self, action):
        old_lives = self.env.ale.lives()
        o, r, done, info = self.env.step(action)
        o = self.process_image_observation(o)
        state = self.state_buffer.append_state(o)
        self.ep_len += 1
        self.rp_rew += r
        if self.env.ale.lives() < old_lives and self.env.ale.lives() > 0:
            o, r, done, info = self.env.step(1)
            self.ep_len += 1
            self.rp_rew += r
            o = self.process_image_observation(o)
            state = self.state_buffer.append_state(o)
        
        if done:
            if 'episode' not in info.keys():
                info['episode'] = {'r':self.rp_rew, 'l': self.ep_len}
            else:
                info['_episode'] = {'r':self.rp_rew, 'l': self.ep_len}
        else:
            if 'stats' not in info.keys():
                info['stats'] = {'r':self.rp_rew, 'l': self.ep_len}
            else:
                info['stats'] = {'r':self.rp_rew, 'l': self.ep_len}
        return state, r, done, info
    
    def render(self, mode='human', **kwargs):
        self.env.render(mode, **kwargs)
        
    def seed(self, seed=None):
        self.env.seed(seed)
        
    
        
    def close(self):
        self.env.close()
        