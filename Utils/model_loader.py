#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:55:31 2020

@author: yiningma
"""

import os
import tensorflow as tf

from spinup.utils.logx import *

# for GPU users only
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
tf_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf_config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class model_loader:
    def __init__(self, model_dir, model_save_name,):
        
        # create a session
        self.sess = tf.compat.v1.Session(config=tf_config)
    
        # load the pre-trained model
        self.model = restore_tf_graph(sess=self.sess, fpath=os.path.join(model_dir, model_save_name))
        self.config = self.load_json_obj(os.path.join(model_dir, 'config'))
        self.x_ph = self.model['x_ph']
        self.mu = self.model['mu']
        self.pi = self.model['pi']
        self.action_probs = self.model['action_probs']
        self.log_action_probs = self.model['log_action_probs']
        self.max_ep_len = self.config['rl_params']['max_ep_len']
        self.thresh = self.config['rl_params']['thresh']
    
    def load_json_obj(self, name):
        with open(name + '.json', 'r') as fp:
            return json.load(fp)
    
    
    # define the useful functions 

    def get_action_probabilistic(self, state):
        state = state.astype('float32') / 255.
        return self.sess.run(self.action_probs, feed_dict={self.x_ph: [state]})[0]
    
    def get_action_log_probabilistic(self, state):
        state = state.astype('float32') / 255.
        return self.sess.run(self.log_action_probs, feed_dict={self.x_ph: [state]})[0]

    def get_action(self, state, deterministic=False):
        state = state.astype('float32') / 255.
        act_op = self.mu if deterministic else self.pi
        return self.sess.run(act_op, feed_dict={self.x_ph: [state]})[0]
