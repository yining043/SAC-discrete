#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:16:29 2020

@author: yiningma
"""
import os
import imageio
import numpy as np
import gym
from Utils.envGym import envGym
from Utils.model_loader import model_loader
    

def record(model,env,seed,max_len,num):
    video_length = max_len
    game = model.config['rl_params']['env_name']
    
    images = []

    obs = env.reset()
    img = env.render(mode='rgb_array')
    images.append(img)
    
    _r = 0
    _ret = 0
    print('run ',num)
    
    
    for i in range(video_length+1):
        print("\r", f'{i}/{video_length}', end = '')
        action = model.get_action(obs, False)
        obs, rew, done, info = env.step(action)
        _r += rew
        _ret += int(rew>0) - int(rew<0)
        img = env.render(mode='rgb_array')
        images.append(img)
        if done:
            break
    env.close()
    
    
    output_dir =  f'./saved_gifs/{game}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    imageio.mimsave(f'{output_dir}/{env_id}_r{num}_r{_r}({_ret}).gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
    print('\n done', f'{output_dir}/{env_id}_r{num}_r{_r}({_ret}).gif saved.')

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help='choices:SpaceInvaders, Breakout, BeamRider, Qbert, ...'
                        , required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--seed', type=str, default='3')
    parser.add_argument('--num', type=int, default='5')
    parser.add_argument('--max_len', type=int, default='6000')
    args = parser.parse_args()
    params = vars(args)

    env_id = params['env']
    model_id = params['model_id']
    seed = params['seed']
    max_len = params['max_len']
    num = params['num']
    
    model_dir = 'saved_models/sac_discrete_atari_'+env_id+'-v4/sac_discrete_atari_'+env_id+'-v4_s'+seed+'/'
    model_save_name = 'tf1_save' + model_id
    
    model = model_loader(model_dir, model_save_name)
    
    original_env = gym.make(model.config['rl_params']['env_name'])
    env = envGym(original_env, 4)
    env.reset()
    for _ in range(num):
        record(model,env,seed,max_len,_)