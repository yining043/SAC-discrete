import gym
import numpy as np
from Utils.envGym import envGym
from Utils.model_loader import model_loader
import time

    
def calculate_entropy(actions):
    entropy_actions = [ -probs * np.log(probs) / np.log(len(actions)) if probs>0 else 0 for probs in actions]
    entropy = np.sum(entropy_actions)
    return entropy
    
def get_avg_entropy(model_dir, model_save_name, render=False, deter=False):

    # load out model
    model = model_loader(model_dir, model_save_name)

    # create a env from gym and wrap it using our class
    original_env = gym.make(model.config['rl_params']['env_name'])
    env = envGym(original_env, 4)

    # test our model 
    n = 1 # run just one times
    
    entropy = []
        
    for _ in range(n):

        ep_len, done = 0, False
        
        obs = env.reset()
        if render: env.render()
        
        while not(done or (ep_len == model.max_ep_len)):
            action = model.get_action(obs, deter)
            
            action_prob = model.get_action_probabilistic(obs)
            etrp = calculate_entropy(action_prob)
            entropy.append(etrp)
            if(etrp == np.nan):
                print(action_prob)
            
            obs, rew, done, info = env.step(action)
            if not done:
                ep_len = info.get('stats')['l']
            else:
                ep_len = info.get('episode')['l']
            
            if render: env.render()
        if render: env.close()
        
        ep_rew = info.get('episode')['r']
    
    model.close()  
    print('mean Entropy for '+model.config['rl_params']['env_name'], np.mean(np.array(entropy)))

if __name__ == '__main__':
    
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, help='choices:SpaceInvaders, Breakout, BeamRider, Qbert, ...'
    #                     , required=True)
    # parser.add_argument('--model_id', type=str, required=True)
    # parser.add_argument('--seed', type=str, default='3')
    # args = parser.parse_args()
    # params = vars(args)
    
    sacmodels = [   ['Qbert', '24', '6', True], 
                    ['Assault',	'20','6', False],
                    ['Enduro',	'12', '3', True	],
                    ['BeamRider', '24',	 '3', False],
                    ['SpaceInvaders', '20',	'6',False],
                    ['Breakout', '6', '6', True]    ]
   
    home_dir = '../soft-actor-critic-master/saved_models/'
    
    for conf in sacmodels:
        model_dir = f'{home_dir}sac_discrete_atari_{conf[0]}-v4/sac_discrete_atari_{conf[0]}-v4_s{conf[2]}/'
        model_save_name = f'tf1_save{conf[1]}'
        get_avg_entropy(model_dir, model_save_name, deter = conf[3])
