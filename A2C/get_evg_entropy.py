import gym
import numpy as np
from Utils.envGym import envGym
from Utils.model_loader import model_loader
import time
from stable_baselines import A2C
    
def calculate_entropy(actions):
    entropy_actions = [ -probs * np.log(probs) / np.log(len(actions)) if probs>0 else 0 for probs in actions]
    entropy = np.sum(entropy_actions)
    return entropy
    
def get_avg_entropy(model_path,env_name, render=False, deter=False):

    # load out model
    model = A2C.load(model_path)

    # create a env from gym and wrap it using our class
    original_env = gym.make(env_name+'-v4')
    env = envGym(original_env, 4)

    # test our model 
    n = 1 # run just one times
    
    max_ep_len = 2500000
    
    entropy = []
        
    for _ in range(n):

        ep_len, done = 0, False
        
        obs = env.reset()
        if render: env.render()
        
        while not(done or (ep_len == max_ep_len)):
            action = model.predict(obs, deterministic=deter)
            
            action_prob = model.action_probability(obs)
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
    
    del model  
    print('mean Entropy for '+env_name, np.mean(np.array(entropy)))

if __name__ == '__main__':
    
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, help='choices:SpaceInvaders, Breakout, BeamRider, Qbert, ...'
    #                     , required=True)
    # parser.add_argument('--model_id', type=str, required=True)
    # parser.add_argument('--seed', type=str, default='3')
    # args = parser.parse_args()
    # params = vars(args)
    
#    sacmodels = [   ['Qbert', '24', '6', True], 
 #                   ['Assault',	'20','6', False],
  #                  ['Enduro',	'12', '3', True	],
   #                 ['BeamRider', '24',	 '3', False],
    #                ['SpaceInvaders', '20',	'6',False],
     #               ['Breakout', '6', '6', True]    ]
   
    sacmodels = [   ['Assault',	'final','6', True],
                    ['Enduro',	'final', '3', True	],
                    ['BeamRider', 'final',	 '3', False],
                    ['SpaceInvaders', 'final',	'6',True],
                    ['Breakout', 'final', '6', False],
                    ['Qbert', 'final', '6', False]  ] 

    model_dir = 'trained_agents/'
    
    for conf in sacmodels:

        model_path = model_dir + conf[0] + '_A2C_final.pkl'
        get_avg_entropy(model_path, env_name=conf[0], deter = conf[3])
