import gym
import numpy as np

from Utils.envGym import envGym
from Utils.model_loader import model_loader
from stable_baselines import A2C

#def load_and_test_model(model_dir, model_save_name, render=True, deterministic = False):

def load_and_test_model(model_path, end_id, render=True, deterministic = False):

    # load out model
    model = A2C.load(model_path)

    # create a env from gym and wrap it using our class
    original_env = gym.make(env_id + '-v4')
    env = envGym(original_env, 4)

    # test our model 
    _rew = []
    _ret = []
    _len = []
    n = 5 # run 5 times
        
    max_ep_len = 2500000
    for j in range(n):

        ep_ret, ep_len, done, info = 0, 0, False, {}
        
        obs = env.reset()
        if render: env.render()
        
        while not(done or (ep_len == max_ep_len)):
            action = model.predict(obs, deterministic=deterministic)
            #action_prob = model.get_action_probabilistic(obs)
            obs, rew, done, info = env.step(action)
            ep_ret += int(rew>0)
            if not done:
                ep_len = info.get('stats')['l']
            else:
                ep_len = info.get('episode')['l']
            if render: env.render()

        if render: env.close()
        
        ep_rew = info.get('episode')['r']
        
        print('Ep Return: ', ep_ret, 'Ep Reward: ', ep_rew,  'Ep_len: ', ep_len)
        _rew.append(ep_rew)
        _ret.append(ep_ret)
        _len.append(ep_len)
        
    print('mean Ep Return: ', np.mean(np.array(_ret)))
    print('mean Ep Reward: ', np.mean(np.array(_rew)))
    print('mean Ep Length: ', np.mean(np.array(_len)))
        
if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help='choices:SpaceInvaders, Breakout, BeamRider, Qbert, ...'
                        , required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--seed', type=str, default='3')
    parser.add_argument('--sample', action="store_true") # sample means stochastic
    args = parser.parse_args()
    params = vars(args)

    env_id = params['env']
    model_id = params['model_id']
    seed = params['seed']
    d = not params['sample']
#    print(d)
    model_dir = 'trained_agents/'+env_id+'_A2C_'+model_id +'.pkl'
#    model_save_name = 'tf1_save' + model_id
#    load_and_test_model(model_dir, model_save_name,deterministic = d, render=False)
    load_and_test_model(model_dir,env_id, deterministic = d, render=False)
