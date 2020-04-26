import sys, os
import numpy as np
import time
import gym
import tensorflow as tf

from spinup.utils.logx import *
from image_observation.sac_discrete_atari.common_utils import *

# for GPU users only
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
tf_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf_config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_json_obj(name):
    with open(name + '.json', 'r') as fp:
        return json.load(fp)

def load_and_test_model(model_dir, model_save_name):

    sess = tf.compat.v1.Session(config=tf_config)

    model = restore_tf_graph(sess=sess, fpath=os.path.join(model_dir, model_save_name))
    config = load_json_obj(os.path.join(model_dir, 'config'))
    test_env = gym.make(config['rl_params']['env_name'])

    # load the placehoders
    x_ph = model['x_ph']
    mu = model['mu']
    pi = model['pi']
    action_probs = model['action_probs']
    log_action_probs = model['log_action_probs']

    obs_dim = config['network_params']['input_dims']
    test_state_buffer = StateBuffer(m=obs_dim[2])
    max_ep_len = config['rl_params']['max_ep_len']
    max_noop = config['rl_params']['max_noop']
    thresh = config['rl_params']['thresh']

    def get_action_probabilistic(state):
        state = state.astype('float32') / 255.
        return sess.run(action_probs, feed_dict={x_ph: [state]})[0]
    
    def get_action_log_probabilistic(state):
        state = state.astype('float32') / 255.
        return sess.run(log_action_probs, feed_dict={x_ph: [state]})[0]

    def get_action(state, deterministic=False):
        state = state.astype('float32') / 255.
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: [state]})[0]

    def reset(env, state_buffer):
        o, ep_rew, r, d, ep_ret, ep_len = env.reset(), 0, 0, False, 0, 0

        # fire to start game and perform no-op for some frames to randomise start
        o, _, _, _ = env.step(1) # Fire action to start game
        for _ in range(np.random.randint(1, max_noop)):
                o, _, _, _ = env.step(0) # Action 'NOOP'

        o = process_image_observation(o, obs_dim, thresh)
        _r = r
        r = process_reward(r)
        old_lives = env.ale.lives()
        state = state_buffer.init_state(init_obs=o)
        return o, _r, r, d, ep_ret, ep_len, old_lives, state

    def test_agent(n=10, render=True):
        global sess, mu, pi, q1, q2
        
        _rew = []
        _ret = []
        
        for j in range(n):
            o, ep_rew, r, d, ep_ret, ep_len, test_old_lives, test_state = reset(test_env, test_state_buffer)
            terminal_life_lost_test = False

            if render: 
                test_env.render()

            while not(d or (ep_len == max_ep_len)):

                # start by firing
                if terminal_life_lost_test:
                    a = 1
                else:
                    # Take  lower variance actions at test(noise_scale=0.05)
                    a = get_action(test_state, False)
                    #action_prob = get_action_probabilistic(test_state)
                    #log_action_prob = get_action_log_probabilistic(test_state)

                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(a)
                o = process_image_observation(o, obs_dim, thresh)
                ep_rew += r
                r = process_reward(r)
                test_state = test_state_buffer.append_state(o)
                ep_ret += r
                ep_len += 1

                if test_env.ale.lives() < test_old_lives:
                    test_old_lives = test_env.ale.lives()
                    terminal_life_lost_test = True
                else:
                    terminal_life_lost_test = False

                if render: test_env.render()

            if render: test_env.close()

            print('Ep Return: ', ep_ret, 'Ep Reward: ', ep_rew)
            _rew.append(ep_rew)
            _ret.append(ep_ret)
        print('mean Ep Return: ', np.mean(np.array(_ret)))
        print('mean Ep Reward: ', np.mean(np.array(_rew)))



    test_agent(n=5, render=True)
    test_env.close()

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help='choices:SpaceInvaders, Breakout, BeamRider, Qbert, ...'
                        , required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--seed', type=str, default='3')
    args = parser.parse_args()
    params = vars(args)

    env_id = params['env']
    model_id = params['model_id']
    seed = params['seed']
    
    model_dir = 'saved_models/sac_discrete_atari_'+env_id+'-v4/sac_discrete_atari_'+env_id+'-v4_s'+seed+'/'
    model_save_name = 'tf1_save' + model_id
    load_and_test_model(model_dir, model_save_name)
