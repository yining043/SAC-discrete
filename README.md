# SAC-discrete


# Paper: 

Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Haarnoja et al, 2018 https://arxiv.org/abs/1801.01290


# Dependencies:
```
tensorflow 1.15.0
gym[atari] 0.15.7
cv2
mpi4py
numpy
matplotlib
```

# Usage
```
# install environment from .yml
conda env create -f env.yml

# source env
conda activate sac

# install some packages from pip
pip install -r req.txt

# install Spinningup from openAI
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .
cd ..
```


# To train a model:

```
#train with GPU
python ./image_observation/sac_discrete_atari/sac.py --env 'Breakout' --use_gpu --gpu 1

#train with CPU
python ./image_observation/sac_discrete_atari/sac.py --env 'Breakout'

```
After training the model will be saved in dir ./saved_models/

# To plot the training curve:
```
python ./plot_progress.py  --env BeamRider
```

# To reload the trained model:
```
python ./load_atari_model.py  --env BeamRider --model_id 4
```

After the command is executed, the program will run the atari game 5 times and calculate the mean of cumulated reward and clipped reward (+1 for positive reward, -1 for negative reward, 0 for no reward).

See the source code for details on how to reload a trained model and get actions for each observation for attack. Note the following function:
```
# Get the optimal action given current state
get_action(obs, True)

# Get the sampled action given current state
get_action(obs, False)

# Get the probability of next action given current state
get_action_probabilistic(test_state)

# Get the log probability of next action given current state
get_action_log_probabilistic(test_state)
```


# Notice:

SAC-discrete works well for Atari game Space Invaders, Qbert, Breakout, BeamRider, but perform terrible for Pong, Freeway; other environments are testing...