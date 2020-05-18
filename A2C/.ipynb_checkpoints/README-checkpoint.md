# A2C version


```
## load the model
model = A2C.load(model_path)

# to get the optimal action given current state
model.predict(obs, deterministic=True)

# to get the sampled action given current state
model.predict(obs, deterministic=False)

# to get the log probability of next action given current state
model.action_probability(obs)

```


        



# Notice:
SAC-discrete works well for some Atari game Space Invaders, Qbert, Breakout, BeamRider, but performs terrible for Pong, Freeway; other environments are testing...

|  Works well  | Not bad| Doesn't work |
|  ----:  | ----: | ----:|
| Assault      | Breakout      |Pong | 
|  BeamRider   | SpaceInvaders   |Seaquest | 
| Enduro| Qbert        |Battlezone |
|        | |Berzerk    |
|              | |Asterix    |

Main results:
|  env  |model_id| seed| Deterministic?| avg_reward(return)| avg_ep_length|
|  ----:  |  ----:| ----:|----:|----: | ----:
| Assault      |final| 6   | True |   399.0 (+19.0)|  735.8
| Assault      |final| 6   | False |   327.6 (+15.6)|  690.4
|  Enduro      |final| 3   | True, False  |   0 (0)| 4427.2
|  BeamRider   |final| 3   | False |   396.0 (+9.0)|  1510.8
|  BeamRider   |final| 3   | True |   -|  -
| SpaceInvaders|final| 6   | False |   260.0 (+15.8) |   1012.4
| SpaceInvaders|final| 6   | True |   284.0 (+17.4) |   1031.8
| Breakout     |final | 6   | False  |   1.4 (+1.4)  |   233.2
| Breakout     |final | 6   | True  |   0.4 (+0.4)  |   181.4
|  Qbert       |final| 6   | False  |   285.0 (+11.4) |   474.2    
|  Qbert       |final| 6   | True  |   290 (+11.6) |   414.6    




# To calculate the avg entropy for attack - A2C
```
python ./get_evg_entropy.py
```
The results:
```
mean Entropy for Assault-v4 0.5689810415866755
mean Entropy for Enduro-v4 0.999998979032379
mean Entropy for BeamRider-v4 0.9901177356752034
mean Entropy for SpaceInvaders-v4 0.6189367346611049
mean Entropy for Breakout-v4 0.9790558607589552
mean Entropy for Qbert-v4 0.8099412249717282
```