import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")

def plot_error_band(axs, x_data, y_data, min, max, data_name, colour='b'):
    upper_bound = max
    lower_bound = min
    axs.plot(x_data, y_data, color=colour)
    # axs.fill_between(x_data, lower_bound, upper_bound, color=colour, alpha=0.5)
    axs.set(xlabel='Epoch', ylabel=data_name)
    # axs.set_ylim([np.min(min) - 0.5, np.max(max) + 0.5])
    for item in ([axs.title, axs.xaxis.label, axs.yaxis.label] +
              axs.get_xticklabels() + axs.get_yticklabels()):
        item.set_fontsize(20)

def plot_progress(progess_file):
    fig, axs = plt.subplots(1, 2, figsize=(18,6))

    data = pd.read_csv(progess_file, sep="\t")
    data_len = len(data)

    plot_error_band(axs[0], data['Epoch'], data['AverageEpRet'],        data['MinEpRet'],     data['MaxEpRet'],     'Episode Return',          colour='r' )
    plot_error_band(axs[1], data['Epoch'], data['AverageTestEpRet'],    data['MinTestEpRet'], data['MaxTestEpRet'], 'Test Episode Return',     colour='b' )
    # plot_error_band(axs[0], data['Epoch'], data['AverageQ1Vals'],        data['MinEpRet'],     data['MaxEpRet'],     'Episode Return',          colour='r' )
    # plot_error_band(axs[1], data['Epoch'], data['AverageQ2Vals'],    data['MinTestEpRet'], data['MaxTestEpRet'], 'Test Episode Return',     colour='b' )

    plt.show()
    fig.savefig(os.path.join(os.path.dirname(progess_file), 'training_curves.png'), dpi=320, pad_inches=0.01, bbox_inches='tight')
    # fig.savefig(os.path.join(os.path.dirname(progess_file), 'q_vals.png'), dpi=320, pad_inches=0.01, bbox_inches='tight')


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help='choices:SpaceInvaders, Breakout, BeamRider, Qbert, ...'
                        , required=True)
    parser.add_argument('--seed', type=str, default='3')
    args = parser.parse_args()
    params = vars(args)
    
    env_id = params['env']
    seed = params['seed']
    # model_dir = 'saved_models/sac_discrete_kl_atari_BreakoutDeterministic-v4/sac_discrete_kl_atari_BreakoutDeterministic-v4_s1/'
    progess_file = 'saved_models/sac_discrete_atari_'+env_id+'-v4/sac_discrete_atari_'+env_id+'-v4_s'+seed+'/progress.txt'
    plot_progress(progess_file)
