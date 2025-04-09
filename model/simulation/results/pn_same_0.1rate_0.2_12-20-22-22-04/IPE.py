# Run probabilistic simulation for IPE model
import pickle
import json
import os
import numpy as np
import random
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd
from time import strftime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pouring.simulator import PourSimulator


# Load task config from the path folder 
def load_task_config(path, sid=0, eid=54):
    configs = []
    names = []
    for file in os.listdir(path)[sid:eid]:
        if file.endswith('.pkl'):
            config = pickle.load(open(os.path.join(path, file), 'rb'))
            configs.append(config)
            names.append(file[:-4])
    return configs, names

# Run the simulator
def simulate_trial(trial, config, name, save_path, position_noise):
    # print(f'Trial {trial}')
    random.seed(trial)
    RESIZE = 1.2
    task_count = config['task_count']
    simulator = PourSimulator(width=800*RESIZE, height=800*RESIZE, name=f'{task_count}', seed=trial)
    init_info = config['init_info']
    # assign init info to simulator class
    for key in init_info:
        setattr(simulator, key, init_info[key])
    assert simulator.rotate_rate < 0
    simulator.rotate_rate = -0.1
    simulator.max_step = 70000
    simulator.step = simulator.motor_activation_step
    
    _, pouring_angle = simulator.simulate(
        save_seq_dir=f'{save_path}/IPE_sequences/{name}/{trial}', 
        save_interval=100,
        fps=60,
        position_noise=position_noise,
        # angle_noise=0.1,
        # cup_angle_noise=0.02,
        # size_noise=0,
        # velocity_noise=0.5,
        # position_noise=position_noise,
        # velocity_noise=velocity_noise,
        fast_mode=True  # stop simulation when pouring out
        )
    return pouring_angle


if __name__ == '__main__':
    for position_noise in np.arange(0.1, 1, 0.1):
        save_path = f'model/simulation/results/pn_same_0.1rate_{position_noise}_{strftime("%m-%d-%H-%M-%S")}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save this py file
        with open(f'{save_path}/IPE.py', 'w') as f:
            f.write(open(__file__).read())
        with open(f'{save_path}/simulator.py', 'w') as f:
            f.write(open('pouring/simulator.py').read())
        for round in [1, 2, 3]:
            print(f'Round {round}')
            # Load init space
            path = f'pouring/stimuli/round1-{round}/initial_configs'
            configs, names = load_task_config(path, sid=0, eid=54)
            times = 30
            gts, noisy_means, noisy_stds = [], [], []
            for task_id, (config, name) in enumerate(zip(configs, names)):
                print(f'Round {round} Task {task_id} {name}')
                noisy_pouring_angles = []
                # Probabilistic simulation
                with multiprocessing.Pool(processes=times) as pool:
                    results = pool.starmap(simulate_trial, [(trial, config, name, save_path, position_noise) for trial in range(times)])

                noisy_pouring_angles = results
                print(f'Noisy pouring angles: {noisy_pouring_angles}')
                print(f'Mean noisy pouring angle: {np.mean(noisy_pouring_angles)}')
                # noisy_means[name] = np.mean(noisy_pouring_angles)
                noisy_means.append(np.mean(noisy_pouring_angles))
                print(f'SD noisy pouring angle: {np.std(noisy_pouring_angles)}')
                # noisy_stds[name] = np.std(noisy_pouring_angles)
                noisy_stds.append(np.std(noisy_pouring_angles))
                print(f'GT pouring angle: {config["pouring_angle"]}')
                # gts[name] = config['pouring_angle']
                gts.append(config['pouring_angle'])

                # save the results in json
                with open(f'{save_path}/IPE_results_round1-{round}.json', 'a') as f:
                    json.dump({'name': name, 'gt': config['pouring_angle'], 
                            'mean_noisy_angles': np.mean(noisy_pouring_angles), 
                                'std_noisy_angles': np.std(noisy_pouring_angles),
                                'noisy_angles': noisy_pouring_angles,
                                }, 
                                f)
                    f.write('\n')
        
