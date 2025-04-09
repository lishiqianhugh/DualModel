import pygame
import pymunk
import pymunk.pygame_util
import random
import yaml
import math
import pickle
import json
import os
from simulator import *


with open('config_diverse.yml', 'r') as f:
    config_diverse = yaml.safe_load(f)

RESIZE = 1.2
cup_center = [280, 190]
cup_size = [240, 420]
num_names = ['full', 'half']
size_names = ['l', 'm', 's']
cup_shapes = ['regular', 'erlenmeyer', 'ierlenmeyer']
object_shapes = ['circle', 'triangle', 'trapezoid']
add_object_method = 'random' # ['random'，'fixed']

if __name__ == '__main__':
    task_count = 0
    loading = os.path.exists(f'stimuli/initial_configs')
    for cup_shape in cup_shapes:
        for object_shape in object_shapes:
            object_sizes = config_diverse[cup_shape][object_shape]['object_sizes'] * 2
            object_nums = config_diverse[cup_shape][object_shape]['object_nums']
            filling_heights = ['full'] * 3 + ['half'] * 3
            size_names = ['l', 'm', 's'] * 2
            for idx, (object_size, object_num, filling_height, size_name) in enumerate(zip(object_sizes, object_nums, filling_heights, size_names)):
                random.seed(2023)
                task_count += 1
                object_size *= RESIZE
                simulator = PourSimulator(width=800*RESIZE, height=800*RESIZE, name=f'{cup_shape}_{object_shape}_{num_names[idx // 3]}_{size_name}')
                print(f'[{task_count}] Simulating with: cup_shape={cup_shape}, object_shape={object_shape}, object_size={object_size}, object_num={object_num}, filling_height={filling_height}')
                simulator.add_cup(x=cup_center[0]*RESIZE, y=cup_center[1]*RESIZE, cup_width=cup_size[0]*RESIZE, cup_height=cup_size[1]*RESIZE, cup_shape=cup_shape)
                # load object info from json and simulate
                if loading: 
                    print(f'Loading info from pkl: task {task_count}')
                    with open(f'stimuli/initial_configs/{cup_shape}_{object_shape}_{num_names[idx // 3]}_{size_names[idx % 3]}.pkl', 'rb') as f:
                        config = pickle.load(f)
                        init_info = config['init_info']
                        for key in init_info:
                            # assign init config to simulator class
                            setattr(simulator, key, init_info[key])
                        simulator.step = init_info['motor_activation_step']
                        simulator.max_step = 70000
                        _, pouring_angle = simulator.simulate(
                                # save_init_dir=f'stimuli/initial_scenes/',
                                # save_seq_dir=f'stimuli/sequences/{cup_shape}_{object_shape}_{num_names[idx // 3]}_{size_names[idx % 3]}',
                                # save_interval=50
                                )
                # generate object info and simulate
                else:
                    print(add_object_method)
                    if add_object_method == 'random':
                        for i in range(object_num):
                            x = cup_center[0] + 0.5 * cup_size[0] + random.randint(-40, 40)
                            y = cup_center[1] + 0.5 * cup_size[1] + random.randint(-80, 80)
                            simulator.add_object(x, y, size=object_size, mass=1, object_shape=object_shape)
                    elif add_object_method == 'fixed':
                        shift = {'circle_l_full': [[0, 0], [1, 1], [0, 0],  [0, 0]],
                                'triangle_l_full': [[0, 0], [0, 0], [1, 1],  [1, 1]],
                                'triangle_l_half': [[-20, 0], [20, -150], [1, 1],  [1, 1]],
                                'trapezoid_l_full': [[0, 0], [0, 0], [100, -200],  [0, -20]],
                                'trapezoid_l_half': [[0, 0], [0, 100], [1, 1],  [1, 1]],}
                        for i in range(object_num):
                            x = cup_center[0] + 0.5 * cup_size[0] + shift[f'{object_shape}_{size_name}_{filling_height}'][i][0]
                            y = cup_center[1] + 0.5 * cup_size[1] + shift[f'{object_shape}_{size_name}_{filling_height}'][i][1]
                            simulator.add_object(x, y, size=object_size, mass=1, object_shape=object_shape)
                    init_info, pouring_angle = simulator.simulate(
                                save_init_dir=f'stimuli/initial_scenes/',
                                save_seq_dir=f'stimuli/sequences/{cup_shape}_{object_shape}_{num_names[idx // 3]}_{size_names[idx % 3]}',
                                save_interval=50
                                )
                    print(init_info['space'].bodies[2].position)
                    print(simulator.space.bodies[2].position)
                    # save init space and pouring angle together in pickle
                    mode = ''
                    if not os.path.exists(f'stimuli/initial_configs{mode}'):
                        os.makedirs(f'stimuli/initial_configs{mode}')
                    with open(f'stimuli/initial_configs{mode}/{cup_shape}_{object_shape}_{num_names[idx // 3]}_{size_names[idx % 3]}.pkl', 'wb') as f:
                        pickle.dump({'task_count': task_count, 'init_info': init_info, 'pouring_angle': pouring_angle}, f)

    print('Total number of tasks', task_count)
