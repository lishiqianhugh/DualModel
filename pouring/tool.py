import pandas as pd
import statsmodels.api as sm
import numpy as np
import os
import pickle
import cv2

def calculate_correlation():
    # load gt data from pickle file
    gt_path='pouring/stimuli/stimuli_round1/initial_configs'
    gt_results = []
    ori_names = []
    for file in os.listdir(gt_path):
        if file.endswith('.pkl'):
            ori_names.append(file[:-4])
            task_info = pickle.load(open(os.path.join(gt_path, file), 'rb'))
        gt_results.append(task_info['pouring_angle'])
    gt_results = np.array(gt_results).astype(np.float32)

    qual_names = ['regular_triangle_full_s', 'erlenmeyer_circle_half_s', 'regular_trapezoid_full_s', 
                    'erlenmeyer_trapezoid_full_l', 'ierlenmeyer_trapezoid_half_s', 'erlenmeyer_trapezoid_half_s', 
                    'regular_triangle_half_m', 'erlenmeyer_circle_full_l', 'erlenmeyer_circle_half_l', 
                    'erlenmeyer_circle_full_m', 'erlenmeyer_trapezoid_half_l', 'erlenmeyer_triangle_full_l', 
                    'ierlenmeyer_circle_full_m', 'regular_circle_full_s', 'ierlenmeyer_circle_full_s', 
                    'regular_trapezoid_half_s', 'regular_circle_half_s', 'regular_circle_half_m', 
                    'ierlenmeyer_triangle_half_l', 'erlenmeyer_circle_half_m', 'erlenmeyer_circle_full_s', 
                    'ierlenmeyer_triangle_full_m', 'ierlenmeyer_triangle_half_s', 'regular_trapezoid_full_m', 
                    'ierlenmeyer_trapezoid_full_s', 'erlenmeyer_triangle_half_s', 'erlenmeyer_triangle_full_s', 
                    'erlenmeyer_trapezoid_full_s', 'erlenmeyer_trapezoid_half_m', 'ierlenmeyer_trapezoid_half_l', 
                    'erlenmeyer_triangle_half_l', 'ierlenmeyer_circle_half_l', 'regular_triangle_full_m', 
                    'erlenmeyer_triangle_full_m', 'regular_triangle_full_l', 'regular_circle_full_l', 
                    'erlenmeyer_triangle_half_m', 'regular_circle_half_l', 'ierlenmeyer_trapezoid_full_m', 
                    'ierlenmeyer_triangle_full_l', 'ierlenmeyer_circle_half_s', 'regular_trapezoid_half_m', 
                    'ierlenmeyer_trapezoid_full_l', 'ierlenmeyer_trapezoid_half_m', 'ierlenmeyer_circle_full_l', 
                    'ierlenmeyer_triangle_full_s', 'regular_trapezoid_half_l', 'regular_circle_full_m', 
                    'ierlenmeyer_triangle_half_m', 'regular_triangle_half_l', 'regular_triangle_half_s', 
                    'ierlenmeyer_circle_half_m', 'erlenmeyer_trapezoid_full_m', 'regular_trapezoid_full_l']
    # load human results from csv file and sort based on qual_names
    human_path='../human_results/Pouring Balls - Round 1_October 23, 2023_20.50.csv'
    data = pd.read_csv(human_path, encoding='gbk')
    titles = data.columns.values.tolist()
    human_results = data.iloc[2:, 24:-3].values.tolist()
    human_results = np.array(human_results).astype(np.float32)

    sorted_order = np.argsort(qual_names)
    sorted_names = [qual_names[i] for i in sorted_order]
    human_results = human_results[:, sorted_order]
    gt_results = gt_results[sorted_order]

    config = {'size': {'index': -1, 'label': ['l', 'm', 's']}, 
                'filling_height': {'index': -2, 'label': ['full', 'half']},
                'object_shape': {'index': -3, 'label': ['circle', 'trapezoid', 'triangle']},
                'cup_shape': {'index': -4, 'label': ['erlenmeyer', 'ierlenmeyer', 'regular']}}
    data = {}
    for attr in config.keys():
        data[attr] = [config[attr]['label'].index(name.split('_')[config[attr]['index']]) for name in sorted_names]

    # transform data into dataframe
    data = pd.DataFrame(data)
    data['pouring_angle'] = np.mean(np.abs(human_results - gt_results), axis=0)

    X = data[['size', 'filling_height', 'object_shape', 'cup_shape']]
    y = data['pouring_angle']

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    print(model.summary())

def load_task_config(path, sid=0, eid=54):
    configs = []
    names = []
    for file in os.listdir(path)[sid:eid]:
        if file.endswith('.pkl'):
            config = pickle.load(open(os.path.join(path, file), 'rb'))
            configs.append(config)
            names.append(file)
    return configs, names

def merge_init_configs(npspace_path='pouring/stimuli/initial_configs_nospace', 
                       nogt_path='pouring/stimuli/initial_configs_nogt', 
                       save_path='pouring/stimuli/initial_configs'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in os.listdir(nogt_path):
        if file.endswith('.pkl'):
            task_info_nogt = pickle.load(open(os.path.join(nogt_path, file), 'rb'))
            task_info_npspace = pickle.load(open(os.path.join(npspace_path, file), 'rb'))
            task_info_nogt['pouring_angle'] = task_info_npspace['pouring_angle']
            with open(os.path.join(save_path, file), 'wb') as f:
                pickle.dump(task_info_nogt, f)

def state_similarity(image1, image2):
    similarity = np.mean((image1 - image2) ** 2)
    return similarity

def caculate_similarity_all():
    for round in [1,2,3]:
        sequence_path=f'pouring/stimuli/round1-{round}/sequences'
        config_path=f'pouring/stimuli/round1-{round}/initial_configs'
        for task in os.listdir(sequence_path):
            initial_path = os.path.join(sequence_path, task, '0.png')
            initial_image = cv2.imread(initial_path)
            pouring_angle = pickle.load(open(os.path.join(config_path, f'{task}.pkl'), 'rb'))['pouring_angle']
            print(round, task, pouring_angle)
            pouring_frame = int(pouring_angle * 10.44) # TODO: change to real frame
            pouring_path = os.path.join(sequence_path, task, f'{pouring_frame}.png')
            pouring_image = cv2.imread(pouring_path)
            similarity = state_similarity(initial_image, pouring_image)
            print(similarity)

if __name__ == '__main__':
    # merge_init_configs() # 2023_random_angle_-20_20_-40_40/
    # configs, names = load_task_config(path='pouring/stimuli/initial_configs', sid=0, eid=54)
    # print(configs[0]['init_info']['space'].bodies[2].position, configs[0]['pouring_angle'])
    # # import pdb; pdb.set_trace()
    # print(len(configs))
    caculate_similarity_all()