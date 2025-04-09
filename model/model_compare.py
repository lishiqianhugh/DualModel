import numpy as np
import random
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import statsmodels.api as sm
import json
import os
import scipy.stats as stats
import seaborn as sns

def load_data(path):
    qual_names_1 = [(1, 'regular_triangle_full_s'), (2, 'erlenmeyer_circle_half_s'), (3, 'regular_trapezoid_full_s'), (4, 'erlenmeyer_trapezoid_full_l'), (5, 'ierlenmeyer_trapezoid_half_s'), (6, 'erlenmeyer_trapezoid_half_s'), (7, 'regular_triangle_half_m'), (8, 'erlenmeyer_circle_full_l'), (9, 'erlenmeyer_circle_half_l'), (10, 'erlenmeyer_circle_full_m'), (11, 'erlenmeyer_trapezoid_half_l'), (12, 'erlenmeyer_triangle_full_l'), (13, 'ierlenmeyer_circle_full_m'), (14, 'regular_circle_full_s'), (15, 'ierlenmeyer_circle_full_s'), (16, 'regular_trapezoid_half_s'), (17, 'regular_circle_half_s'), (18, 'regular_circle_half_m'), (19, 'ierlenmeyer_triangle_half_l'), (20, 'erlenmeyer_circle_half_m'), (21, 'erlenmeyer_circle_full_s'), (22, 'ierlenmeyer_triangle_full_m'), (23, 'ierlenmeyer_triangle_half_s'), (24, 'regular_trapezoid_full_m'), (25, 'ierlenmeyer_trapezoid_full_s'), (26, 'erlenmeyer_triangle_half_s'), (27, 'erlenmeyer_triangle_full_s'), (28, 'erlenmeyer_trapezoid_full_s'), (29, 'erlenmeyer_trapezoid_half_m'), (30, 'ierlenmeyer_trapezoid_half_l'), (31, 'erlenmeyer_triangle_half_l'), (32, 'ierlenmeyer_circle_half_l'), (33, 'regular_triangle_full_m'), (34, 'erlenmeyer_triangle_full_m'), (35, 'regular_triangle_full_l'), (36, 'regular_circle_full_l'), (37, 'erlenmeyer_triangle_half_m'), (38, 'regular_circle_half_l'), (39, 'ierlenmeyer_trapezoid_full_m'), (40, 'ierlenmeyer_triangle_full_l'), (41, 'ierlenmeyer_circle_half_s'), (42, 'regular_trapezoid_half_m'), (43, 'ierlenmeyer_trapezoid_full_l'), (44, 'ierlenmeyer_trapezoid_half_m'), (45, 'ierlenmeyer_circle_full_l'), (46, 'ierlenmeyer_triangle_full_s'), (47, 'regular_trapezoid_half_l'), (48, 'regular_circle_full_m'), (49, 'ierlenmeyer_triangle_half_m'), (50, 'regular_triangle_half_l'), (51, 'regular_triangle_half_s'), (52, 'ierlenmeyer_circle_half_m'), (53, 'erlenmeyer_trapezoid_full_m'), (54, 'regular_trapezoid_full_l')]
    qual_names_1 = [t[1] for t in qual_names_1]
    all_human_results = []
    all_gt_results = []
    all_errors = []
    ori_names = np.sort(qual_names_1)
    df = pd.read_csv(path)

    def filter_outlier(df):
        threshold = 3000 
        outlier_names = []
        all_gt_results = []
        all_human_results = []
        ori_names = np.sort(qual_names_1)
        for name in ori_names:
            size = name.split('_')[3][:1]
            filling_height = name.split('_')[2]
            object_shape = name.split('_')[1]
            cup_shape = name.split('_')[0]
            gt_results = df[(df['size'] == size) & (df['filling_height'] == filling_height) & (df['object_shape'] == object_shape) & (df['cup_shape'] == cup_shape)]['gt_pouring_angle'].values
            human_results = df[(df['size'] == size) & (df['filling_height'] == filling_height) & (df['object_shape'] == object_shape) & (df['cup_shape'] == cup_shape)]['human_pouring_angle'].values
            all_gt_results.append(gt_results)
            all_human_results.append(human_results)
        mean_gt_results = np.mean(all_gt_results, axis=1)
        all_human_results = np.array(all_human_results)
        for i in range(all_human_results.shape[1]):
            loss = np.mean((all_human_results[:,i] - mean_gt_results)**2)
            name = df['participant_name'].values[i*54]
            if loss >= threshold:
                outlier_names.append(name)
                # print('Filter out', i, name, loss)

        mean_human_results = np.mean(all_human_results, axis=1)
        for out_name in outlier_names:
            df = df[df['participant_name'] != out_name] 
        return df

    df = filter_outlier(df)
    df['error'] = df['human_pouring_angle'] - df['gt_pouring_angle']
    for name in ori_names:
        size = name.split('_')[3][:1]
        filling_height = name.split('_')[2]
        object_shape = name.split('_')[1]
        cup_shape = name.split('_')[0]
        gt_results = df[(df['size'] == size) & (df['filling_height'] == filling_height) & (df['object_shape'] == object_shape) & (df['cup_shape'] == cup_shape)]['gt_pouring_angle'].values
        human_results = df[(df['size'] == size) & (df['filling_height'] == filling_height) & (df['object_shape'] == object_shape) & (df['cup_shape'] == cup_shape)]['human_pouring_angle'].values
        errors = df[(df['size'] == size) & (df['filling_height'] == filling_height) & (df['object_shape'] == object_shape) & (df['cup_shape'] == cup_shape)]['error'].values
        all_gt_results.append(np.mean(gt_results))
        all_human_results.append(np.mean(human_results))
        all_errors.append(np.mean(errors))
    return ori_names, np.array(all_gt_results), np.array(all_human_results), np.array(all_errors)

def load_IPE_results(IPE_dir):
    three_IPE_results = []
    three_IPE_stds = []
    for round in [1,2,3]:
        path = f'{IPE_dir}/IPE_results_round1-{round}.json'
        # load IPE results from json file, the dicts are all in one line, sepearate them based on {}
        IPE_results = []
        IPE_stds = []
        with open(path, 'r') as f:
            content = f.readlines()
            if len(content) == 1:
                # all dicts are in one line
                content = content[0].split('}')[:-1]
                content = [c + '}' for c in content]
            # import pdb; pdb.set_trace()
            for res_dict in content:
                # import pdb; pdb.set_trace()
                noisy_angles = json.loads(res_dict)['noisy_angles']
                IPE_results.append(np.mean(noisy_angles))
                IPE_stds.append(np.std(noisy_angles))
        three_IPE_results.append(IPE_results)
        three_IPE_stds.append(IPE_stds)
    three_IPE_results = np.array(three_IPE_results)
    IPE_results = np.mean(three_IPE_results, axis=0)
    IPE_stds = np.mean(three_IPE_stds, axis=0)

    return IPE_results, IPE_stds

def encode_attrs(names, gt_results):
    sizes = [name.split('_')[3][:1] for name in names]
    filling_heights = [name.split('_')[2] for name in names]
    object_shapes = [name.split('_')[1] for name in names]
    cup_shapes = [name.split('_')[0] for name in names]
    
    init_configs = []
    for name in names:
        init_configs.append(pickle.load(open(f'./stimuli/round1-1/initial_configs/{name}.pkl', 'rb')))
    
    filling_volumes = []
    for i, init_config in enumerate(init_configs):
        filling_volume = 0
        for obj in init_config['init_info']['objects']:
            filling_volume += obj.area
        filling_volumes.append(filling_volume)
    sizes_encoding = {'s': 0, 'm': 1, 'l': 2}
    filling_heights_encoding = {'half': 1, 'full': 2}
    object_shapes_encoding = {'circle': 0, 'triangle': 3, 'trapezoid': 4} # num of corners
    cup_shapes_encoding = {'erlenmeyer': -1, 'ierlenmeyer': 1, 'regular': 0}
    sizes = [sizes_encoding[size] for size in sizes]
    filling_heights = [filling_heights_encoding[filling_height] for filling_height in filling_heights]
    object_shapes = [object_shapes_encoding[object_shape] for object_shape in object_shapes]
    cup_shapes = [cup_shapes_encoding[cup_shape] for cup_shape in cup_shapes]
    filling_volumes = [filling_volume / 1000 for filling_volume in filling_volumes]
    return [
            sizes, 
            filling_heights, 
            object_shapes, 
            cup_shapes, 
            filling_volumes,
            gt_results,
            ]

def learn_heuristic_model(attrs, gt_results):
    heuristic_model = sm.OLS(gt_results-90, attrs).fit()
    # print(heuristic_model.summary())
    return heuristic_model.params

def load_Heuristic_results(attrs, coeffs, fix_coeffs=False, noise=0):
    if fix_coeffs:
        coeffs = [7.02898528, -19.95479845, 1.57705401, -11.5281999] # filter outlier > 3000
    if noise:
        noisy_Heuristiic_results = []
        for seed in range(10):
            random.seed(seed)
            tmp_attrs = attrs + np.random.normal(0, noise, attrs.shape)
            Heuristic_results = 0
            for i in range(len(coeffs)):
                Heuristic_results += coeffs[i] * tmp_attrs[:, i]
            Heuristic_results += 90
            noisy_Heuristiic_results.append(Heuristic_results)
        all_Heuristic_results = np.mean(noisy_Heuristiic_results, axis=0)

    else:
        all_Heuristic_results = 0
        for i in range(len(coeffs)):
            all_Heuristic_results += coeffs[i] * attrs[:, i]
        all_Heuristic_results += 90
    
    return all_Heuristic_results

def grid_search_switch_point(names, all_gt_results, all_human_results, all_IPE_results, all_Heuristic_results):
    rmses = {}
    errors = all_human_results - all_gt_results
    switch_metric = all_IPE_results.copy()
    for switch_point in np.arange(40, 100, 0.1):
        hybrid_results = all_IPE_results * (switch_metric < switch_point) + all_Heuristic_results * (switch_metric >= switch_point)
        rmses[switch_point] = np.sqrt(np.mean((hybrid_results - all_human_results)**2))
    rmses = sorted(rmses.items(), key=lambda x: x[1])
    min_rmse = rmses[0][1]
    best_switch_point = rmses[0][0]

    return best_switch_point, min_rmse

def model_selection(names, all_model_name, setting, all_attrs, all_gt_results, all_human_results, all_IPE_results, all_Heuristic_results, switch_point):
    def plot(all_results, all_model_name, all_R_square=None, all_AIC=None, all_BIC=None, all_paras=None):
        fontsize = 10
        plt.figure(figsize=(10, 10))
        for i in range(len(all_model_name)):
            results = all_results[i]
            model_name = all_model_name[i]
            R_square = all_R_square[i]
            AIC = all_AIC[i]
            BIC = all_BIC[i]
            plt.subplot(2, 2, i+1)
            plt.scatter(all_gt_results, results, color='blue', label='human', marker='o', s=20)
            plt.plot([20, 120], [20, 120], '--', color='gray')
            plt.xlabel(f'{model_name} prediction', fontsize=fontsize)
            plt.ylabel('Human prediction', fontsize=fontsize)
            plt.xlim([20, 120])
            plt.ylim([20, 120])
            plt.title(f'Human and {model_name} results \n {setting}', fontsize=fontsize)
            plt.text(90, 40, f'R: {np.sqrt(R_square):.3f}', color='red', fontsize=fontsize )
            plt.text(90, 35, f'R^2: {R_square:.3f}', color='red', fontsize=fontsize )
            plt.text(90, 30, f'AIC: {AIC:.3f}', color='red', fontsize=fontsize)
            plt.text(90, 25, f'BIC: {BIC:.3f}', color='red', fontsize=fontsize)
        plt.show()
    
    # Determinsitic model
    Deter_model = sm.OLS(all_human_results, sm.add_constant(all_gt_results)).fit()
    Deter_regression_results = Deter_model.predict(sm.add_constant(all_gt_results))

    # IPE model
    IPE_model = sm.OLS(all_human_results, sm.add_constant(all_IPE_results)).fit()
    IPE_regression_results = IPE_model.predict(sm.add_constant(all_IPE_results))

    # Heuristic model
    Heuristic_model = sm.OLS(all_human_results, sm.add_constant(all_Heuristic_results)).fit()
    Heuristic_regression_results = Heuristic_model.predict(sm.add_constant(all_Heuristic_results))

    # IPE + Heuristic model
    switch_metric = all_IPE_results.copy()
    hybrid_results = all_IPE_results * (switch_metric < switch_point) + all_Heuristic_results * (switch_metric >= switch_point)
    hybrid_model = sm.OLS(all_human_results, sm.add_constant(hybrid_results)).fit()
    hybrid_regression_results = hybrid_model.predict(sm.add_constant(hybrid_results))
    
    # Results
    all_model_results = [Deter_regression_results, IPE_regression_results, Heuristic_regression_results, hybrid_regression_results]
    all_R_square = [Deter_model.rsquared, IPE_model.rsquared, Heuristic_model.rsquared, hybrid_model.rsquared]
    all_AIC = [Deter_model.aic, IPE_model.aic, Heuristic_model.aic, hybrid_model.aic]
    all_BIC = [Deter_model.bic, IPE_model.bic, Heuristic_model.bic, hybrid_model.bic]
    # plot(all_model_results, all_model_name, all_R_square, all_AIC, all_BIC)
    all_r = [np.sqrt(r) for r in all_R_square]

    return all_r, all_model_results

def calculate_underestimation(all_gt_results, all_human_results, all_model_results, all_IPE_results, all_Heuristic_results, switch_point):
    human_oversetimation = (all_human_results - all_gt_results) < 0
    Deter_overestimation = (all_gt_results - all_gt_results) < 0
    IPE_overestimation = (all_IPE_results - all_gt_results) < 0
    Heuristic_overestimation = (all_Heuristic_results - all_gt_results) < 0
    hybrid_results = all_IPE_results * (all_IPE_results < switch_point) + all_Heuristic_results * (all_IPE_results >= switch_point)
    hybrid_overestimation = (hybrid_results - all_gt_results) < 0

    Deter_acc = np.sum(Deter_overestimation == human_oversetimation) / len(human_oversetimation)
    IPE_acc = np.sum(IPE_overestimation == human_oversetimation) / len(human_oversetimation)
    Heuristic_acc = np.sum(Heuristic_overestimation == human_oversetimation) / len(human_oversetimation)
    hybrid_acc = np.sum(hybrid_overestimation == human_oversetimation) / len(human_oversetimation)
    all_acc = [Deter_acc, IPE_acc, Heuristic_acc, hybrid_acc]

    return all_acc

def run_model_compare():
    all_model_name = ['Determinisitic Physics', 'IPE', 'Heuristic', 'IPE + Heuristic']
    IPE_path = 'model/simulation/results/pn_same_0.1rate_0.2_12-20-22-22-04'
    noise_num = 20
    all_cup_shape = ['regular', 'erlenmeyer', 'ierlenmeyer']
    all_object_shape = ['circle', 'triangle', 'trapezoid']
    all_object_size = ['s', 'm', 'l']
    all_filling_height = ['half', 'full']
    all = ['all']
    all_attr_names = all
    for i, attr in enumerate(all_attr_names):
        IPE_dir = IPE_path
        all_names, all_gt_results, all_human_results, all_errors = load_data(path='data.csv')
        all_IPE_results, all_IPE_stds = load_IPE_results(IPE_dir)
        all_attrs = encode_attrs(all_names, all_gt_results)
        if attr == 'all' or attr == 'shift_all':
            mask = np.ones(len(all_names), dtype=bool)
        else:
            mask = np.array([attr in name.split('_') for name in all_names])
        names = np.array(all_names)[mask]
        gt_results = np.array(all_gt_results)[mask]
        human_results = np.array(all_human_results)[mask]
        IPE_results = np.array(all_IPE_results)[mask]
        attrs = np.array(all_attrs).T[mask]
        attrs = attrs[:, :4]
        coeffs = learn_heuristic_model(np.array(all_attrs).T[:,:4], np.array(all_gt_results))
        Heuristic_results = load_Heuristic_results(attrs, coeffs, fix_coeffs=False, noise=0)
        best_switch_point, min_rmse = grid_search_switch_point(names, gt_results, human_results, IPE_results, Heuristic_results)
        all_r, all_model_results = model_selection(names, all_model_name, attr, attrs, gt_results, human_results, IPE_results, Heuristic_results, best_switch_point)
        all_acc = calculate_underestimation(gt_results, human_results, all_model_results, IPE_results, Heuristic_results, best_switch_point)
        print(f'Attr: {attr}, min_rmse: {min_rmse}, best switch point: {best_switch_point}')
        print(all_r)

if __name__ == '__main__':
    qual_names_1 = [(1, 'regular_triangle_full_s'), (2, 'erlenmeyer_circle_half_s'), (3, 'regular_trapezoid_full_s'), (4, 'erlenmeyer_trapezoid_full_l'), (5, 'ierlenmeyer_trapezoid_half_s'), (6, 'erlenmeyer_trapezoid_half_s'), (7, 'regular_triangle_half_m'), (8, 'erlenmeyer_circle_full_l'), (9, 'erlenmeyer_circle_half_l'), (10, 'erlenmeyer_circle_full_m'), (11, 'erlenmeyer_trapezoid_half_l'), (12, 'erlenmeyer_triangle_full_l'), (13, 'ierlenmeyer_circle_full_m'), (14, 'regular_circle_full_s'), (15, 'ierlenmeyer_circle_full_s'), (16, 'regular_trapezoid_half_s'), (17, 'regular_circle_half_s'), (18, 'regular_circle_half_m'), (19, 'ierlenmeyer_triangle_half_l'), (20, 'erlenmeyer_circle_half_m'), (21, 'erlenmeyer_circle_full_s'), (22, 'ierlenmeyer_triangle_full_m'), (23, 'ierlenmeyer_triangle_half_s'), (24, 'regular_trapezoid_full_m'), (25, 'ierlenmeyer_trapezoid_full_s'), (26, 'erlenmeyer_triangle_half_s'), (27, 'erlenmeyer_triangle_full_s'), (28, 'erlenmeyer_trapezoid_full_s'), (29, 'erlenmeyer_trapezoid_half_m'), (30, 'ierlenmeyer_trapezoid_half_l'), (31, 'erlenmeyer_triangle_half_l'), (32, 'ierlenmeyer_circle_half_l'), (33, 'regular_triangle_full_m'), (34, 'erlenmeyer_triangle_full_m'), (35, 'regular_triangle_full_l'), (36, 'regular_circle_full_l'), (37, 'erlenmeyer_triangle_half_m'), (38, 'regular_circle_half_l'), (39, 'ierlenmeyer_trapezoid_full_m'), (40, 'ierlenmeyer_triangle_full_l'), (41, 'ierlenmeyer_circle_half_s'), (42, 'regular_trapezoid_half_m'), (43, 'ierlenmeyer_trapezoid_full_l'), (44, 'ierlenmeyer_trapezoid_half_m'), (45, 'ierlenmeyer_circle_full_l'), (46, 'ierlenmeyer_triangle_full_s'), (47, 'regular_trapezoid_half_l'), (48, 'regular_circle_full_m'), (49, 'ierlenmeyer_triangle_half_m'), (50, 'regular_triangle_half_l'), (51, 'regular_triangle_half_s'), (52, 'ierlenmeyer_circle_half_m'), (53, 'erlenmeyer_trapezoid_full_m'), (54, 'regular_trapezoid_full_l')]
    qual_names_1 = [t[1] for t in qual_names_1]
    run_model_compare()
