
import os
os.environ['CURL_CA_BUNDLE'] = ''
import argparse
from collections import defaultdict
import copy
import pickle
import random
import pandas as pd
import torch
import time
import torch.nn as nn
import numpy as np


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import NMF  



from model_sim_RSA import FDA_score, SFDA_score
from feature_extractor import get_forward, get_forward_random
# from .archi import get_archi
from archi_graph import get_archi_parc
np.set_printoptions(suppress=True)

import datasets, constants, utils

all_datasets = constants.variables['Target Dataset']
all_architectures = constants.variables['Architecture']
source_datasets = constants.variables['Source Dataset']



def get_children(model: torch.nn.Module):
    target_layers = []
    module_list = [module for module in model.modules()] # this is needed
    for count, value in enumerate(module_list):
        
        if isinstance(value, (nn.Conv2d, nn.Linear)):
        #if isinstance(value, (nn.Conv2d)):
            # print(count, value)
            target_layers.append(value)
    return target_layers, len(target_layers)

def get_class_number(model: torch.nn.Module):
    module_list = [module for module in model.modules()]
    linear = module_list[-1]
    assert isinstance(linear, nn.Linear) == True
    res = linear.out_features
    return res

def dataset_meta(data_name, train=True):
    res = []
    dataset_obj = datasets.DatasetCache(data_name, train)
    

    data_length = dataset_obj.length
    res.append(data_length)

    class_number = len(dataset_obj.class_map.class_to_idx.keys())
    res.append(class_number)

    print(f'dataset {data_name} meta [length: {data_length}, class: {class_number}]')
    return res

def model_meta(architecture, source_dataset, idx=None, num=None):
    # __import__('IPython').embed()
    model = utils.load_source_model(architecture, source_dataset)
    model.eval()
    res = []

    params_size = sum(p.numel() for p in model.parameters())
    res.append(params_size)
    _, layer_size = get_children(model)
    res.append(layer_size)
    # archi_type = get_archi(source=source, archi=archi)  # using the sequence2sequence method
    
    archi_type = get_archi_parc(source=source, archi=archi) # new
    res.extend(archi_type) # new

    # print(f'model {architecture}_{source_dataset}: input size:{input_size}, params_size:{params_size}, architecture family:{archi_type}\
    #         layer_size:{layer_size}, class_number:{class_number}')
    print(res)
    return res


model_meta_res = {}
dataset_meta_res = {}
models_name = [a + '%' + b for a in all_architectures for b in source_datasets]
for idx, m in enumerate(models_name):
    archi, source = m.split('%')
    model_meta_res[m] = model_meta(architecture=archi, source_dataset=source, idx=idx, num=len(models_name))

for data in all_datasets:
    dataset_meta_res[data] = dataset_meta(data_name=data)
# get the dimension of meta-features
dim = model_meta_res[models_name[0]] + dataset_meta_res[all_datasets[0]] # 1 for transfer score
performance_tensor = np.zeros((len(models_name), len(all_datasets), len(dim)))
for i, model_name in enumerate(models_name):
    for j, data_name in enumerate(all_datasets):
        model_feature = model_meta_res[model_name]
        data_feature = dataset_meta_res[data_name]
        tmp = model_feature + data_feature
        performance_tensor[i][j] = tmp
    

print('################### meta features DONE! ############################')

def build_full_performance_matrix(result_file='./oracles/controlled.csv'):

    # build zero matrix

    zero_matrix = np.zeros((32, len(all_datasets)), dtype=np.float64)
    models_name = [a + '%' + b for a in all_architectures for b in source_datasets]
    # 32  'stanford_dogs%googlenet', 'stanford_dogs%alexnet', 'voc2007%resnet50'...
    print(len(models_name))
    columns_index = all_datasets

    rows_index = models_name
    matrix = pd.DataFrame(zero_matrix, columns=columns_index, index=rows_index)

    # fill the matrix using actual transfer score  e.g. resnet50,nabird,cifar10,89.97
    gt_csv = pd.read_csv(result_file)
    for index, row in matrix.iterrows():
        archi, source = index.split('%')
        for column in columns_index:
            # __import__('IPython').embed()
            try:
                x = gt_csv[gt_csv['Target Dataset'] == column]
                x = x[x['Architecture'] == archi]
                x = x[x['Source Dataset'] == source]
                x = x['Oracle']
                matrix.loc[index, column] = x.values[0]
            except IndexError:
                matrix.loc[index, column] = 0
                # tmp = matrix.loc[index]
                # matrix.loc[index, column] = np.max(tmp) + 1

    for index, row in matrix.iterrows():
        for r in columns_index:
            if matrix.loc[index, r] == 0:
                tmp = matrix.loc[index]
                matrix.loc[index, r] = np.max(tmp) + 1
    return matrix

out_file = './results/test_cold.csv'
timing_file = './results/test_cold_timing.pkl'

key = ['Run', 'Architecture', 'Source Dataset', 'Target Dataset']
headers = key + list(['mf_cold'])
out_cache = utils.CSVCache(out_file, headers, key=key, append=False)
def test_mf_result(res_matrix, run):
    for model_name, datas in res_matrix.iterrows():
        archi, source = model_name.split("%")
        for d in res_matrix.columns:
            if source == d:
                continue
            # __import__('IPython').embed() 
            row = [run,
                       archi,
                       source,
                       d,
                       res_matrix.loc[model_name, d]]
            out_cache.write_row(row)
    print('write to file ok!')
    return 'write to file ok!'


def seed_all(seed:int):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)


def test(predict_dataset, args, run=0):
    seed = args.seed
    seed_all(seed)
    weight = args.weight
    pca_dim = args.pca_dim
    k = args.k
    lr = args.alpha
    reg = args.reg
    iteration = args.iteration
    no_completion_rebuilding = args.no_completion_rebuilding
    proxy_model = args.proxy_model

    # 1. build sparse matrix from performance matrix
    print(f'{predict_dataset} testing ...')
    performance_matrix = build_full_performance_matrix()
    sparse_p = performance_matrix.to_numpy()

    # 2. use different ways to complete the matrix
    if no_completion_rebuilding == "SFDA":
        print("SFDA")
        f = SFDA_score()
        sparse_complete = f.cal_FDA(pca_dim=pca_dim)
        #cluster_performance = np.delete(cluster_performance, inverse_id, axis=1)
    elif no_completion_rebuilding == "FDA":
        print('FDA')
        f = FDA_score(pre_path='./cache/probes/fixed_budget_500')
        sparse_complete = f.cal_FDA(pca_dim=pca_dim, run=run)
    else:
        raise NotImplementedError

    trans_matrix = copy.deepcopy(sparse_complete)
    #sparse_complete = copy.deepcopy(sparse_p) 

 
    inverse_id = all_datasets.index(predict_dataset)
    sparse_complete = np.delete(sparse_complete, inverse_id, axis=1)



    # conda install -c conda-forge scikit-learn=0.20
    nmf = NMF(n_components=k, max_iter=iteration, random_state=seed, alpha_W=reg, alpha_H='same')
    U = nmf.fit_transform(sparse_complete)
    V = nmf.components_.T
   

    model_vector = U
    dataset_vector = V
    new_dataset = np.zeros((1, k))

    
    # 4.1 get the forward features of historical datasets on clip models
    sample_size = 100
    all_features = {}
    for data in all_datasets:
        features, _ = get_forward(
            data_name=data, sample_size=sample_size, model_name=proxy_model, run=run)
        # all_features[data] = feature_reduce(features, f=128)
        all_features[data] = features

    # 4.2 train the regression model
    # get the corresponding features and vector:
    regression_train = {data_name: feature for data_name, feature in all_features.items() if data_name != predict_dataset}

    # copy dataset vector to sample size
    test_datasets = np.delete(all_datasets, inverse_id).tolist()
    assert test_datasets == list(regression_train.keys())

    training_data = {}
    training_data['features'] = []
    training_data['values'] = []
    for idx, name in enumerate(test_datasets):
        training_data['features'].extend(regression_train[name])
        training_data['values'].extend([dataset_vector[idx]] * sample_size)

    regression_model = RandomForestRegressor(n_estimators=25, random_state=seed)
    regression_model.fit(training_data['features'], training_data['values'])


    print('Meta process...')
    meta_training = []
    meta_value = []
    for i, model_name in enumerate(models_name):
        for j, data_name in enumerate(all_datasets):
            if data_name == predict_dataset:
                continue
            trans_score = trans_matrix[i][j]
            model_feature = model_meta_res[model_name]
            data_feature = dataset_meta_res[data_name]
            tmp = model_feature + data_feature
            meta_training.append(tmp)
            meta_value.append(trans_score)
    

    meta_regressor = LinearRegression()
    meta_regressor.fit(np.array(meta_training), np.array(meta_value))
    

    ########### online #############
    predict_feats, _ = get_forward(
            data_name=predict_dataset, sample_size=sample_size, model_name=proxy_model, run=run)
    stage1_tic = time.time()
    scores = regression_model.predict(predict_feats)
    score = np.mean(scores, axis=0)
    new_dataset = score

    trans_scores = model_vector @ new_dataset.T
    stage1_toc = time.time()


    meta_scores = []
    stage2_tic = time.time()
    for i, model_name in enumerate(models_name):
        model_feature = model_meta_res[model_name]
        tmp = model_feature + dataset_meta_res[predict_dataset]
        tmp = np.array(tmp).reshape(1, -1)
        score = meta_regressor.predict(tmp).item()
        meta_scores.append(score)

    final =  (1-weight) * np.array(trans_scores) + weight * np.array(meta_scores)
   
    stage2_toc = time.time()

    times['stage1'].append(stage1_toc - stage1_tic)
    times['stage2'].append(stage2_toc - stage2_tic )
    times['total'].append(stage2_toc - stage1_tic )
    co = []
    co.append(predict_dataset)
    X_pd = pd.DataFrame(final, columns=co,
                        index=performance_matrix.index)
    test_mf_result(X_pd, run=run)



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', type=int, default=2023, help='seed')
    parser.add_argument('--weight', type=float,
                        default=0.5, help='weight of meta_score')
    parser.add_argument('--pca_dim', type=int, default=32,
                        help='pca reduction dimension')
    parser.add_argument('--k', type=int, default=2, help='latent factor size')
    parser.add_argument('--alpha', type=float, default=10,
                        help='learning rate in MF sgd')
    parser.add_argument('--reg', type=float, default=0,
                        help='regularization parameter in MF sgd')
    parser.add_argument('--iteration', type=int,
                        default=2000, help='iteration in MF sgd')
    parser.add_argument('--no_completion_rebuilding', type=str,
                        default='FDA', help='do not use completion, using [RSA or SFDA] directly')
    parser.add_argument('--proxy_model', help='large vision model', type=str, default='clip')
    args = parser.parse_args()

    times = defaultdict(list)


    for item in all_datasets:
        for run in range(0,5):
            test(predict_dataset=item, args=args, run=run)

    # for item in all_datasets:
    #     test(predict_dataset=item, args=args, run=0)


    for name, t in times.items():
        print(f'{name:20s}: {(sum(t) / len(t))*1000:.3f} ms average')
    with open(timing_file, 'wb') as f:
        pickle.dump(dict(times), f)


# python meta_features_plus.py --weight 0.5 --pca_dim 32 --k 5 --alpha 0.0001 --reg 0 --iteration 1000 --seed 2023  --no_completion_rebuilding 'FDA' --proxy_model 'clip' &&  python metric_cold.py 