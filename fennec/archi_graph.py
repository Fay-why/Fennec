
import os
from graphviz import Digraph
import torch
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
from pytorchcv.model_provider import get_model as ptcv_get_model
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from karateclub.graph_embedding import Graph2Vec
import sys
import utils, constants

def feature_reduce(features:np.ndarray, f:int=None) -> np.ndarray:
    """
    Use PCA to reduce the dimensionality of the features.

    If f is none, return the original features.
    If f < features.shape[0], default f to be the shape.
    """
    if f is None:
        return features

    if f > features.shape[0]:
        f = features.shape[0]

    return PCA(
        n_components=f,
        svd_solver='randomized',
        random_state=1919,
        iterated_power=1).fit_transform(features)

def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph
    
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    # __import__('IPython').embed()
    param_map = {id(v): k for k, v in params.items()}
    #print(param_map) # 139741267009112: 'layer4.2.bn3.bias',
    
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    res1 = []
    res2 = []
    res3 = []
    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                res1.append(size_to_str(var.size()))
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
                res2.append(node_name)
            else:
                dot.node(str(id(var)), str(type(var).__name__))
                #print( str(type(var).__name__))
                res3.append(str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    length = len(res1) + len(res2) + len(res3)
    print(length)
    return dot, length


def demo_from_homespace(model_name, status=True):
    net = ptcv_get_model(model_name, pretrained=status)
    print(f"{model_name}: download ok")
    return net


def load_model_name(type='imgclob', path='./oracles/model_100.txt'):
    model_names = []
    if type == 'imgclob':
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(':')
                if len(line) != 2:
                    continue
                model_names.append(eval(line[0]))
    elif type == 'parc':
        all_datasets = [
		'cifar10',
		'oxford_pets',
		'cub200',
		'caltech101',
		'stanford_dogs',
		'nabird',]
        all_architectures =[
		'resnet50',
		'resnet18',
		'googlenet',
		'alexnet',
	]
        source_datasets = [
		'nabird',
		'oxford_pets',
		'cub200',
		'caltech101',
		'stanford_dogs',
		'voc2007',
		'cifar10',
		'imagenet' # 8
	]

        model_names = [a + '%' + b for a in all_architectures for b in source_datasets]
    return model_names

def count_atom(var, params):
    param_map = {id(v): k for k, v in params.items()}
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    res = []
    res1 = []
    res2 = []
    res3 = []

    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'
    
    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                res1.append(size_to_str(var.size()))
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                if 'None' not in node_name:
                    raise NameError
                res2.append('None')
            else:
                res3.append(str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    add_nodes(t)
    add_nodes(var.grad_fn)
    print(len(set(res3)))
    return res1, res2, res3

ok_32 = ['resnet20_cifar10', 'resnet20_cifar100', 'resnet20_svhn', 'resnet56_cifar10',
'resnet56_cifar100', 'resnet56_svhn', 'preresnet20_cifar10', 'preresnet20_cifar100', 'preresnet20_svhn',
 'preresnet56_cifar10', 'preresnet56_cifar100', 'preresnet56_svhn', 'seresnet56_cifar10', 'seresnet56_cifar100',
 'seresnet56_svhn', 'ror3_56_cifar10', 'ror3_56_cifar100', 'ror3_56_svhn', 'shakeshakeresnet20_2x16d_cifar10',
 'shakeshakeresnet20_2x16d_cifar100', 'shakeshakeresnet20_2x16d_svhn' ]
ok_299 = ['xception', 'inceptionv3', 'inceptionv4']

def build_mapping_1(type:str='both'):
    mapping = dict()
    count = 0
    all_models_name = []
    atoms = set()
    atoms.add('None')
    # 1. load model name:
    if type == 'both':
        name_imgclob = load_model_name()
        name_parc = load_model_name(path=None, type='parc')
        all_models_name = name_imgclob + name_parc
        assert len(all_models_name) == 32+105
    else:
        all_models_name = load_model_name(type=type)
    print(all_models_name) # 105+32=137

    # traverse all the models
    for m in tqdm(all_models_name):
        if '%' in m:
            archi, source = m.split('%')
            model = utils.load_source_model(archi, source)
        else:
            model = demo_from_homespace(m)  
        model.eval()
        if m in ok_32:
            inputs = torch.randn(1,3,32,32)
        elif m in ok_299:
            inputs = torch.randn(1,3,299,299)
        else:
            inputs = torch.randn(1,3,224,224)
        output = model(Variable(inputs))
        atom_1, atom_var, atom_back = count_atom(output, model.state_dict())
        g, lens = make_dot(output, model.state_dict())
        
        assert lens == len(atom_1) + len(atom_back) + len(atom_var)
        assert len(atom_1) == 0

        # g.view()
        # g.render(filename=f'tmp/{m}.png', view=False)
        # __import__('IPython').embed()
        for item in atom_back:
            atoms.add(item)

        del model, g

    #__import__('IPython').embed()
# build_mapping_1(type='both')


mapping_list = ['AdaptiveAvgPool2DBackward0',
 'AdaptiveMaxPool2DBackward0',
 'AddBackward0',
 'AddmmBackward0',
 'AvgPool2DBackward0',
 'AvgPool3DBackward0',
 'CatBackward0',
 'CloneBackward0',
 'ConstantPadNdBackward0',
 'ConvolutionBackward0',
 'DivBackward0',
 'ExpandBackward0',
 'HardtanhBackward0',
 'IndexSelectBackward0',
 'MaxBackward0',
 'MaxPool2DWithIndicesBackward0',
 'MeanBackward1',
 'MulBackward0',
 'NativeBatchNormBackward0',
 'None',
 'PowBackward0',
 'PreluBackward0',
 'ReluBackward0',
 'RepeatBackward0',
 'ReshapeAliasBackward0',
 'SigmoidBackward0',
 'SliceBackward0',
 'SoftmaxBackward0',
 'SplitWithSizesBackward0',
 'SqueezeBackward1',
 'SumBackward1',
 'TBackward0',
 'TransposeBackward0',
 'UnsqueezeBackward0',
 'UpsampleBilinear2DBackward1',
 'UpsampleNearest2DBackward1',
 'ViewBackward0']

def make_graph(type='both'):
    count = 0
    all_models_name = []
    all_graphs = []
    # 1. load model name:
    if type == 'both':
        name_imgclob = load_model_name()
        name_parc = load_model_name(path=None, type='parc')
        all_models_name = name_imgclob + name_parc
        assert len(all_models_name) == 32+105
    else:
        all_models_name = load_model_name(type=type)
    print(all_models_name) # 105+32=137

    # traverse all the models
    for m in tqdm(all_models_name):
        if '%' in m:
            archi, source = m.split('%')
            model = utils.load_source_model(archi, source)
        else:
            model = demo_from_homespace(m)  

        
        # model = model_loader.load_source_model('alexnet', 'cifar10')
        # model = demo_from_homespace('resnet18')
        model.eval()
        # __import__('IPython').embed()
        if m in ok_32:
            inputs = torch.randn(1,3,32,32)
        elif m in ok_299:
            inputs = torch.randn(1,3,299,299)
        else:
            inputs = torch.randn(1,3,224,224)
   
        output = model(Variable(inputs))
        g = gen_graph(output, model.state_dict())
        all_graphs.append(g)
        del model
    return all_graphs, all_models_name

        


def gen_graph(var, params):
    param_map = {id(v): k for k, v in params.items()}
    dot = nx.DiGraph()
    seen = set()
    node_to_idx = {}
    res1 = []
    res2 = []
    res3 = []

    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def get_node_index(s):
        
        if len(node_to_idx) == 0:
            print(f'facing {s}')
            node_to_idx[str(id(s))] = 0
            return 0
        if str(id(s)) not in node_to_idx:
            x = max(node_to_idx, key=node_to_idx.get)
            node_to_idx[str(id(s))] = node_to_idx[x] + 1
        # print(node_to_idx[str(id(s))])
        # print(node_to_idx)
        return node_to_idx[str(id(s))]

    def add_nodes(var=None):
        if var not in seen:
            idx = get_node_index(var)

            if torch.is_tensor(var):
                raise KeyError
                dot.add_node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                res1.append(size_to_str(var.size()))
            elif hasattr(var, 'variable'):
                # __import__('IPython').embed()
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
            
                if 'None' in node_name:
                    # dot.add_node(mapping_list.index('None'))
                    dot.add_node(idx, feature='None')
                else:
                    raise KeyError
                res2.append(node_name)
            else:
                
                node_name = str(type(var).__name__)
                dot.add_node(idx, feature=node_name)
                res3.append(str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.add_edge(get_node_index(u[0]), idx)
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    raise KeyError
                    dot.add_edge(str(id(t)), str(id(var)))
                    add_nodes(t)    
            
    add_nodes(var=var.grad_fn)
    node_labels = nx.get_node_attributes(dot, 'feature')

    assert set(node_labels.values()).issubset(set(mapping_list)), True

    # fig, ax = plt.subplots(figsize=(20, 20))
    # node_labels = nx.get_node_attributes(dot, 'feature')
    # # nx.draw_networkx_edge_labels(G=dot,ax=ax, pos=nx.kamada_kawai_layout(dot),)
    # nx.draw(G=dot, ax=ax, with_labels=True, pos=nx.nx_pydot.graphviz_layout(dot))
    # nx.draw_networkx_labels(G=dot, ax=ax, pos=nx.nx_pydot.graphviz_layout(dot), labels=node_labels)
    # plt.savefig('alex.pdf')
    # __import__('IPython').embed()
    return dot

# make_graph(type='parc')

def cluster(arr, k):
    print('k means...')
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(arr)
    labels = model.predict(arr)
    clusters = np.unique(labels)
    return labels, clusters

def graph_embs_cluster(data_type:str='parc', dim:int=128, centroid:int=4):
    path = f'{data_type}_archigraph.npy'

    print('generating...')
    all_graphs, all_models_names = make_graph(type=data_type)
    graph_to_vec = Graph2Vec(attributed=True, wl_iterations=2, dimensions=dim, epochs=100, workers=1)

    graph_to_vec.fit(graphs=all_graphs, )
    embs = graph_to_vec.get_embedding()
    

    labels, centers = cluster(embs, centroid)
    all_archi = pd.get_dummies(centers)


    res = {}
    embeddings = {}

    for idx, item in enumerate(all_models_names):
        one_hot = all_archi[labels[idx]].values
        res[item] = list(one_hot)
        embeddings[item] = embs[idx]
    np.save(path, res)
    np.save(f'{data_type}_embs.npy', embeddings)
    return res


def get_archi_parc(source, archi):
    mapping = graph_embs_cluster(data_type='parc', centroid=4)
    name = f'{archi}%{source}'
    return mapping[name]


def get_archi_imgclob(model_name=None, centroid=10):
    path = 'imgclob_archigraph.npy'
    # __import__('IPython').embed()
    if os.path.exists(path):
        mapping = np.load(path, allow_pickle=True).item()
    else:
        mapping = graph_embs_cluster(data_type='imgclob', centroid=centroid)

    return mapping[model_name]


get_archi_imgclob('alexnet')

    
