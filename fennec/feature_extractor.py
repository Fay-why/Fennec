# get the forward features of different models

import itertools
import os
import pickle
import time

import torch.nn as nn
import torch
import numpy as np
import clip
import open_clip
from tqdm import tqdm
from torchvision.models import resnet50, resnet101, resnet152
# from segment_anything import sam_model_registry, SamPredictor
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model

import datasets
import utils
from constants import variables, num_classes

features_pre_path = './'
target_dataloaders = {}

# input size should be 299 or 32 for some models
ok_32 = ['resnet20_cifar10', 'resnet20_cifar100', 'resnet20_svhn', 'resnet56_cifar10',
'resnet56_cifar100', 'resnet56_svhn', 'preresnet20_cifar10', 'preresnet20_cifar100', 'preresnet20_svhn',
 'preresnet56_cifar10', 'preresnet56_cifar100', 'preresnet56_svhn', 'seresnet56_cifar10', 'seresnet56_cifar100',
 'seresnet56_svhn', 'ror3_56_cifar10', 'ror3_56_cifar100', 'ror3_56_svhn', 'shakeshakeresnet20_2x16d_cifar10',
 'shakeshakeresnet20_2x16d_cifar100', 'shakeshakeresnet20_2x16d_svhn' ]
ok_299 = ['xception', 'inceptionv3', 'inceptionv4']

def load_model_name(path='./oracles/model_100.txt'):
    model_names = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(':')
            if len(line) != 2:
                continue
            model_names.append(eval(line[0]))
    return model_names

def demo_from_homespace(model_name, status=True):
    net = ptcv_get_model(model_name, pretrained=status)
    # print(f"{model_name}: download ok")
    return net

def get_model(model_name):
    model = demo_from_homespace(model_name)
    model.eval()

    def extract_feats(self, args):
        x = args[0]
        # print(x.get_device()) -1
        # __import__('IPython').embed()

        #model._extracted_feats[x.get_device()] = x
        model._extracted_feats[0] = x
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.register_forward_pre_hook(extract_feats)

    return model

def get_source_model(dataset):
    model = resnet50(pretrained=False)
    classifier = nn.Linear(model.fc.in_features, num_classes[dataset])
    model.fc = classifier
    base_name_path = f'./models/rsa_models/{dataset}.pth'
    # __import__('IPython').embed()
    model.load_state_dict(torch.load(base_name_path, map_location=torch.device('cpu'))['state_dict'])
    
    model.eval()

    def extract_feats(self, args):
        x = args[0]
        # print(x.get_device()) -1
        # __import__('IPython').embed()

        #model._extracted_feats[x.get_device()] = x
        model._extracted_feats[0] = x
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.register_forward_pre_hook(extract_feats)
    return model


def gen_probe_sets(cache_path, sample_size, source_dataset, target_dataset, architecture, run, save=True):
    if source_dataset == target_dataset:
        return

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    if source_dataset != target_dataset:
        model = get_model(architecture)
    else: 
        raise NotImplementedError
        model = get_source_model(target_dataset)

    input_size = 224
    if architecture in ok_299:
        input_size = 299
    elif architecture in ok_32:
        input_size = 32
        

    dataloader_key = (target_dataset, run, input_size)
    if dataloader_key not in target_dataloaders:
        utils.seed_all(2020+run*3037)
        dataloader = datasets.FixedBudgetSampler(
            target_dataset, batch_size=128, probe_size=sample_size, input_size=input_size)
        target_dataloaders[dataloader_key] = dataloader

    dataloader = target_dataloaders[dataloader_key]

    # get feat:
    with torch.no_grad():
        all_y = []
        all_feats = []
        all_probs = []

        for x, y in dataloader:
            # Support for using multiple GPUs
            # self.model._extracted_feats = [None] * torch.cuda.device_count()
            model._extracted_feats = [None]
            # x = x.cuda() # we don't use cuda...

            preds = model(x)

            all_y.append(y.cpu())
            all_probs.append(torch.nn.functional.softmax(preds, dim=-1).cpu())
            all_feats.append(
                torch.cat([x for x in model._extracted_feats], dim=0))

        all_y = torch.cat(all_y, dim=0).numpy()
        all_feats = torch.cat(all_feats, dim=0).numpy()
        all_probs = torch.cat(all_probs, dim=0).numpy()
        print(len(all_feats))
        params = {
            'features': all_feats,
            'probs': all_probs,
            'y': all_y,
            'source_dataset': source_dataset,
            'target_dataset': target_dataset,
            'architecture': architecture
        }

    if save:
        utils.make_dirs(cache_path)
        with open(cache_path, 'wb') as f:
            pickle.dump(params, f)

    return params


def offline(probe_size=500):
    factors = [variables['Architecture'], variables['Source Dataset'],
               variables['Target Dataset'], list(range(5))]
    # for rsa training...
    # factors = [['resnet50'], variables['Source Dataset'],
    #            variables['Target Dataset'], list(range(5))]

    iter_obj = []
    iter_obj += list(itertools.product(*factors))
    # __import__('IPython').embed()
    for arch, source, target, run in tqdm(iter_obj):
        cache_path = f'{features_pre_path}/cache/probes_{probe_size}/{arch}_{source}_{target}_{run}.pkl'
        gen_probe_sets(cache_path=cache_path, sample_size=probe_size, source_dataset=source,
                       target_dataset=target, architecture=arch, run=run, save=True)


def get_forward_vit(dataloader, model_name, data_name, sample_size, run):
    # https://huggingface.co/blog/zh/image-similarity
    # https://huggingface.co/google/vit-huge-patch14-224-in21k

    model = ViTModel.from_pretrained('google/vit-huge-patch14-224-in21k')
    model.eval()
    all_feats = []
    all_y = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
   
    model.to(device)
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            all_y.append(labels)
            new_batch = {'pixel_values': images.to(device)}
            feats = model(**new_batch).last_hidden_state[:,0].cpu()
            all_feats.append(feats)
    all_feats = torch.cat(all_feats, dim=0).numpy()
    all_y = torch.cat(all_y, dim=0).numpy()

    np.save(
            f'{features_pre_path}data_{sample_size}/{data_name}_{model_name}_{run}.npy', all_feats)
    print(f"saving features done! {model_name} from {data_name}")
    return all_feats, all_y


def get_forward_divo(dataloader, model_name, data_name, sample_size, run):
    # model = hubconf.dinov2_vitg14()
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    model = torch.hub.load('~/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vitg14',
                            source='local')
    all_feats = []
    all_y = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
   
    model.to(device)
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            all_y.append(labels)
            feats = model(images.to(device))
            all_feats.append(feats)
    # __import__('IPython').embed()
    all_feats = torch.cat(all_feats, dim=0).numpy()
    all_y = torch.cat(all_y, dim=0).numpy()

    np.save(
            f'{features_pre_path}data_{sample_size}/{data_name}_{model_name}_{run}.npy', all_feats)
    print(f"saving features done! {model_name} from {data_name}")
    return all_feats, all_y


def get_forward_clip(dataloader, model_name="clip", data_name=None, sample_size=None, run=None, save=True):
    '''
        clip.available_models():
        ['RN50',
        'RN101',
        'RN50x4',
        'RN50x16',
        'RN50x64',
        'ViT-B/32',
        'ViT-B/16',
        'ViT-L/14',
        'ViT-L/14@336px']

        open_clip.list_pretrained():
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L/14', pretrained='datacomp_xl_s13b_b90k')
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k') #1280d
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load('ViT-B/32', device) # 512d
    if model_name == "clip":
        print('using clip')
        model, _ = clip.load('ViT-L/14', device, jit=True)  # 768d
    elif model_name == "open_clip":
        print('using open clip')
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-bigG-14', jit=True, pretrained='~/.cache/huggingface/hub/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/blobs/0d5318839ad03607c48055c45897c655a14c0276a79f6b867934ddd073760e39')
    model.eval()
    
    all_feats = []
    all_y = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            all_y.append(labels)
            feats = model.encode_image(images.to(device))
            all_feats.append(feats)
    all_feats = torch.cat(all_feats, dim=0).numpy()
    all_y = torch.cat(all_y, dim=0).numpy()
    if save:
        np.save(
            f'{features_pre_path}data_{sample_size}/{data_name}_{model_name}_{run}.npy', all_feats)
        print(f"saving features done! {model_name} from {data_name}")
    return all_feats, all_y


def get_refer_model(model_name=None, load_type=None):
    if model_name == "resnet50":
        embedder = resnet50(pretrained=True).eval()
    elif model_name == "resnet101":
        embedder = resnet101(pretrained=True).eval()
    else:
        raise NotImplementedError

    if load_type == "normal":
        print("using Identity hook method")
        embedder.fc = torch.nn.Identity()

    for p in embedder.parameters():
        p.requires_grad = False
    return embedder

def get_forward_resnet(dataloader=None, model_name="resnet50", data_name=None, sample_size=None, run=None):
    # using identity fc layer, no need any hook function
    model = get_refer_model(model_name=model_name, load_type="normal")
    model.eval()
    all_y = []
    all_feats = []
    for batch in dataloader:
        batch0 = batch[0]
        with torch.no_grad():
            pred = model(batch0)
            all_feats.append(pred)
            all_y.append(batch[1])
    all_feats = torch.cat(all_feats, dim=0).numpy()
    all_y = torch.cat(all_y, dim=0).numpy()
    np.save(
        f'{features_pre_path}data_{sample_size}/{data_name}_{model_name}_{run}.npy', all_feats)
    return all_feats, all_y


def get_forward(data_name=None, sample_size=None, model_name="resnet50", batch_size=128, load_type="normal", run=0):
    if os.path.exists(f"{features_pre_path}data_{sample_size}/{data_name}_{model_name}_{run}.npy"):
        print(
            f"loading from {features_pre_path}data_{sample_size}/{data_name}_{model_name}_{run}.npy")
        features = np.load(
            f'{features_pre_path}data_{sample_size}/{data_name}_{model_name}_{run}.npy')
        assert features.shape[0] == sample_size
        return features, 'fake_y'

    utils.seed_all(2020 + run * 3037)
    dataloader = datasets.FixedBudgetSampler(
        data_name, batch_size=batch_size, probe_size=sample_size)

    if model_name == "clip" or model_name == "open_clip" or model_name == "open_clip_L":
        return get_forward_clip(dataloader, model_name=model_name,  data_name=data_name, sample_size=sample_size, run=run)
    elif model_name == "divo":
        return get_forward_divo(dataloader, model_name=model_name, data_name=data_name, sample_size=sample_size, run=run)
    elif 'resnet' in model_name:
        return get_forward_resnet(dataloader, model_name, data_name, sample_size, run)
    elif model_name == 'vit':
        return get_forward_vit(dataloader, model_name, data_name, sample_size, run)


def get_forward_random(data_name=None, sample_size=None, model_name="resnet50", batch_size=128, load_type="normal", run=0):
    run = 'random'
    if os.path.exists(f"{features_pre_path}data_{sample_size}/{data_name}_{model_name}_{run}.npy"):
        print(
            f"loading from {features_pre_path}data_{sample_size}/{data_name}_{model_name}_{run}.npy")
        features = np.load(
            f'{features_pre_path}data_{sample_size}/{data_name}_{model_name}_{run}.npy')
        assert features.shape[0] == sample_size
        return features, 'fake_y'
    # utils.seed_all(2020 + 1 * 3037)
    dataloader = datasets.RandomFixedBudgetSampler(
        data_name, batch_size=batch_size, probe_size=sample_size)

    if model_name == "clip" or model_name == "open_clip" or model_name == "open_clip_L":
        return get_forward_clip(dataloader, model_name=model_name,  data_name=data_name, sample_size=sample_size, run=run, save=True)
    elif model_name == "divo":
        return get_forward_divo(dataloader, model_name=model_name, data_name=data_name, sample_size=sample_size, run=run)
    else:
        return get_forward_resnet(dataloader, model_name, data_name, sample_size, run)
