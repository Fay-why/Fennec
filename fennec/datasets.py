import datetime as dt
import logging
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import pickle
from collections import defaultdict
import random
import glob

import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import constants
import utils

dataset_objs     = {}
test_transforms  = {}
train_transforms = {}



data_transforms_224 = {
    "train": transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])]),
    "valid": transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])
}

data_transforms_32 = {
    "train": transforms.Compose([
                                transforms.RandomResizedCrop(32),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])]),
    "valid": transforms.Compose([transforms.Resize(32),
                                      transforms.CenterCrop(32),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([transforms.Resize(32),
                                transforms.CenterCrop(32),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])
}

data_transforms_299 = {
    "train": transforms.Compose([
                                transforms.RandomResizedCrop(299),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])]),
    "valid": transforms.Compose([transforms.Resize(299),
                                      transforms.CenterCrop(299),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([transforms.Resize(299),
                                transforms.CenterCrop(299),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])
}



#####################
# CIFAR 10 Dataset
#####################

class CIFAR10_base(datasets.CIFAR10):
	"""`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
	This is a subclass of the `CIFAR10` Dataset.
	"""
	base_folder = 'cifar-10'
	url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
	filename = "cifar-10-python.tar.gz"
	tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
	train_list = [
		['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
		['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
		['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
		['data_batch_4', '634d18415352ddfa80567beed471001a'],
		['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
	]

	test_list = [
		['test_batch', '40351d587109b95175f43aff81a1287e'],
	]
	meta = {
		'filename': 'batches.meta',
		'key': 'label_names',
		'md5': '5ff9c542aee3614f3951f8cda6e48888',
	}

class CIFAR10(Dataset):
	def __init__(self, root, train, transform, download=False):
		self.cifar10_base = CIFAR10_base(root=root,
										train=train,
										download=download,
										transform=transform)
		
	def __getitem__(self, index):
		data, target = self.cifar10_base[index]        
		return data, target, index

	def __len__(self):
		return len(self.cifar10_base)

dataset_objs['cifar10'] = CIFAR10

train_transforms['cifar10'] = transforms.Compose([
	transforms.Resize(224),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transforms['cifar10'] = transforms.Compose([
	transforms.Resize(224),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class ImageDataBase(Dataset):
	def __init__(self, root, train, transform):
		assert train == True
		# __import__('IPython').embed()
		train_dir = os.path.join(root, 'train')
		print(train_dir)
		self.base = datasets.ImageFolder(train_dir, transform=transform)

	def __getitem__(self, index):
		data, target = self.base[index]
		return data, target, index
	
	def __len__(self):
		return len(self.base)


# class Sport(ImageDataBase):
# 	def __init__(self, root, train, transform):
# 		assert train == True
# 		ImageDataBase.__init__(self, root, train, transform)



####################
# Dataset Loader
####################

path = '.'

def construct_dataset(dataset:str, path:str, train:bool=False, input_size:int=224, **kwdargs) -> torch.utils.data.Dataset:
	# transform = (train_transforms[dataset] if train else test_transforms[dataset])
	if input_size == 32:
		transform = data_transforms_32['test']
	elif input_size == 299:
		transform = data_transforms_299['test']
	else:
		transform = data_transforms_224['test']
	# transform = test_transforms[dataset][input_size] # Note: for training, use the above line. We're using the train set as the probe set, so use test transform
	print(path)
	# return dataset_objs[dataset](path, train, transform=transform, **kwdargs)
	return ImageDataBase(path, train, transform=transform, **kwdargs)

def get_dataset_path(dataset:str) -> str:
	return f'./data/{dataset}/'


class ClassMapCache:
	""" Constructs and stores a cache of which instances map to which classes for each datset. """

	def __init__(self, dataset:str, train:bool):
		self.dataset = dataset
		self.train = train

		if not os.path.exists(self.cache_path):
			self.construct_cache()
		else:
			with open(self.cache_path, 'rb') as f:
				self.idx_to_class, self.class_to_idx = pickle.load(f)


	def construct_cache(self):
		print(f'Constructing class map for {self.dataset}...')
		dataset    = construct_dataset(self.dataset, get_dataset_path(self.dataset), self.train)
		dataloader = torch.utils.data.DataLoader(dataset, 32, shuffle=False)

		self.idx_to_class = []
		self.class_to_idx = defaultdict(list)

		idx = 0

		for batch in tqdm(dataloader):
			y = batch[1]
			single_class = (y.ndim == 1)

			for _cls in y:
				if single_class:
					_cls = _cls.item()
				
				self.idx_to_class.append(_cls)
				
				if single_class:
					self.class_to_idx[_cls].append(idx)
				
				idx += 1
		
		self.class_to_idx = dict(self.class_to_idx)

		utils.make_dirs(self.cache_path)
		with open(self.cache_path, 'wb') as f:
			pickle.dump((self.idx_to_class, self.class_to_idx), f)



	@property
	def cache_path(self):
		return f'{path}/cache/class_map/{self.dataset}_{"train" if self.train else "test"}.pkl'


class DatasetCache(torch.utils.data.Dataset):
	""" Constructs and stores a cache for the dataset post-transform. """

	def __init__(self, dataset:str, train:bool, input_size:int=224):
		self.dataset = dataset
		self.train = train
		self.input_size = input_size
		self.cache_folder = os.path.split(self.cache_path(0))[0]
		
		if not os.path.exists(self.cache_path(0)):
			os.makedirs(self.cache_folder, exist_ok=True)
			self.construct_cache()
		
		self.length = len(glob.glob(self.glob_path()))
		self.class_map = ClassMapCache(dataset, train)
		
		super().__init__()

	def cache_path(self, idx:int) -> str:
		return f'{path}/cache/datasets/{self.dataset}_{self.input_size}/{"train" if self.train else "test"}_{idx}.npy'
		
	def glob_path(self) -> str:
		return f'{path}/cache/datasets/{self.dataset}_{self.input_size}/{"train" if self.train else "test"}_*'

	def construct_cache(self):
		print(f'Constructing dataset cache for {self.dataset}...')
		dataset    = construct_dataset(self.dataset, get_dataset_path(self.dataset), self.train, self.input_size)
		dataloader = torch.utils.data.DataLoader(dataset, 32, shuffle=False)

		idx = 0

		for batch in tqdm(dataloader):
			x = batch[0]
			
			for i in range(x.shape[0]):
				np.save(self.cache_path(idx), x[i].numpy().astype(np.float16))
				idx += 1
	
	def __getitem__(self, idx:int) -> tuple:
		x = torch.from_numpy(np.load(self.cache_path(idx)).astype(np.float32))
		y = self.class_map.idx_to_class[idx]
		return x, y

	def __len__(self):
		return self.length


class BalancedClassSampler(torch.utils.data.DataLoader):
	""" Samples from a dataloader such that there's an equal number of instances per class. """

	def __init__(self, dataset:str, batch_size:int, instances_per_class:int, train:bool=True, **kwdargs):
		num_classes = constants.num_classes[dataset]
		dataset_obj = DatasetCache(dataset, train)
		map_cache = ClassMapCache(dataset, train)

		sampler_list = []

		for _, v in map_cache.class_to_idx.items():
			random.shuffle(v)
		
		for _ in range(instances_per_class):
			for i in range(num_classes):
				if i in map_cache.class_to_idx:
					idx_list = map_cache.class_to_idx[i]
					
					if len(idx_list) > 0:
						sampler_list.append(idx_list.pop())
		
		super().__init__(dataset_obj, batch_size, sampler=sampler_list, **kwdargs)


class FixedBudgetSampler(torch.utils.data.DataLoader):
	""" Samples from a dataloader such that there's a fixed number of samples. Classes are distributed evenly. """

	def __init__(self, dataset:str, batch_size:int, probe_size:int, train:bool=True, min_instances_per_class:int=2, input_size=224, **kwdargs):
		num_classes = constants.num_classes[dataset]
		dataset_obj = DatasetCache(dataset, train, input_size)
		map_cache = ClassMapCache(dataset, train)

		sampler_list = []
		last_len = None

		for _, v in map_cache.class_to_idx.items():
			random.shuffle(v)
		
		#class_indices = list(range(num_classes))
		#class_indices = [i for i in class_indices if i in map_cache.class_to_idx] # Ensure that i exists
		class_indices = [i for i in map_cache.class_to_idx]
		#__import__('IPython').embed()
		# Whether or not to subsample the classes to meet the min_instances and probe_size quotas 
		if num_classes * min_instances_per_class > probe_size:
			# Randomly shuffle the classes so if we need to subsample the classes, it's random.
			random.shuffle(class_indices)
			# Select a subset of the classes to evaluate on.
			class_indices = class_indices[:probe_size // min_instances_per_class]
		# Updated the list of samples (sampler_list) each iteration with 1 image for each class
		# We stop when we're finished or there's a class we didn't add an image for (i.e., out of images).
		while last_len != len(sampler_list) and len(sampler_list) < probe_size:
			# This is to ensure we don't infinitely loop if we run out of images
			last_len = len(sampler_list)

			for i in class_indices:
				idx_list = map_cache.class_to_idx[i]
				
				# If we still have images left of this class
				if len(idx_list) > 0:
					# Add it to the list of samples
					sampler_list.append(idx_list.pop())
				
				if len(sampler_list) >= probe_size:
					break
		print(sampler_list)
		super().__init__(dataset_obj, batch_size, sampler=sampler_list, **kwdargs)
		

class RandomFixedBudgetSampler(torch.utils.data.DataLoader):
	def __init__(self, dataset:str, batch_size:int, probe_size:int, train:bool=True, **kwdargs) -> None:
		num_classes = constants.num_classes[dataset]
		dataset_obj = DatasetCache(dataset, train)
		map_cache = ClassMapCache(dataset, train)

		if dataset == 'voc2007':
			samples = list(range(len(dataset_obj)))
			random.shuffle(samples)
			super().__init__(dataset_obj, batch_size, sampler=samples[:probe_size], **kwdargs)
			return
		
		data_len = len(dataset_obj)
		sampler_list = np.random.choice(a=data_len, size=probe_size, replace=False)
		super().__init__(dataset_obj, batch_size, sampler=sampler_list, **kwdargs)


		
		
if __name__ == '__main__':
	#FixedBudgetSampler('voc2007', 128, 500, train=True)
	FixedBudgetSampler('sport', 128, 100, train=True)
	__import__('IPython').embed()