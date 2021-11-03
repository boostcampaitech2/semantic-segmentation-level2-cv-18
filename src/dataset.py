import torch
import os, json, cv2, random
import pandas as pd
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.transforms import *
from src.utils import get_classname


def data_settting(dataset_path, anns_file_path):
	# read annotations
	with open(anns_file_path, 'r') as f:
		dataset = json.loads(f.read())

	categories = dataset['categories']
	anns = dataset['annotations']
	imgs = dataset['images']
	l_cats = len(categories)
	l_annotations = len(anns)
	l_images = len(imgs)

	# load categories & super categories
	cat_names = []
	super_cat_names = []
	super_cat_ids = {}
	super_cat_last_name = ''
	l_super_cats = 0
	for category in categories:
		cat_names.append(category['name'])
		super_cat_name = category['supercategory']
		# add new supercat
		if super_cat_name != super_cat_last_name:
			super_cat_names.append(super_cat_name)
			super_cat_ids[super_cat_name] = l_super_cats
			super_cat_last_name = super_cat_name
			l_super_cats += 1
	print('Number of super categories:', l_super_cats)
	print('Number of categories:', l_cats)
	print('Number of annotations:', l_annotations)
	print('Number of images:', l_images)

	# count annotations
	cat_histogram = np.zeros(l_cats,dtype=int)
	for ann in anns:
		cat_histogram[ann['category_id']-1] += 1

	# convert to DataFrame
	df = pd.DataFrame({
			'Categories': cat_names,
			'Number of annotations': cat_histogram
			})
	df = df.sort_values('Number of annotations', 0, False)

	# category labeling 
	sorted_temp_df = df.sort_index()
	# add background label, relabeling
	sorted_df = pd.DataFrame(["Backgroud"], columns = ["Categories"])
	sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)
	category_names = list(sorted_df.Categories)

	return category_names


# dataset including augmix
class AugmixDataSet(Dataset):
	"""COCO format"""""
	def __init__(self, data_dir, mode = 'train', transform=None,
			category_names=None, dataset_path=None, augmix=None, augmix_prob=0):
		super().__init__()
		self.mode = mode
		self.transform = transform
		self.category_names = category_names
		self.dataset_path = dataset_path
		self.coco = COCO(data_dir)							        
		self.augmix = augmix
		self.augmix_prob = augmix_prob
		
	def __getitem__(self, index: int):
		image_id = self.coco.getImgIds(imgIds=index)
		image_infos = self.coco.loadImgs(image_id)[0]
	
		images = cv2.imread(os.path.join(self.dataset_path,
						image_infos['file_name']))
		# need to be uint8 for augmentation
		images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
		
		if (self.mode in ('train', 'val')):
			ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
			anns = self.coco.loadAnns(ann_ids)

			cat_ids = self.coco.getCatIds()
			cats = self.coco.loadCats(cat_ids)
			
			masks = np.zeros((image_infos["height"], image_infos["width"]))
			anns = sorted(anns,
					key=lambda idx : len(idx['segmentation'][0]),
					reverse=False)
			for i in range(len(anns)):
				className = get_classname(anns[i]['category_id'], cats)
				pixel_value = self.category_names.index(className)
				masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
			masks = masks.astype(np.int8)

			if self.augmix is not None:
				tmp = np.random.rand(1)
				if tmp < self.augmix_prob:
					images, masks = self.search_augmix(images, masks)

			if self.transform is not None:
				transformed = self.transform(image=images, mask=masks)
				images = transformed["image"]
				masks = transformed["mask"]
			return images, masks, image_infos

		if self.mode == 'test':
			if self.transform is not None:
				transformed = self.transform(image=images)
				images = transformed["image"]
			return images, image_infos
				
	def __len__(self) -> int:
		return len(self.coco.getImgIds())

	def search_augmix(self, images, masks):
		tmp_dict = {0: 3, 1: 4, 2: 5, 3: 9, 4: 10}
		num = [3, 4, 5, 9, 10]

		label = random.choice(num)
		idx = np.random.randint(len(self.augmix[label]))
		augmix_img = self.augmix[label][idx]
		
		augmix_mask = np.zeros((512, 512))
		augmix_mask[augmix_img[:, :, 0] != 0] = label
		images[augmix_img != 0] = augmix_img[augmix_img != 0]
		masks[augmix_mask!= 0] = augmix_mask[augmix_mask != 0]
		
		return images, masks


# dataset for pseudo
class PseudoDataset(Dataset):
	def __init__(self, data_dir, pseudo_csv, transform=None):
		self.data_dir = data_dir
		self.df = pd.read_csv(pseudo_csv)
		self.transform = transform
		self.tf = A.Resize(512, 512, interpolation= cv2.INTER_NEAREST)
	
	def __len__(self):
		return len(self.df)
	
	def __getitem__(self, idx):
		img_path = os.path.join(self.data_dir, self.df.iloc[idx, 0])
		mask = np.array(list(map(int, self.df.iloc[idx, 1].split())))
		mask = mask.reshape(256, 256)
		mask = self.tf(image=mask)['image']
		img = cv2.cvtColor(cv2.imread(img_path),
						cv2.COLOR_BGR2RGB).astype(np.uint8)
		
		if self.transform is not None:
			transformed = self.transform(image=img, mask=mask)
			images = transformed['image']
			masks = transformed['mask']
			
		return images, masks, img_path
	

# select dataset
def select_dataset(dataset, data_dir, augmix_data=None):
	# path
	dataset_path = data_dir
	train_path = dataset_path + '/train_all.json'
	val_path = dataset_path + '/val.json'
	test_path = dataset_path + '/test.json'

	# transform
	train_transform = train_tf
	val_transform = tf
	test_transform = tf
	# category_names
	category_names = data_settting(dataset_path, train_path)
	
	# test dataloader
	if dataset == 'test':
		test_dataset = AugmixDataSet(
				data_dir=test_path, mode='test', transform=test_transform,
				category_names=category_names, dataset_path=dataset_path
				)
		return test_dataset

	# basic train & val dataloader
	if dataset == 'custom':
		train_dataset = AugmixDataSet(
				data_dir=train_path, mode='train', transform=train_transform,
				category_names=category_names, dataset_path=dataset_path
				)
		val_dataset = AugmixDataSet(
				data_dir=val_path, mode='val', transform=val_transform,
				category_names=category_names, dataset_path=dataset_path
				)
	
	# train & val dataloader including augmix
	elif dataset == 'augmix':
		train_dataset = AugmixDataSet(
				data_dir=train_path, mode='train', transform=train_transform,
				category_names=category_names,dataset_path=dataset_path,
				augmix=augmix_data.item(), augmix_prob=0.4
				)
		val_dataset = AugmixDataSet(
				data_dir=val_path, mode='val', transform=val_transform,
				category_names=category_names, dataset_path=dataset_path,
				augmix=augmix_data.item(), augmix_prob=0.4
				)

	return train_dataset, val_dataset
