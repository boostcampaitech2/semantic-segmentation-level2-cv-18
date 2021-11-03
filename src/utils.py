import torch
import os, random
from glob import glob
import numpy as np
import wandb

import pydensecrf.utils as utils
import pydensecrf.densecrf as dcrf


# check yaml
def check_yaml(yaml):
	assert yaml['model'] in ['unet++', 'deeplabv3', 'deeplabv3+'], f'Wrong Model: {yaml[model]}'
	assert yaml['dataset'] in ['custom', 'augmix'], f'Wrong Model: {yaml[dataset]}'
	assert yaml['loss'] in ['CE', 'Dice', 'Focal', 'IoU', 'DiceCE', 'DiceFocal'], f'Wrong Loss: {yaml[loss]}'
	assert yaml['optimizer'] in ['madgrad', 'Adam', 'AdamW'], f'Wrong Loss: {yaml[optimizer]}'
	assert yaml['checkpoint'] in ['all', 'loss', 'iou'], f'Wrong Loss: {yaml[checkpoint]}'


# fix seed
def seed_everything(random_seed=21):
	torch.manual_seed(random_seed)
	torch.cuda.manual_seed(random_seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(random_seed)
	random.seed(random_seed)


# wandb
def wandb_setting(max_lr, seed, batch_size, wandb_name):
	wandb.login()
	config = dict(learning_rate = max_lr, seed = seed, batch_size = batch_size)
	wandb.init(project="seg_LSE", entity="ai_tech_level2-cv-18",
			name=wandb_name, save_code=True, config=config)


# load model
def load_model(model, load_path, yaml_model, device):
	load_model_path = os.path.join(load_path, yaml_model + '.pt')
	checkpoint = torch.load(load_model_path, map_location=device)
	model.load_state_dict(checkpoint)
	print('Finish load model!')
	return model


# get classname for dataset
def get_classname(classID, cats):
	for i in range(len(cats)):
		if cats[i]['id']==classID:
			return cats[i]['name']
	return 'None'


# make directory
def make_dir(directory):
	tmp = ''
	lst = glob(os.path.join(directory, '*'))
	if len(lst) == 0:
		save_dir = os.path.join(directory, 'exp1')
	else:
		nums = []
		for l in lst:
			if 'exp' in l:
				nums.append(int(l[(l.index('exp') + 3):]))
		save_dir = os.path.join(directory, 'exp' + str(max(nums) + 1))
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	return save_dir


# get learning rate
def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']


# Reference
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
def label_accuracy_score(hist):
	acc = np.diag(hist).sum() / hist.sum()
	with np.errstate(divide='ignore', invalid='ignore'):
		acc_cls = np.diag(hist) / hist.sum(axis=1)
	acc_cls = np.nanmean(acc_cls)

	with np.errstate(divide='ignore', invalid='ignore'):
		iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
	mean_iu = np.nanmean(iu)
	
	freq = hist.sum(axis=1) / hist.sum()
	fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
	
	return acc, acc_cls, mean_iu, fwavacc, iu


def add_hist(hist, label_trues, label_preds, n_class):
	for lt, lp in zip(label_trues, label_preds):
		hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
	
	return hist


def _fast_hist(label_true, label_pred, n_class):
	mask = (label_true >= 0) & (label_true < n_class)
	hist = np.bincount(
		n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
		).reshape(n_class, n_class)

	return hist


# crf
def dense_crf_wrapper(args):
	return dense_crf(args[0], args[1])


def dense_crf(img, output_probs):
	# MAX_ITER = 10
	# POS_W = 3
	# POS_XY_STD = 1
	# Bi_W = 4
	# Bi_XY_STD = 67
	# Bi_RGB_STD = 3

	MAX_ITER = 50
	POS_W = 3
	POS_XY_STD = 3
	Bi_W = 4
	Bi_XY_STD = 49
	Bi_RGB_STD = 5

	c = output_probs.shape[0]
	h = output_probs.shape[1]
	w = output_probs.shape[2]
	U = utils.unary_from_softmax(output_probs)
	U = np.ascontiguousarray(U)
	img = np.ascontiguousarray(img)
	
	d = dcrf.DenseCRF2D(w, h, c)
	d.setUnaryEnergy(U)
	d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
	d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)
	
	Q = d.inference(MAX_ITER)
	Q = np.array(Q).reshape((c, h, w))
	return Q
