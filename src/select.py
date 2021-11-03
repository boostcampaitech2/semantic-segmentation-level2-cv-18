from torch.optim import AdamW, Adam
from madgrad import MADGRAD
import ttach as tta

from src.models import *
from src.losses import *
from src.transforms import tta_tf


# select model
def select_model(model_name):
	model = None
	if model_name == 'deeplabv3':
		model = DeepLabV3()
	elif model_name == 'deeplabv3+':
		model = DeepLabV3Plus()
	elif model_name == 'unet++':
		model = UnetPlusPlus()
	return model


# if tta is possible
def select_tta(model, is_tta):
	if is_tta:
		tta_model = tta.SegmentationTTAWrapper(model, tta_tf, merge_mode='mean')
		model = tta_model
	return model
		

# select criterion
def select_criterion(loss):
	criterion = None
	if loss == 'CE':
		criterion = nn.CrossEntropyLoss()
	elif loss == 'Dice':
		criterion = Dice()
	elif loss == 'Focal':
		criterion = Focal()
	elif loss == 'IoU':
		criterion = IoU()
	elif loss == 'DiceCE':
		criterion = DiceCELoss()
	elif loss == 'DiceFocal':
		criterion = DiceFocal()
	return criterion


# select optimizer
def select_optimizer(opt, model, lr, weight_decay):
	optimizer = None
	if opt == 'madgrad':
		optimizer = MADGRAD(
				params=model.parameters(),
				lr=lr, weight_decay=weight_decay
				)
	elif opt == 'Adam':
		optimizer = Adam(
				params=model.parameters(),
				lr=lr, weight_decay=weight_decay
				)
	elif opt == 'AdamW':
		optimizer = AdamW(
				params=model.parameters(),
				lr=lr, weight_decay=weight_decay
				)
	return optimizer
