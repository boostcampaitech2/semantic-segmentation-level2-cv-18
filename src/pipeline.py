import torch
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import wandb

import albumentations as A
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

from src.utils import *
from src.save import save_checkpoint


# train
def train(yaml, model, train_loader, val_loader, 
		criterion, optimizer, scheduler, device, tmp, pseduo_loader):
	seed_everything(yaml['seed'])
	print('Training...')
	saved_dir = make_dir(yaml['saved_dir'])
	epochs = yaml['epochs']
	log_step = yaml['log_step']
	n_class = 11
	best_epoch = 0
	best_iou = 0.0
	best_loss = 100

	for epoch in range(epochs):
		print('-' * 22 + f' Epoch [{epoch + 1}/{epochs}] ' + '-' * 22)
		train_iou = 0.0
		val_iou = 0.0
		train_loss = 0.0
		val_iou = 0.0

		if tmp == 1:
			print('\nStart pseudo traing...')
			model.train()
			hist = np.zeros((n_class, n_class))
			for step, (images, masks, _) in enumerate(tqdm(pseduo_loader)):
				images = torch.stack(images).to(device) # (B, C, H, W)
				masks = torch.stack(masks).long().to(device) # (B, C, H, W)

				optimizer.zero_grad()
				outputs = model(images)
				loss = criterion(outputs, masks)
				loss.backward()
				optimizer.step()
				scheduler.step()

				wandb.log({'Learning Rate': get_lr(optimizer)})

				outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
				masks = masks.detach().cpu().numpy()
				hist = add_hist(hist, masks, outputs, n_class=n_class)
				miou = label_accuracy_score(hist)[2]

				train_loss += loss
				train_iou += miou

				if (step + 1) % log_step == 0:
					print(
						f'Step [{step + 1}/{len(pseduo_loader)}] | Train_Loss: {train_loss / log_step:.4f}, Train_IoU: {train_iou / log_step:.4f}'
						)
					wandb.log({
							'Pseduo/Loss': train_loss / log_step,
							'Pseduo/mIoU': train_iou / log_step
							})
					train_loss = 0.0
					train_iou = 0.0

			train_loss = 0.0
			train_iou = 0.0

		print('\nStart trainig...')
		model.train()
		hist = np.zeros((n_class, n_class))
		for step, (images, masks, _) in enumerate(tqdm(train_loader)):
			images = torch.stack(images).to(device) # (B, C, H, W)
			masks = torch.stack(masks).long().to(device) # (B, C, H, W)

			outputs = model(images)
			loss = criterion(outputs, masks)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			wandb.log({'Learning Rate': get_lr(optimizer)})

			outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
			masks = masks.detach().cpu().numpy()
			hist = add_hist(hist, masks, outputs, n_class=n_class)
			miou = label_accuracy_score(hist)[2]

			train_loss += loss
			train_iou += miou

			if (step + 1) % log_step == 0:
				print(
					f'Step [{step + 1}/{len(train_loader)}] | Train_Loss: {train_loss / log_step:.4f}, Train_IoU: {train_iou / log_step:.4f}'
					)
				wandb.log({
						'Train/Loss': train_loss / log_step,
						'Train/mIoU': train_iou / log_step
						})
				train_loss = 0.0
				train_iou = 0.0
		
		val_loss, val_iou = validation(
				yaml['seed'], epoch, epochs, model,
				val_loader, criterion, device
				)
		best_epoch, best_loss, best_iou = save_checkpoint(
				yaml['checkpoint'],	yaml['model'], saved_dir, model,
				epoch, val_loss, val_iou, best_epoch, best_loss, best_iou
				)
		print(f"* Current best mIoU is {best_iou:.4f} & best loss is {best_loss:.4f} at Epoch {best_epoch + 1}")


# validation
def validation(seed, epoch, epochs, model, val_loader, criterion, device):
	seed_everything(seed)
	print('\nStart validation...')
	model.eval()

	with torch.no_grad():
		category = [
				'Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal',
				'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
		]
		n_class = 11
		val_loss = 0
		cnt = 0

		hist = np.zeros((n_class, n_class))
		for step, (images, masks, _) in enumerate(tqdm(val_loader)):
			images = torch.stack(images).to(device)
			masks = torch.stack(masks).long().to(device)

			outputs = model(images)
			loss = criterion(outputs, masks)
			val_loss += loss
			cnt += 1

			outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
			masks = masks.detach().cpu().numpy()
			hist = add_hist(hist, masks, outputs, n_class=n_class)

		acc, acc_cls, val_iou, fwavacc, IoU = label_accuracy_score(hist)
		IoU_by_class = {classes : round(IoU, 4) for IoU, classes in zip(IoU, category)}
		avrg_loss = val_loss / cnt

		print(
			f'Val_Average_Loss: {avrg_loss:.4f}, Val_Acc : {acc:.4f}, Val_IoU: {val_iou:.4f}'
			)
		print(f'IoU by class : {IoU_by_class}')
		wandb.log({
				'Val/Loss': avrg_loss, 'Val/Acc': acc, 'Val/mIoU': val_iou
				})
		wandb.log({
				f'Category/{class_name}': IoU for class_name, IoU in IoU_by_class.items()
				})

	return avrg_loss, val_iou


# test
def test(seed, model, test_loader, is_crf, device):
	seed_everything(seed)
	print('\nStart prediction...')
	model.eval()

	size = 256
	transform = A.Compose([A.Resize(size, size)])
	file_name_list = []
	preds_array = np.empty((0, size*size), dtype=np.long)

	with torch.no_grad():
		for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
			outs = model(torch.stack(imgs).float().to(device))
			if is_crf == True:
				probs = F.softmax(outs, dim=1).data.cpu().numpy()
				pool = mp.Pool(mp.cpu_count())
				images = torch.stack(imgs).data.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
				probs = np.array(pool.map(dense_crf_wrapper, zip(images, probs)))
				pool.close()
				oms = np.argmax(probs, axis=1)
			else:
				oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()

			temp_mask = []
			for img, mask in zip(np.stack(imgs), oms):
				transformed = transform(image=img, mask=mask)
				mask = transformed['mask']
				temp_mask.append(mask)
		
			oms = np.array(temp_mask)
			oms = oms.reshape([oms.shape[0], size*size]).astype(int)
			preds_array = np.vstack((preds_array, oms))

			file_name_list.append([i['file_name'] for i in image_infos])
	print('Finish prediction!')

	return file_name_list, preds_array
