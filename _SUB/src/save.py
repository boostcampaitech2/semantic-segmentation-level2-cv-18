import torch, os 
import pandas as pd
from tqdm import tqdm


# save model
def save_model(model, saved_dir, file_name):
	check_point = {'net': model.state_dict()}
	output_path = os.path.join(saved_dir, file_name)
	torch.save(model.state_dict(), output_path)


# save best model
def save_best_model(yaml_model, saved_dir, model, epoch, val, best_epoch, best, checkpoint):
	best = val
	best_epoch = epoch
	print(f'* Best performace: Epoch: {epoch + 1}, {checkpoint}: {best:.4f}')
	save_model(model, saved_dir, yaml_model + f'_{checkpoint}.pt')
	print(f'* Save best {checkpoint} model!')
	
	return best, best_epoch


# select checkpoint
def select_checkpoint(yaml_model, saved_dir, model, epoch, val, best_epoch, best, tmp):
	if tmp == 'iou':
		if val >= best:
			best, best_epoch = save_best_model(
				yaml_model, saved_dir, model, epoch, val, best_epoch, best, tmp
				)
	elif tmp == 'loss':
		if val <= best:
			best, best_epoch = save_best_model(
				yaml_model, saved_dir, model, epoch, val, best_epoch, best, tmp
				)

	return best, best_epoch


# save model according to the checkpoint
def save_checkpoint(checkpoint, yaml_model, saved_dir, model,
					epoch, val_loss, val_iou, best_epoch, best_loss, best_iou):
	if checkpoint == 'all':
		best_iou, best_epoch = select_checkpoint(
			yaml_model, saved_dir, model, epoch, val_iou, best_epoch, best_iou, 'iou'
			)
		best_loss, best_epoch = select_checkpoint(
			yaml_model, saved_dir, model, epoch, val_loss, best_epoch, best_loss, 'loss'
			)
	elif checkpoint == 'loss':
		best_loss, best_epoch = select_checkpoint(
			yaml_model, saved_dir, model, epoch, val_loss, best_epoch, best_loss, 'loss'
			)
	elif checkpoint == 'iou':
		best_iou, best_epoch = select_checkpoint(
			yaml_model, saved_dir, model, epoch, val_iou, best_epoch, best_iou, 'iou'
			)
	return best_epoch, best_loss, best_iou
