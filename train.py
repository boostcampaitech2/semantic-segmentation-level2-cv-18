import argparse, yaml
import torch, os
import wandb
import warnings
warnings.filterwarnings('ignore')

from src.utils import *
from src.select import *
from src.dataloader import tv_dataloader
from src.schedulers import CosineAnnealingWarmupRestarts
from src.pipeline import train


print(f'Pytorch version: {torch.__version__}')
print(f'GPU available: {torch.cuda.is_available()}')
print(f'Device name: {torch.cuda.get_device_name(0)}')
print(f'Device count: {torch.cuda.device_count()}')


# main
def main(yaml):
	def collate_fn(batch):
		return tuple(zip(*batch))

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# fix seed
	seed_everything(yaml['seed'])
	# wandb
	wandb_setting(yaml['max_lr'], yaml['seed'], yaml['batch_size'], yaml['wandb_name'])
	# loader
	train_loader, val_loader, pseudo_loader = tv_dataloader(
			yaml['dataset'], yaml['data_dir'], yaml['batch_size'],
			collate_fn, pseudo_csv=args.ps_csv
			)
	# model
	model = select_model(yaml['model'])
	if yaml['is_load']:
		model = load_model(model, yaml['load_path'], yaml['model'])
	model = model.to(device)
	wandb.watch(model)
	# criterion
	criterion = select_criterion(yaml['loss'])
	# optimizer
	optimizer = select_optimizer(
			yaml['optimizer'], model,
			float(yaml['learning_rate']), float(yaml['weight_decay'])
			)
	#scheduler
	first_cycle_steps = len(train_loader) * yaml['epochs'] // yaml['scheduler_step']
	if args.pseudo == 1:
		first_cycle_steps = (len(train_loader) + len(pseudo_loader)) \
											* yaml['epochs'] // yaml['scheduler_step']
	scheduler = CosineAnnealingWarmupRestarts(
			optimizer, first_cycle_steps=first_cycle_steps, 
			cycle_mult=1.0, max_lr=float(yaml['max_lr']), min_lr=float(yaml['min_lr']),
			warmup_steps=int(first_cycle_steps * 0.25), gamma=0.5
			)
	# train
	train(
		yaml, model, train_loader, val_loader,
		criterion, optimizer, scheduler, device, args.pseudo, pseudo_loader
		)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-y', '--yaml', type=str, default='default.yaml', help='(default: default.yaml)')
	parser.add_argument('-p', '--pseudo', type=int, default=0, help='(default: 0)')
	parser.add_argument('-csv', '--ps_csv', type=str, default='/opt/ml/submission/pseudo.csv', help='(default: /opt/ml/submission/pseudo.csv)')
	args = parser.parse_args()
	
	with open(args.yaml, errors='ignore') as f:
		yaml = yaml.safe_load(f)

	check_yaml(yaml)
	main(yaml)
