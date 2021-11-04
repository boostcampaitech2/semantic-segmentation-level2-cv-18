import argparse, yaml
import torch, os
import warnings
warnings.filterwarnings('ignore')

from src.utils import *
from src.select import *
from src.dataloader import test_dataloader
from src.pipeline import test
from src.make import make_submission


def inference(yaml):
	def collate_fn(batch):
		return tuple(zip(*batch))

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# fix seed
	seed_everything(yaml['seed'])
	# loader
	test_loader = test_dataloader(
			yaml['data_dir'], yaml['batch_size'], collate_fn
			)
	# model
	model = select_model(yaml['model'])
	model = load_model(model, yaml['inference_dir'], yaml['save_name'], device).to(device)
	model = select_tta(model, yaml['is_tta'])
	# test
	file_name_list, preds_array = test(
			yaml['seed'], model, test_loader, yaml['is_crf'], device
			)
	# make submission
	make_submission(yaml['submission_dir'], yaml['save_name'], file_name_list, preds_array)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-y', '--yaml', type=str, default='./configs/default.yaml', help='(default: ./configs/default.yaml)')
	args = parser.parse_args()

	with open(args.yaml, errors='ignore') as f:
		yaml = yaml.safe_load(f)

	inference(yaml)
