import torch
from torch.utils.data import DataLoader

from src.dataset import *
from src.transforms import *
from src.make import make_augmix


# train & val dataloader
def tv_dataloader(dataset, data_dir, batch_size, collate_fn, pseudo_csv):
	if dataset == 'augmix':
		np_load_old = np.load
		np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
		if not os.path.exists('augmix.npy'):
			t_loader, v_loader, p_loader = tv_dataloader(
					'custom', data_dir, batch_size, collate_fn, pseudo_csv=pseudo_csv
					)
			make_augmix(t_loader)
		augmix_data=np.load('augmix.npy')
	else:
		augmix_data=None

	train_dataset, val_dataset = select_dataset(dataset, data_dir, augmix_data)
	pseudo_dataset = PseudoDataset(data_dir=data_dir, pseudo_csv=pseudo_csv, transform=tf)

	train_loader = DataLoader(
			dataset=train_dataset,batch_size=batch_size,
			shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True
			)
	val_loader = DataLoader(
			dataset=val_dataset, batch_size=batch_size,
			shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True
			)
	pseudo_csv
	pseudo_loader = DataLoader(
			dataset=pseudo_dataset, batch_size=batch_size,
			shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True
			)
	print('Finish train & val & pseudo dataloaer!')

	return train_loader, val_loader, pseudo_loader


# test dataloader
def test_dataloader(data_dir, batch_size, collate_fn):	
	test_dataset = select_dataset('test', data_dir)
	test_loader = DataLoader(
			dataset=test_dataset, batch_size=batch_size,
			shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True
			)
	print('Finish test dataloader!')

	return test_loader
	
