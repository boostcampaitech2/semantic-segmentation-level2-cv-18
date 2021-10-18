import torch

from dataset import*
from augmentations import*

def collate_fn(batch):
    return tuple(zip(*batch))

def dataloader(cfg):
    dataset_path  = '../input/data'
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'
    test_path = dataset_path + '/test.json'

    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform)
    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=cfg['batch_size'],
                                            shuffle=True,
                                            num_workers=0,
                                            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=cfg['batch_size'],
                                            shuffle=False,
                                            num_workers=0,
                                            collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=cfg['batch_size'],
                                            num_workers=0,
                                            collate_fn=collate_fn)
    return train_loader, val_loader, test_loader