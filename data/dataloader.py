import os

from torch.utils.data import DataLoader

from .dataset import CustomDataSet
from .augmentations import get_transforms


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def get_datasets(cfg, category_names):
    train_trans, val_trans, test_trans = get_transforms(cfg)
    
    dataset_path = cfg["DATASET"]["PATH"]
    train_path = os.path.join(dataset_path, cfg["DATASET"]["TRAIN_FILE_NAME"])
    val_path = os.path.join(dataset_path, cfg["DATASET"]["VAL_FILE_NAME"])
    test_path = os.path.join(dataset_path, cfg["DATASET"]["TEST_FILE_NAME"])
    
    # datasets
    train_dataset = CustomDataSet(data_dir=train_path, 
                                  mode='train', 
                                  cfg=cfg,
                                  transform=train_trans,
                                  category_names=category_names)
    val_dataset = CustomDataSet(data_dir=val_path, 
                                mode='val',
                                cfg=cfg,
                                transform=val_trans,
                                category_names=category_names)
    test_dataset = CustomDataSet(data_dir=test_path, 
                                 mode='test',
                                 cfg=cfg,
                                 transform=test_trans,
                                 category_names=category_names)

    return [train_dataset, val_dataset, test_dataset]
    
    
def get_dataloaders(cfg, category_names):
    batch_size = cfg["EXPERIMENTS"]["BATCH_SIZE"]
    num_workers = cfg["EXPERIMENTS"]["NUM_WORKERS"]
    
    train_dataset, val_dataset, test_dataset = get_datasets(cfg, category_names)
    
    # dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_dataset, 
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    
    return [train_dataloader, val_dataloader, test_dataloader]
