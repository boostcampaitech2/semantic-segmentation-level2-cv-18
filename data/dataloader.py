import os

from torch.utils.data import DataLoader

from .dataset import CustomDataSet
from .augmentations import get_transforms, get_train_transform, get_val_transform, get_test_transform


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def get_train_dataset(cfg, category_names):
    train_trans = get_train_transform(cfg)
    
    dataset_path = cfg["DATASET"]["PATH"]
    train_path = os.path.join(dataset_path, cfg["DATASET"]["TRAIN_FILE_NAME"])
    
    train_dataset = CustomDataSet(
        data_dir=train_path,
        mode="train",
        cfg=cfg,
        transform=train_trans,
        category_names=category_names,
    )
    
    return train_dataset


def get_val_dataset(cfg, category_names):
    val_trans = get_val_transform(cfg)
    
    dataset_path = cfg["DATASET"]["PATH"]
    val_path = os.path.join(dataset_path, cfg["DATASET"]["VAL_FILE_NAME"])
    
    val_dataset = CustomDataSet(
        data_dir=val_path,
        mode="val",
        cfg=cfg,
        transform=val_trans,
        category_names=category_names,
    )
    
    return val_dataset


def get_test_dataset(cfg, category_names):
    test_trans = get_test_transform(cfg)

    dataset_path = cfg["DATASET"]["PATH"]
    test_path = os.path.join(dataset_path, cfg["DATASET"]["TEST_FILE_NAME"])
    
    test_dataset = CustomDataSet(
        data_dir=test_path,
        mode="test",
        cfg=cfg,
        transform=test_trans,
        category_names=category_names,
    )
    
    return test_dataset


def get_datasets(cfg, category_names):
    """Get train/val/test datasets from configs."""
    # datasets
    train_dataset = get_train_dataset(cfg, category_names)
    val_dataset = get_val_dataset(cfg, category_names)
    test_dataset = get_test_dataset(cfg, category_names)

    return [train_dataset, val_dataset, test_dataset]


def get_train_dataloader(cfg, category_names):
    batch_size = cfg["EXPERIMENTS"]["BATCH_SIZE"]
    num_workers = cfg["EXPERIMENTS"]["NUM_WORKERS"]
    
    train_dataset = get_train_dataset(cfg, category_names)
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return train_dataloader


def get_val_dataloader(cfg, category_names):
    batch_size = cfg["EXPERIMENTS"]["BATCH_SIZE"]
    num_workers = cfg["EXPERIMENTS"]["NUM_WORKERS"]
    
    val_dataset = get_val_dataset(cfg, category_names)
    
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return val_dataloader


def get_test_dataloader(cfg, category_names):
    batch_size = cfg["EXPERIMENTS"]["BATCH_SIZE"]
    num_workers = cfg["EXPERIMENTS"]["NUM_WORKERS"]
    
    test_dataset = get_test_dataset(cfg, category_names)
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return test_dataloader


def get_dataloaders(cfg, category_names):
    """Get train/val/test data loaders from configs."""
    
    # dataloaders
    train_dataloader = get_train_dataloader(cfg, category_names)
    val_dataloader = get_val_dataloader(cfg, category_names)
    test_dataloader = get_test_dataloader(cfg, category_names)

    return [train_dataloader, val_dataloader, test_dataloader]


def get_val_dataset_for_kfold(cfg, category_names):
    """Get val dataset applying train transforms."""
    val_trans = get_val_transform(cfg)

    dataset_path = cfg["DATASET"]["PATH"]
    train_path = os.path.join(dataset_path, cfg["DATASET"]["TRAIN_FILE_NAME"])

    val_dataset = CustomDataSet(
        data_dir=train_path,
        mode="val",
        cfg=cfg,
        transform=val_trans,
        category_names=category_names,
    )
    
    return val_dataset
