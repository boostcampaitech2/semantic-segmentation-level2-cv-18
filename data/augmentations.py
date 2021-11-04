import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(cfg):
    cfg_train_trans = cfg["EXPERIMENTS"]["TRAIN_TRANS"]
    
    train_transform = (
        A.Compose(
            [
                A.Resize(height=512, width=512),
                A.OneOf(
                    [
                        getattr(A, trans)(**config)
                        for trans, config in cfg_train_trans.items()
                    ],
                    p=1.0,
                ),
                A.RandomRotate90(p=1.0),
                ToTensorV2(),
            ]
        )
        if cfg_train_trans is not None
        else A.Compose([ToTensorV2()])
    )
    
    return train_transform


def get_val_transform(cfg):
    cfg_test_n_val_trans = cfg["EXPERIMENTS"]["TEST_n_VAL_TRANS"]

    val_transform = (
        A.Compose(
            [
                A.OneOf(
                    [
                        getattr(A, trans)(**config)
                        for trans, config in cfg_test_n_val_trans.items()
                    ]
                ),
                ToTensorV2(),
            ]
        )
        if cfg_test_n_val_trans is not None
        else A.Compose([ToTensorV2()])
    )
    
    return val_transform
    
    
def get_test_transform(cfg):
    cfg_test_n_val_trans = cfg["EXPERIMENTS"]["TEST_n_VAL_TRANS"]

    test_transform = (
        A.Compose(
            [
                A.OneOf(
                    [
                        getattr(A, trans)(**config)
                        for trans, config in cfg_test_n_val_trans.items()
                    ]
                ),
                ToTensorV2(),
            ]
        )
        if cfg_test_n_val_trans is not None
        else A.Compose([ToTensorV2()])
    )
    
    return test_transform
    

def get_transforms(cfg):
    """Get train/val/test transforms."""
    train_trans = get_train_transform(cfg)
    val_trans = get_val_transform(cfg)
    test_trans = get_test_transform(cfg)

    return [train_trans, val_trans, test_trans]
