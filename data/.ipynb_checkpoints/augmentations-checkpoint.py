import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(cfg):
    cfg_train_trans = cfg["EXPERIMENTS"]["TRAIN_TRANS"]
    cfg_test_n_val_trans = cfg["EXPERIMENTS"]["TEST_n_VAL_TRANS"]
    
    train_trans = A.Compose([
        A.OneOf([getattr(A, trans)(**config) for trans, config in cfg_train_trans.items()]),
        ToTensorV2()
    ]) if cfg_train_trans is not None else A.Compose([ToTensorV2()])
    
    val_trans, test_trans = [A.Compose([
        A.OneOf([getattr(A, trans)(**config) for trans, config in cfg_test_n_val_trans.items()]),
        ToTensorV2()
    ])] * 2 if cfg_test_n_val_trans is not None else [A.Compose([ToTensorV2()])] * 2
    
    return [train_trans, val_trans, test_trans]