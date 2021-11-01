import os
from pprint import pprint
from tqdm import tqdm
import warnings 
warnings.filterwarnings('ignore')

import numpy as np

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torchvision
import segmentation_models_pytorch as smp

# criterion
from pytorch_toolbelt import losses as L
# optimizer
from madgrad import MADGRAD
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# Fold를 위한 라이브러리
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import GroupKFold, KFold

import wandb


from util.ploting import (
    plot_examples, plot_train_dist
)
from util.utils import (
    label_accuracy_score, add_hist
)
from util.eda import (
    eda, get_df_train_categories_counts, add_bg_index_to
)

from data.dataloader import (
    get_dataloaders, get_datasets, collate_fn, get_val_dataset_for_kfold
)

from config.read_config import (
    print_ver_n_settings, get_args, get_cfg_from
)
from config.fix_seed import (
    fix_seed_as
)
from config.wnb import (
    wnb_init
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def is_same_width(t1, t2):
    return t1.shape[-2] == t2.size(-2)

def is_same_height(t1, t2):
    return t1.shape[-1] == t2.size(-1)

    
def get_model_inference(cfg, model, images):
    """"""
    frame_selected = cfg["SELECTED"]["FRAMEWORK"]
    # inference
    if frame_selected == "torchvision": 
        outputs = model(images)['out']
    elif frame_selected == "segmentation_models_pytorch": 
        outputs = model(images)
    
    return outputs
    

def simple_check(cfg, model):
    """구현된 model에 임의의 input을 넣어 output이 잘 나오는지 test"""
    x = torch.randn([2, 3, 512, 512])
    
    if cfg["SELECTED"]["FRAMEWORK"] == "torchvision":
        out = model(x)['out']
    elif cfg["SELECTED"]["FRAMEWORK"] == "segmentation_models_pytorch":
        out = model(x)
    
    assert(is_same_width(x, out) and is_same_height(x, out))
    assert(out.size(-3) == cfg["DATASET"]["NUM_CLASSES"])
    
    
def set_torchvision_model(cfg, model):
    num_classes = cfg["DATASET"]["NUM_CLASSES"]
    model_selected = cfg["SELECTED"]["MODEL"]
    
    if model_selected == "lraspp_mobilenet_v3_large":
        model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif "deeplab" in model_selected:
        model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif "fcn" in model_selected:
        model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    return model


###################################
## set_torchvision_model 함수처럼 짤 수 있으면 좋을 것 같습니다.
def set_smp_model(cfg, model):
    return model
###################################


def get_trainable_model(cfg):
    cfg_selected = cfg["SELECTED"]
    frame_selected = cfg_selected["FRAMEWORK"]
    num_classes = cfg["DATASET"]["NUM_CLASSES"]
    
    if frame_selected == "torchvision":
        model_selected = cfg_selected["MODEL"]
        Model = getattr(torchvision.models.segmentation, model_selected)
        model = Model(**cfg_selected["MODEL_CFG"])
        model = set_torchvision_model(cfg, model)
    elif frame_selected == "segmentation_models_pytorch":
        model = smp.create_model(**cfg_selected["MODEL_CFG"])
        model = set_smp_model(cfg, model)
    
    simple_check(cfg, model)
    
    return model


def save_model(model, saved_dir, file_name):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)
    
    
    
def calc_loss(cfg, model, images, masks, criterion, device):
    frame_selected = cfg["SELECTED"]["FRAMEWORK"]
    if cfg["EXPERIMENTS"]["AUTOCAST_TURN_ON"]:
        with autocast(enabled=True):
            # device 할당
            model = model.to(device)
            # inference
            outputs = get_model_inference(cfg, model, images)
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
    else:
        # device 할당
        model = model.to(device)
        # inference
        outputs = get_model_inference(cfg, model, images)
        # loss 계산 (cross entropy loss)
        loss = criterion(outputs, masks)
    
    return [model, outputs, loss]
    
    
def get_model_file_name(cfg, fold:int=None):
    saved_model_file = ""
    seleceted_framework = cfg["SELECTED"]["FRAMEWORK"]
    
    if seleceted_framework == "torchvision":
        saved_model_file = f"{cfg['SELECTED']['MODEL']}"
    elif seleceted_framework == "segmentation_models_pytorch":
        arch_name = cfg['SELECTED']['MODEL_CFG']['arch']
        enc_name = cfg['SELECTED']['MODEL_CFG']['encoder_name']
        enc_weights_name = cfg['SELECTED']['MODEL_CFG']['encoder_weights']
        saved_model_file = f"{arch_name}_{enc_name}_{enc_weights_name}"
        
    if fold is not None:
        saved_model_file += f"_{fold+1}"
    
    saved_model_file += ".pt"
    return saved_model_file
   

def get_scaler():
    return GradScaler()
    
    
def get_criterion():
    selected_criterion_framework = cfg["SELECTED"]["CRITERION"]["FRAMEWORK"]
    selected_criterion = cfg["SELECTED"]["CRITERION"]["USE"]
    selected_criterion_cfg = cfg["SELECTED"]["CRITERION"]["CFG"]
    
    if selected_criterion_framework == "torch.nn":
        Creterion = getattr(nn, selected_criterion)
    elif selected_criterion_framework == "pytorch_toolbelt":
        Creterion = getattr(L, selected_criterion)
    
    assert(Creterion is not None)
    creterion = Creterion() if selected_criterion_cfg is None else Creterion(**selected_criterion_cfg)
    return creterion


def get_optim(cfg, model):
    return MADGRAD(params = model.parameters(), 
                   lr=cfg["EXPERIMENTS"]["LEARNING_RATE"], 
                   weight_decay=1e-6)


def get_scheduler(cfg, optimizer):
    return CosineAnnealingWarmRestarts(optimizer, 
                                       T_0=cfg["EXPERIMENTS"]["NUM_EPOCHS"], 
                                       T_mult=1)
    
    
def train_one(num_epochs, 
              model, 
              train_dataloader, 
              val_dataloader, 
              criterion, 
              optimizer, 
              scheduler,
              scaler,
              saved_dir, 
              val_every, 
              device, 
              category_names, 
              cfg, 
              fold:int=None):
    
    print(f'Start training ...')
    n_class = 11
    best_mIoU = 0.0
    
    # WandB watch model.
    if cfg["EXPERIMENTS"]["WNB"]["TURN_ON"]:
        wandb.watch(model, log=all)
    
    for epoch in range(num_epochs):
        model.train()

        train_avg_loss, train_avg_mIoU = 0.0, 0.0

        hist = np.zeros((n_class, n_class))
        pbar_train = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, (images, masks, _) in pbar_train:
            images, masks = torch.stack(images), torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            model, outputs, loss = calc_loss(cfg, model, images, masks, criterion, device)
            
            train_avg_loss += loss.item() / len(masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            train_avg_mIoU += mIoU
            
            description_train = f"# epoch : {epoch + 1}  Loss: {round(loss.item(),4)}   mIoU: {round(mIoU,4)}"
            pbar_train.set_description(description_train)
            
        scheduler.step()
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            mIoU = validation(epoch + 1, model, val_dataloader, criterion, device, category_names, cfg)
            if mIoU > best_mIoU:
                print(f"Best performance at epoch: {epoch + 1}, Best mIoU: {round(best_mIoU, 4)} --> {round(mIoU, 4)}")
                print(f"Save model in {saved_dir}")
                best_mIoU = mIoU
                
                save_model(model, saved_dir, file_name=get_model_file_name(cfg, fold))
                print()
        
        if cfg["EXPERIMENTS"]["WNB"]["TURN_ON"]:
            train_avg_loss /= len(train_dataloader)
            train_avg_mIoU /= len(train_dataloader)
            wandb.log({'Train/Epoch':epoch+1, 
                       'Train/Avg Loss':train_avg_loss, 
                       'Train/Avg mIoU':train_avg_mIoU})
            plot_examples(model=model,
                          cfg=cfg,
                          device=device,
                          mode="train", 
                          batch_id=0, 
                          num_examples=cfg["EXPERIMENTS"]["BATCH_SIZE"],
                          dataloader=train_dataloader)
            
    print("End of train\n")
            

def validation(epoch, 
               model, 
               val_dataloader, 
               criterion, 
               device, 
               category_names, 
               cfg, 
               fold:int=None):
    
    print(f'Start validation ...')
    
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        pbar_val = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for step, (images, masks, _) in pbar_val:
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model, outputs, loss = calc_loss(cfg, model, images, masks, criterion, device)
            
            total_loss += loss
            cnt += 1
            avrg_loss = total_loss / cnt
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, _, mIoU, _, _ = label_accuracy_score(hist)
            
            description_val = f"# epoch : {epoch} Avg Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, mIoU: {round(mIoU, 4)}"        
            pbar_val.set_description(description_val)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
        avrg_loss = total_loss / cnt
        
        print(f'IoU by class : {IoU_by_class}')
        
        if cfg["EXPERIMENTS"]["WNB"]["TURN_ON"]:
            dict_IoU_by_class = {classes : round(IoU,4) for IoU, classes in zip(IoU , category_names)}
            wandb.log({'Val/Avg Loss': round(avrg_loss.item(), 4), 
                       'Val/Acc': round(acc, 4),
                       'Val/mIoU': round(mIoU, 4),})
            wandb.log({f"Valid/{class_name}": IoU 
                       for class_name, IoU in dict_IoU_by_class.items()})
            plot_examples(model=model,
                          cfg=cfg,
                          device=device,
                          mode="val", 
                          batch_id=0, 
                          num_examples=cfg["EXPERIMENTS"]["BATCH_SIZE"],
                          dataloader=val_dataloader)

    return mIoU


def train_kfold(num_epochs, 
                model, 
                train_dataloader, 
                val_dataloader, 
                criterion, 
                optimizer, 
                scheduler, 
                scaler,
                saved_dir, 
                val_every, 
                device, 
                category_names, 
                cfg):
    kf = KFold(cfg["EXPERIMENTS"]["KFOLD"]["NUM_FOLD"], shuffle=True)
    
    train_dataset, _, _ = get_datasets(cfg, category_names)
    val_dataset = get_val_dataset_for_kfold(cfg, category_names)
    batch_size = cfg["EXPERIMENTS"]["BATCH_SIZE"]
    num_workers = cfg["EXPERIMENTS"]["NUM_WORKERS"]
    
    for fold, (train_ids, val_ids) in enumerate(kf.split(train_dataset)):
    
        print(f'FOLD - {fold+1}')
        print('---------------------------------------------------------')

        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            sampler=train_subsampler,
            persistent_workers=True
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            sampler=val_subsampler,
            persistent_workers=True
        )
        
        train_one(num_epochs=cfg["EXPERIMENTS"]["NUM_EPOCHS"], 
                  model=model, 
                  train_dataloader=train_dataloader, 
                  val_dataloader=val_dataloader, 
                  criterion=criterion, 
                  optimizer=optimizer, 
                  scheduler=scheduler,
                  scaler=scaler,
                  saved_dir=cfg["EXPERIMENTS"]["SAVED_DIR"]["BEST_MODEL"], 
                  val_every=cfg["EXPERIMENTS"]["VAL_EVERY"], 
                  device=device,
                  category_names=category_names,
                  cfg=cfg,
                  fold=fold)


def train(cfg, model, train_dataloader, val_dataloader, category_names, device):
    scaler = get_scaler()
    criterion = get_criterion()
    optimizer = get_optim(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    
    if not cfg["EXPERIMENTS"]["KFOLD"]["TURN_ON"]:    
        train_one(num_epochs=cfg["EXPERIMENTS"]["NUM_EPOCHS"], 
                  model=model, 
                  train_dataloader=train_dataloader, 
                  val_dataloader=val_dataloader, 
                  criterion=criterion, 
                  optimizer=optimizer, 
                  scheduler=scheduler,
                  scaler=scaler,
                  saved_dir=cfg["EXPERIMENTS"]["SAVED_DIR"]["BEST_MODEL"], 
                  val_every=cfg["EXPERIMENTS"]["VAL_EVERY"], 
                  device=device,
                  category_names=category_names,
                  cfg=cfg)
    else:
        train_kfold(num_epochs=cfg["EXPERIMENTS"]["NUM_EPOCHS"], 
                    model=model, 
                    train_dataloader=train_dataloader, 
                    val_dataloader=val_dataloader, 
                    criterion=criterion, 
                    optimizer=optimizer, 
                    scheduler=scheduler,
                    scaler=scaler,
                    saved_dir=cfg["EXPERIMENTS"]["SAVED_DIR"]["BEST_MODEL"], 
                    val_every=cfg["EXPERIMENTS"]["VAL_EVERY"], 
                    device=device,
                    category_names=category_names,
                    cfg=cfg)
        
    
def main():
    cfg = get_cfg_from(get_args())
    fix_seed_as(cfg["SEED"])
    
    wnb_run = wnb_init(cfg)
    
    print_ver_n_settings()
    eda(cfg)
    pprint(cfg)
    
    df_train_categories_counts = get_df_train_categories_counts(cfg)
    plot_train_dist(cfg, df_train_categories_counts)
    sorted_df_train_categories_counts = add_bg_index_to(df_train_categories_counts)
    category_names = sorted_df_train_categories_counts["Categories"].to_list()
    
    model = get_trainable_model(cfg)
    train_dataloader, val_dataloader, _ = get_dataloaders(cfg, category_names)
    
    train(cfg, model, train_dataloader, val_dataloader, category_names, device=DEVICE)
    
    if wnb_run is not None:
        wnb_run.finish()
        
        
if __name__ == "__main__":
    main()
    
