import os
from pprint import pprint
import warnings 
warnings.filterwarnings('ignore')

import numpy as np

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torchvision
import segmentation_models_pytorch as smp

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
    get_dataloaders
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

def simple_check(cfg, model):
    # 구현된 model에 임의의 input을 넣어 output이 잘 나오는지 test
    x = torch.randn([2, 3, 512, 512])
    out = model(x)['out']
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
    model_selected = cfg_selected["MODEL"]
    frame_selected = cfg_selected["FRAMEWORK"]
    num_classes = cfg["DATASET"]["NUM_CLASSES"]
    
    if frame_selected == "torchvision":
        model = getattr(torchvision.models.segmentation, model_selected)(**cfg_selected["MODEL_CFG"])
        model = set_torchvision_model(cfg, model)
    elif cfg_selected["FRAMEWORK"] == "segmentation_models.pytorch":
        model = getattr(smp, model_selected)(**cfg_selected["MODEL_CFG"])
        model = set_smp_model(cfg, model)
    
    simple_check(cfg, model)
    
    return model


def save_model(model, saved_dir, file_name):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)
    
    
def train(num_epochs, model, train_dataloader, val_dataloader, criterion, optimizer, saved_dir, val_every, device, category_names, cfg):
    print(f'Start training..')
    n_class = 11
    best_loss = 9999999
    
    # WandB watch model.
    if cfg["EXPERIMENTS"]["WNB_TURN_ON"]:
        wandb.watch(model, log=all)
    
    for epoch in range(num_epochs):
        model.train()

        train_avg_loss, train_avg_mIoU = 0.0, 0.0

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(train_dataloader):
            images, masks = torch.stack(images), torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            if cfg["EXPERIMENTS"]["AUTOCAST_TURN_ON"]:
                with autocast():
                    # device 할당
                    model = model.to(device)
                    # inference
                    outputs = model(images)['out']
                    # loss 계산 (cross entropy loss)
                    loss = criterion(outputs, masks)
            else:
                # device 할당
                model = model.to(device)
                # inference
                outputs = model(images)['out']
                # loss 계산 (cross entropy loss)
                loss = criterion(outputs, masks)
            
            train_avg_loss += loss.item() / len(masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            train_avg_mIoU += mIoU
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss = validation(epoch + 1, model, val_dataloader, criterion, device, category_names, cfg)
            if avrg_loss < best_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_loss = avrg_loss
                save_model(model, saved_dir, file_name=f"{cfg['SELECTED']['MODEL']}.pt")
            print()
        
        if cfg["EXPERIMENTS"]["WNB_TURN_ON"]:
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
                          num_examples=8, 
                          dataloaer=train_dataloader)
            

def validation(epoch, model, val_dataloader, criterion, device, category_names, cfg):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(val_dataloader):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')
        if cfg["EXPERIMENTS"]["WNB_TURN_ON"]:
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
                          num_examples=8, 
                          dataloaer=val_dataloader)

    return avrg_loss


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
    
    train(num_epochs=cfg["EXPERIMENTS"]["NUM_EPOCHS"], 
          model=model, 
          train_dataloader=train_dataloader, 
          val_dataloader=val_dataloader, 
          criterion=nn.CrossEntropyLoss(), 
          optimizer=torch.optim.Adam(params = model.parameters(), 
                                     lr=cfg["EXPERIMENTS"]["LEARNING_RATE"], 
                                     weight_decay=1e-6), 
          saved_dir=cfg["EXPERIMENTS"]["SAVED_DIR"]["BEST_MODEL"], 
          val_every=cfg["EXPERIMENTS"]["VAL_EVERY"], 
          device=DEVICE,
          category_names=category_names,
          cfg=cfg)
    
    if wnb_run is not None:
        wnb_run.finish()
        
        
if __name__ == "__main__":
    main()
    
