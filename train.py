import numpy as np

import torch

import wandb

from util.ploting import plot_examples
from util.utils import label_accuracy_score, add_hist


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
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
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

