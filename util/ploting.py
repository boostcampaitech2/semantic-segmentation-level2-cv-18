import numpy as np
import pandas as pd

import webcolors
import wandb

from matplotlib.patches import Patch
from matplotlib import pyplot as plt
plt.rcParams['axes.grid'] = False

import seaborn as sns; sns.set()

import torch

from .utils import label_to_color_image



def plot_examples(
    model, cfg, device, mode:str=None, batch_id:int=0, num_examples:int=8, dataloaer=None
):
    """Visualization of images and masks according to batch size
    Args:
        mode: train/val/test (str)
        batch_id : 0 (int)
        num_examples : 1 ~ batch_size(e.g. 8) (int)
        dataloaer : data_loader (dataloader)
    Returns:
        None
    """
    class_colormap = pd.read_csv("./class_dict.csv")
    
    # variable for legend
    category_and_rgb = [
        [category, (r, g, b)]
        for idx, (category, r, g, b) in enumerate(class_colormap.values)
    ]
    legend_elements = [
        Patch(
            facecolor=webcolors.rgb_to_hex(rgb),
            edgecolor=webcolors.rgb_to_hex(rgb),
            label=category,
        )
        for category, rgb in category_and_rgb
    ]

    # test / validation set에 대한 시각화
    if mode in ("train", "val"):
        with torch.no_grad():
            for index, (imgs, masks, image_infos) in enumerate(dataloaer):
                if index == batch_id:
                    image_infos = image_infos
                    temp_images = imgs
                    temp_masks = masks

                    model.eval()
                    # inference
                    outs = model(torch.stack(temp_images).to(device))["out"]
                    oms = torch.argmax(outs, dim=1).detach().cpu().numpy()

                    break
                else:
                    continue

        fig, ax = plt.subplots(
            nrows=num_examples,
            ncols=3,
            figsize=(12, 4 * num_examples),
            constrained_layout=True,
        )
        fig.tight_layout()
        for row_num in range(num_examples):
            # Original Image
            ax[row_num][0].imshow(temp_images[row_num].permute([1, 2, 0]))
            ax[row_num][0].set_title(
                f"Orignal Image : {image_infos[row_num]['file_name']}"
            )
            # Groud Truth
            ax[row_num][1].imshow(
                label_to_color_image(masks[row_num].detach().cpu().numpy())
            )
            ax[row_num][1].set_title(
                f"Groud Truth : {image_infos[row_num]['file_name']}"
            )
            # Pred Mask
            ax[row_num][2].imshow(label_to_color_image(oms[row_num]))
            ax[row_num][2].set_title(f"Pred Mask : {image_infos[row_num]['file_name']}")
            ax[row_num][2].legend(
                handles=legend_elements,
                bbox_to_anchor=(1.05, 1),
                loc=2,
                borderaxespad=0,
            )
        if cfg["EXPERIMENTS"]["WNB_TURN_ON"]:
            wandb.log({f"{mode.title()}/viz": wandb.Image(fig)})

    # test set에 대한 시각화
    else:
        with torch.no_grad():
            for index, (imgs, image_infos) in enumerate(dataloaer):
                if index == batch_id:
                    image_infos = image_infos
                    temp_images = imgs

                    model.eval()

                    # inference
                    outs = model(torch.stack(temp_images).to(device))["out"]
                    oms = torch.argmax(outs, dim=1).detach().cpu().numpy()
                    break
                else:
                    continue

        fig, ax = plt.subplots(
            nrows=num_examples,
            ncols=2,
            figsize=(10, 4 * num_examples),
            constrained_layout=True,
        )

        for row_num in range(num_examples):
            # Original Image
            ax[row_num][0].imshow(temp_images[row_num].permute([1, 2, 0]))
            ax[row_num][0].set_title(
                f"Orignal Image : {image_infos[row_num]['file_name']}"
            )
            # Pred Mask
            ax[row_num][1].imshow(label_to_color_image(oms[row_num]))
            ax[row_num][1].set_title(f"Pred Mask : {image_infos[row_num]['file_name']}")
            ax[row_num][1].legend(
                handles=legend_elements,
                bbox_to_anchor=(1.05, 1),
                loc=2,
                borderaxespad=0,
            )
        if cfg["EXPERIMENTS"]["WNB_TURN_ON"]:
            wandb.log({f"{mode.title()}/viz": wandb.Image(fig)})


def plot_train_dist(cfg, df):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title("Category Distribution of train set")
    sns.barplot(x="Number of annotations", y="Categories", data=df, ax=ax, color="skyblue")
    if cfg["EXPERIMENTS"]["WNB_TURN_ON"]:
        wandb.log({"Distribution of train set": wandb.Image(ax)})
        
