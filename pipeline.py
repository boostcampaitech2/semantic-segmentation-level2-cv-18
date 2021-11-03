import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn


from pprint import pprint

# IMPORT CUSTOM MODULES
from util.ploting import plot_train_dist
from util.eda import get_df_train_categories_counts, add_bg_index_to, eda

from data.dataloader import get_dataloaders

from train import train, get_trainable_model

from inference import create_submission, get_trained_model

from config.read_config import print_ver_n_settings, get_cfg_from, get_args
from config.fix_seed import fix_seed_as
from config.wnb import wnb_init

# GPU 사용 가능 여부에 따라 device 정보 저장
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    cfg = get_cfg_from(get_args())
    fix_seed_as(cfg["SEED"])
    
    # wandb 시작
    wnb_run = wnb_init(cfg)

    print_ver_n_settings()
    eda(cfg)
    pprint(cfg)

    # 데이터프레임 및 시각화 함수
    df_train_categories_counts = get_df_train_categories_counts(cfg)
    plot_train_dist(cfg, df_train_categories_counts)
    sorted_df_train_categories_counts = add_bg_index_to(df_train_categories_counts)
    category_names = sorted_df_train_categories_counts["Categories"].to_list()

    # 모델 및 데이터로더 불러오기
    model = get_trainable_model(cfg)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        cfg, category_names
    )

    # 모델 훈련
    train(cfg, model, train_dataloader, val_dataloader, category_names, device=DEVICE)

    # CSV 파일 생성
    create_submission(test_dataloader=test_dataloader, device=DEVICE, cfg=cfg)

    # wandb 사용 시 종료
    if wnb_run is not None:
        wnb_run.finish()
