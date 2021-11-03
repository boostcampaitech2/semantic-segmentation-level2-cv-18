import os
import json

import pandas as pd
import numpy as np


def get_df_train_categories_counts(cfg):
    dataset_path = cfg["DATASET"]["PATH"]
    train_file_path = os.path.join(dataset_path, cfg["DATASET"]["TRAIN_FILE_NAME"])

    # Read annotations
    with open(train_file_path, "r") as f:
        dataset = json.loads(f.read())

    df = pd.DataFrame(dataset["annotations"])
    df = df[["id", "category_id"]]
    df = df.groupby(by="category_id", as_index=False).count()
    df["category_id"] = df["category_id"].apply(
        lambda x: dataset["categories"][x - 1]["name"]
    )
    df.columns = ["Categories", "Number of annotations"]
    df = df.sort_values(by="Number of annotations", ascending=False)

    return df


def add_bg_index_to(df):
    # df 에 index(Background) 를 추가한다.
    df_target = pd.DataFrame(["Background"], columns=["Categories"])
    df_target = df_target.append(df)
    df_target = df_target.sort_index()
    df_target.index = range(df_target.shape[0])
    return df_target


# 제공받은 baseline
def eda(cfg):
    dataset_path = cfg["DATASET"]["PATH"]
    anns_file_path = os.path.join(dataset_path, cfg["DATASET"]["ANNS_FILE_NAME"])

    # Read annotations
    with open(anns_file_path, "r") as f:
        dataset = json.loads(f.read())

    categories = dataset["categories"]
    anns = dataset["annotations"]
    imgs = dataset["images"]

    df_categories = pd.DataFrame(categories)

    num_categories = len(df_categories["name"])
    num_super_categories = len(df_categories["supercategory"])
    num_annotations = len(anns)
    num_images = len(imgs)

    print("-" * 30)
    print("Number of super categories:", num_super_categories)
    print("Number of categories:", num_categories)
    print("Number of annotations:", num_annotations)
    print("Number of images:", num_images)
    print("-" * 30)


# MultilabelStratifiedKFold 적용을 위해 annotation과 img를 반환
def get_anns_imgs(cfg):
    dataset_path = cfg["DATASET"]["PATH"]
    anns_file_path = os.path.join(dataset_path, cfg["DATASET"]["ANNS_FILE_NAME"])

    # Read annotations
    with open(anns_file_path, "r") as f:
        dataset = json.loads(f.read())

    categories = dataset["categories"]
    anns = dataset["annotations"]
    imgs = dataset["images"]

    return categories, anns, imgs

