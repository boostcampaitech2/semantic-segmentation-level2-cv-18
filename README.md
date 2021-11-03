# 0. Quick Start (Get our result)

1. Set a virtual environment and requirements for our result *(Recommended)*

   ```bash
   $ conda create -n segmentation python=3.7.11
   $ conda activate segmentation
   $ pip install -r requirements
   ```

2. Execute the following command and get our result.

   ```bash
   $ sh run.sh
   ```

   

# 1. Prepare

## 1.1. Get the trash dataset.

```bash
$ wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000078/data/data.zip
$ mv data.zip rawdata.zip
$ unzip rawdata.zip
$ rm rawdata.zip
```

> **Dataset Copy Right and License:** [*Naver Connect*](https://connect.or.kr/), [*CC-BY-2.0*](https://creativecommons.org/licenses/by/2.0/kr/)   
> For more detail about this dataset, see [*here*](./DATASET_DESCRIPTION.md).

## 1.2. Get Some Libraries.

1. *pytorch-toolbelt* (for various loss)

   ```bash
   $ pip install pytorch_toolbelt
   ```

   > For more detail, see [*here*](https://github.com/BloodAxe/pytorch-toolbelt).

2. *MADGRAD* (An optimizer)

   ```bash
   $ pip install madgrad
   ```

   > For more detail, see [*here*](https://github.com/facebookresearch/madgrad).

## 1.3. Completed Structure

```plain text
./
├─rawdata/
|
├─config/
|    ├─fix_seed.py
|    ├─read_config.py
|    ├─wnb.py
|    └─default.yaml
├─data/
|    ├─dataset.py
|    ├─dataloader.py
|    └─augmentation.py
├─submission/
|    └─sample_submission.csv
├─util/
|    ├─eda.py
|    ├─ploting.py
|    ├─tta.py
|    └─utils.py
|
├─train.py
├─inference.py
├─pipeline.py
|
└─run.sh
```



# 2. Pipeline

1. One Step Execution - From train to inference.

   ```bash
   $ python pipeline.py --cfg-yaml ./config/default.yaml
   ```

2. Two Step Execution

   * Train Step

     ```bash
     $ python train.py --cfg-yaml ./config/default.yaml
     ```

   * Inference Step

     ```bash
     $ python inference.py --cfg-yaml ./config/default.yaml
     ```



# 3. Configurations

## 3.1. Configuration file usage

1. You can see whole configurations in the `yaml` file (`./config/default.yaml`)

2. Copy and paste the `yaml` file and edit it.

3. Then, you can get the proper result.

## 3.2. About configurations

1. Listing the supported frameworks
2. Listing the supported models (including encoders and decoders)
3. Listing the supported criterion
4. **Model Selection**
5. **Experiment configurations**
6. Dataset configurations



# 4. Our Experiments

## 4.1. Model

- Encoder : timm-efficientnet-b7
   - Weight : noisy-student
- Decoder : FPN
   - In channel : 3
   - Classes : 11
- Fold : KFold, MultilabelStratifiedKFold
   - Number of fold : 5
- Learning rate : 0.0001
- TTA : Horizontal flip

## 4.2. Loss

`SoftCrossEntropyLoss` in pytorch_toolbelt

## 4.3. Optimizer

`CosineAnnealingWarmRestarts` in torch.optim.lr_scheduler

## 4.4. Scaler

`Autocast` and `GradScaler` were used to shorten training time. 

## 4.5. Augmentations

By using light model, we perform quickly various augmentation experiments.  

> Selected model : `LRASPP mobilenetv3 Large` in torchvision - for more detail, see [*here*](https://pytorch.org/vision/stable/models.html#semantic-segmentation)

1. Hyper-parameter(`Epochs`) Tuning for the *LRASPP mobilenet v3 large* model.

   |  Epochs   | mIoU  | mIoU derivation |
   | :-------: | :---: | --------------- |
   | 6 epochs  | 0.510 | 0.0             |
   | 12 epochs | 0.553 | **+0.042**      |
   | 24 epochs | 0.571 | **+0.061**      |

   > For fast experiments, we don't try 48 epochs.

2. Single Augmentation Observation

   | Augmentation (Fix 24 epochs) | mIoU  | mIoU derivation |
   | :--------------------------- | :---: | --------------- |
   | None                         | 0.571 | 0.0             |
   | Blur                         | 0.572 | **+0.001**      |
   | GridDistortion               | 0.583 | **+0.012**      |
   | RandomGridShuffle            | 0.585 | **+0.014**      |
   | GridDropout                  | 0.587 | **+0.016**      |
   | ElasticTransform             | 0.598 | **+0.027**      |
   | RandomResizeCrop             | 0.619 | **+0.048**      |

3. A test about compound augmentation by using `albumentations.core.composition.OneOf` (see [*here*](https://albumentations.ai/docs/api_reference/core/composition/#albumentations.core.composition.OneOf))

   * Use *5* augmentations

     * GridDistortion, RandomGridShuffle, GridDropout, ElasticTransform, RandomResizeCrop

   * Result

     | Epoch (Fix augmentation) | mIoU  | mIoU derivation |
     | :----------------------: | :---: | :-------------- |
     |        24 epochs         | 0.609 | **+0.038**      |
     |        48 epochs         | 0.631 | **+0.060**      |
     |        96 epochs         | 0.653 | **+0.082**      |

     > As epoch increase, mIoU also increase. 

## 4.6. K-Fold Ensemble

## 4.7. TTA

We try to use [*ttach* library](https://github.com/qubvel/ttach) but, can't use it. So, we apply only flip TTA, which is satisfied that augmentation is equal to reverse augmentation.

You can add such augmentation function codes in `./util/tta.py` and modify `./config/default.yaml` and `get_tta_list` function in `./util/tta.py`.

## 4.8. Pseudo labeling



# 5. Result

## 5.1. Leader Board in [Competition](https://stages.ai/competitions/78/overview/description)

|            |                mIoU                 |
| :--------: | :---------------------------------: |
| Public LB  | **1st:** `0.781` and **2nd:** `0.1` |
| Private LB |                `0.`                 |

## 5.2. Some images after model inference.



# 6. Participants

| Name           | Github                                        | Role                                                     |
| -------------- | --------------------------------------------- | -------------------------------------------------------- |
| 김서기 (T2035) | [*Link*](https://github.com/seogi98)          | Research(*HRNet*, *MMSeg* library), Pseudo Labeling, TTA |
| 김승훈 (T2042) | [*Link*](https://github.com/lead-me-read-me)  | Find Augmentations, Code Refactoring                     |
| 배민한 (T2260) | [*Link*](https://github.com/Minhan-Bae)       | Research(*smp* library, loss), Model Enhancement, K-Fold |
| 손지아 (T2113) | [*Link*](https://github.com/oikosohn)         | Research(*smp* library, loss), Model Enhancement         |
| 이상은 (T2157) | [*Link*](https://github.com/lisy0123)         | Research(*HRNet*, optimizer, loss), Pseudo Labeling      |
| 조익수 (T2213) | [*Link*](https://github.com/projectcybersyn2) | Research(*MMseg* library)                                |

