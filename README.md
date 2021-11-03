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

2. *MADGRAD*

   ```bash
   $ pip install madgrad
   ```

   > For more detail, see [*here*](https://github.com/facebookresearch/madgrad).

## 1.3. Completed Structure

```plain text
./
├─dataset/
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









