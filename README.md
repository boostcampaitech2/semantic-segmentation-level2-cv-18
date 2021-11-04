**:warning: This is a branch created by the [author](https://github.com/lisy0123)'s wants. The regular branch is [here](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-18/tree/main).**

## Start

`bash run.sh` and follow the instructions.

---

## To Do List

- [x] using yaml
- [x] dataset with/without augmix
- [x] inference with dense crf
- [x] inference with tta
- [x] checkpoint according to iou, loss, both
- [x] automatically generate checkpoint folder (like yolov5)
- [x] pseudo on/off
- [x] bash script
- [x] add model, HRNET+OCR
- [ ] add loss, SoftCE
- [ ] change pseudo parser into store_true, unify parsers



|    Name    | Items                                    |
| :--------: | ---------------------------------------- |
|   model    | deeplabv3, deeplabv3+, unet++, hrnet+ocr |
|  dataset   | custom(without augmix), augmix           |
|    loss    | CE, Dice, Focal, IoU, DiceCE, DiceFocal  |
| optimizer  | madgrad, Adam, AdamW                     |
| scheduler  | CosineAnnealingWarmupRestarts            |
| checkpoint | all(iou & loss), iou, loss               |
| inference  | basic, dense crf(on/off), tta(on/off)    |
| load model | (on/off)                                 |
|   pseudo   | (on/off)                                 |



