# 1. Dataset Introduction

우리는 수많은 쓰레기를 배출하면서 지구의 환경파괴, 야생동물의 생계 위협 등 여러 문제를 겪고 있습니다. 이러한 문제는 쓰레기를 줍는 드론, 쓰레기 배출 방지 비디오 감시, 인간의 쓰레기 분류를 돕는 AR 기술과 같은 여러 기술을 통해서 조금이나마 개선이 가능합니다.

제공되는 이 데이터셋은 위의 기술을 뒷받침하는 쓰레기를 판별하는 모델을 학습할 수 있게 해줍니다.



# 2. Dataset Statistics

1. Image size : (512, 512)

2. Dataset structure

   |             | Images                 | label | file       |
   | ----------- | ---------------------- | ----- | ---------- |
   | Train       | 2,617                  | O     | train.json |
   | Validataion | 655                    | O     | val.json   |
   | Test        | 819 (public + private) | X     | test.json  |

   > Train + Validation = train_all.json

3. Classes

   | Class Name    | Class Number |
   | ------------- | ------------ |
   | Background    | 0            |
   | General trash | 1            |
   | Paper         | 2            |
   | Paper pack    | 3            |
   | Metal         | 4            |
   | Glass         | 5            |
   | Plastic       | 6            |
   | Styrofoam     | 7            |
   | Plastic bag   | 8            |
   | Battery       | 9            |
   | Clothing      | 10           |

4. Warning

   - 참고 : train_all.json/train.json/val.json 에는 background에 대한 annotation이 존재하지 않으므로 background (0) class 추가 

5. Annotation file

   Annotation file은 [coco format](https://cocodataset.org/#home) 으로 이루어져 있습니다.

   [coco format](https://cocodataset.org/#home)은 크게 2가지 (images, annotations)의 정보를 가지고 있습니다.

   - images:
     - id: 파일 안에서 image 고유 id, ex) 1
     - height: 512
     - width: 512
     - file*name: ex) batch*01_vt/002.jpg
   - annotations: (참고 : "bbox", "area"는 Segmentation 경진대회에서 활용하지 않습니다.)
     - id: 파일 안에 annotation 고유 id, ex) 1
     - segmentation: masking 되어 있는 고유의 좌표
     - bbox: 객체가 존재하는 박스의 좌표 (x*min, y*min, w, h)
     - area: 객체가 존재하는 영역의 크기
     - category_id: 객체가 해당하는 class의 id
     - image_id: annotation이 표시된 이미지 고유 id
