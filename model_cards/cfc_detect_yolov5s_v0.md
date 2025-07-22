# Model Card for cfc_detect_yolov5s_v0
**Author**: Madison Van Horn  
**Developed by**: Justin Kay  
**Funded by**: Resnick Sustainability Institute (RSI)


## Model Details
- Training date: 2024-06
- Model type: YOLOv5s
- Framework: PyTorch
- License: MIT
- Model Sources:
  - Repository: [ultralytics/yolov5](https://github.com/ultralytics/yolov5/blob/master/train.py)

## Evaluation

| Metric      | Value   |
|-------------|---------|
| mAP@0.5     | 0.83685 |
| Precision   | 0.86669 |
| Recall      | 0.78457 |


## Training Details
YOLOv5s was trained with a target of 100 epochs using a batch size of 96 and image size of 896. We set the maximum to 100 epochs due to YOLO’s learning rate scheduling strategy, but terminated training after 10 epochs when the model reached its best checkpoint.

### Data
Training data: Kenai, Elwha, Rightbank, Channel, Nushagak, Eel
- [YOLOv5 file lists](https://drive.google.com/file/d/10mXemrZWu4GoJlkFZBj02tfeEw4j2wqe/view?usp=drive_link)

Test data: Kenai, Elwha, Rightbank, Channel, Nushagak, Eel
- [YOLOv5 file lists](https://drive.google.com/file/d/1J_DR6Q0C5xJzGOCbHxdgET41IWp1D-Is/view?usp=drive_link)


### Compute Resources
- Hardware: A100 or V100 on MIT cluster
- Hours: ~4 hours

### Hyperparameters
* weights: yolov5s.pt
* lr0: 0.01
* lrf: 0.01
* momentum: 0.937
* weight_decay: 0.0005
* warmup_epochs: 3.0
* warmup_momentum: 0.8
* warmup_bias_lr: 0.1
* box: 0.05
* cls: 0.5
* cls_pw: 1.0
* obj: 1.0
* obj_pw: 1.0
* iou_t: 0.2
* anchor_t: 4.0
* fl_gamma: 0.0
* hsv_h: 0.015
* hsv_s: 0.7
* hsv_v: 0.4
* degrees: 0.0
* translate: 0.1
* scale: 0.5
* shear: 0.0
* perspective: 0.0
* flipud: 0.0
* fliplr: 0.5
* mosaic: 1.0
* mixup: 0.0
* copy_paste: 0.0
* epochs: 100 
* batch_size: 96
* imgsz: 896
* rect: false
* resume: false
* nosave: false
* noval: false
* noautoanchor: false
* noplots: false
* evolve: null
* evolve_population: ../yolov5/data/hyps
* resume_evolve: null
* cache: null
* image_weights: false
* device: ''
* multi_scale: false
* single_cls: false
* optimizer: SGD
* sync_bn: false
* workers: 8
* name: cfc_all_v5s
* exist_ok: false
* quad: false
* cos_lr: false
* label_smoothing: 0.0
* patience: 100
* freeze:- 0
* save_period: -1
* seed: 0
* local_rank: -1
* entity: null
* upload_dataset: false
* bbox_interval: -1
* artifact_alias: latest
* ndjson_console: false
* ndjson_file: false

## Limitations
May struggle with:
- Far-ranged fish
- Overlapping fish
- Generalization to new river locations

