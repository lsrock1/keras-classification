# Keras Classification

## Installation
  ```bash
    pip install fvcore
    pip install tensorflow-gpu # > 2.1
    pip install autokeras
  ```
## Configuration
All possible options are in [classification/configs/defaults.py](classification/configs/defaults.py)
  ```bash
  # In configs/baseline.yaml
    MODEL:
      NAME: 'resnet50'
    TRAIN_DIR: ('data/train', 'data/train_0513')
    BATCH_SIZE: 48
    OUTPUT_DIR: 'baseline'
  ```

## Train

```bash
python train.py --config-file configs/baseline.yaml
```

## Dataset
```bash
data
 ├── cat
 │    ├─ 1.jpg
 │    ├─ 2.jpg
 │    ├─ ...
 │    └─ 999.jpg
 └── dog
      ├─ 1.jpg
      ├─ 2.jpg
      ├─ ...
      └─ 999.jpg
```
