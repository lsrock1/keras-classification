# Keras Classification

## Installation
  ```bash
    pip install fvcore
    pip install tensorflow-gpu # > 2.1
    pip install autokeras
    pip install tensorflow-model-optimization
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
# baseline training
python train.py --config-file configs/baseline.yaml

# mixed precision training
python train.py --config-file configs/mixed_precision.yaml

# autokeras training
# not supported currently because of generator issue
# python train.py --config-file configs/baseline_auto.yaml
```

## Export

### Find best model in the saved directory and export.

```bash
python export.py --config-file configs/baseline.yaml
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
