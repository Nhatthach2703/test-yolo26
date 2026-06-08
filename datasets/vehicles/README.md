# Vehicle Detection Image Dataset

## Introduction
The **Vehicle Detection Image Dataset** is a curated dataset specifically designed for object detection and vehicle tracking tasks. It was created and shared on Kaggle by Parisa Karimi Darabi in 2024.

* **Original Dataset Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/pkdarabi/vehicle-detection-image-dataset)
* **Author/Creator:** Parisa Karimi Darabi
* **License:** Attribution 4.0 International (CC BY 4.0)

---

## Classes
The dataset contains **5 object classes** (`nc: 5`) defined in [data.yaml](./data.yaml):
1. **Bus**
2. **Car**
3. **Motorcycle**
4. **Pickup**
5. **Truck**

---

## Folder Structure
The dataset is structured in the standard YOLO format with `train`, `validation` (`valid`), and `test` splits:

```text
vehicles/
├── data.yaml
├── README.md
├── train/
│   ├── images/  # Training images
│   └── labels/  # Training labels (.txt format)
├── valid/
│   ├── images/  # Validation images
│   └── labels/  # Validation labels
└── test/
    ├── images/  # Test images
    └── labels/  # Test labels
```

---

## Preprocessing & Data Augmentation
The original dataset underwent key preprocessing and augmentation steps:
* **Preprocessing:** Resizing and orientation adjustments were applied to all images.
* **Augmentation:** Grayscale variations were introduced to diversify the dataset, enhancing model robustness and generalization during training.

---

## Usage with YOLOv8 / YOLOv11
To train a YOLO model using this dataset, point your model configuration to the [data.yaml](./data.yaml) file:

```python
from ultralytics import YOLO

def train():
    # Load a model
    model = YOLO("models/yolo26n.pt")

    # Train the model
    model.train(
        data="datasets/vehicles/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=8, # Number of worker processes for data loading (Modify to match your CPU/GPU specs)
        device=0  # Uses GPU 0
    )

if __name__ == "__main__":
    train()
```

---

## Citation
If you use this dataset in your work, please cite the author:
```text
Parisa Karimi Darabi. (2024). Vehicle Detection Image Dataset: Suitable for Object Detection and tracking Tasks. 
Retrieved from https://www.kaggle.com/datasets/pkdarabi/vehicle-detection-image-dataset/
```
