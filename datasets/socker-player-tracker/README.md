# Soccer Player & Ball Tracking Dataset

## Introduction
The **Soccer Player & Ball Tracking Dataset** is a high-quality object detection dataset designed for tracking soccer players, referees, and the soccer ball. It is based on pseudo-labeled predictions from the DFL Bundesliga Data Shootout dataset.

* **Original Dataset Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/enddl22/bounding-boxes-dflbundesliga-data-shootout)
* **Roboflow Source:** [Roboflow Project](https://app.roboflow.com/inkyu-sa-e0c78/dfl-bundesliga-soccer-football-gpqdy/1)
* **Author/Creator:** enddl22
* **License:** Apache 2.0

---

## Classes & Distribution
The dataset contains **3 object classes** (`nc: 3`) defined in [data.yaml](./data.yaml). The classes are highly imbalanced, which is typical for soccer matches:

1. **Ball**: 41,436 annotations (highly challenging due to small size)
2. **Player**: 490,259 annotations
3. **Ref** (Referee): 53,189 annotations

---

## Folder Structure
The dataset is organized in standard YOLO format:

```text
socker-player-tracker/
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

## Dataset Origin & Pseudo-Labeling Setup
The bounding boxes in this dataset were generated using a two-stage pseudo-labeling pipeline using **YOLO11x**:

1. **Pretraining**: Trained on the Roboflow Soccer dataset (~6k samples) for general domain adaptation.
2. **Finetuning**: Finetuned on the Football Player dataset (663 samples) to match the target Bundesliga distribution.
3. **Pseudo-Labeling**: Inference was run on 460 short videos (460p resolution) from the DFL Bundesliga competition. Annotations were saved in YOLO format only if all four classes (`[ball, player, goalkeeper, referee]`) were present in the frame.

### Inference Parameters
* **Confidence Threshold (`conf`)**: 0.25
* **IoU Threshold (`iou`)**: 0.50
* **Inference Image Size (`imgsz`)**: 2560 (large resolution chosen to mitigate small object detection issues with the soccer ball)

---

## Usage with YOLOv8 / YOLOv11
To train a model on this dataset, reference the local [data.yaml](./data.yaml) config. 

> [!TIP]
> Because the soccer ball is a very small object in 460p/720p frames, training or performing inference with larger image resolutions (e.g., `imgsz=960` or `imgsz=2560`) is highly recommended to improve detection accuracy.

```python
from ultralytics import YOLO

def train():
    # Load model
    model = YOLO("models/yolo11x.pt")

    # Train model
    model.train(
        data="datasets/socker-player-tracker/data.yaml",
        epochs=150,
        imgsz=960,  # Recommended larger resolution
        batch=16,
        workers=8,
        device=0
    )

    # For best results on ball detection, consider test-time augmentation (TTA)
    # or evaluating at imgsz=2560.

if __name__ == "__main__":
    train()
```

---

## Citation
If you use this dataset in your work, please cite the author:
```text
enddl22. (2025). 20k bounding boxes, DFLBundesliga Data Shootout. 
Retrieved from https://www.kaggle.com/datasets/enddl22/bounding-boxes-dflbundesliga-data-shootout
```
