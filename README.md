# Vehicle and Object Detection with YOLO

A Python-based repository for object detection and vehicle classification using YOLO (Ultralytics), featuring GPU acceleration via CUDA 12.1, real-time webcam inference, and custom training capabilities.

## Features

- **Real-time Detection**: Webcam stream inference with live visual output.
- **Image Inference**: Single-image prediction.
- **Custom Training**: Steps and scripts configured for custom dataset training (e.g., vehicles dataset).
- **GPU Acceleration**: Pre-configured support for PyTorch with CUDA 12.1.

---

## Prerequisites

Before setting up the project, ensure you have:

1. **Python**: Version `>=3.11` installed.
2. **Poetry**: Modern Python package manager. If you do not have it, install it using:
   - **Windows (PowerShell)**:

     ```powershell
     (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
     ```

   - **macOS/Linux**:

     ```bash
     curl -sSL https://install.python-poetry.org | python3 -
     ```

3. **NVIDIA GPU (Optional but Recommended)**:
   - Ensure your NVIDIA GPU drivers are up to date.
   - Verify CUDA capability (this project is configured for CUDA 12.1).

---

## Installation & Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Nhatthach2703/test-yolo26.git
   cd test-yolo26
   ```

2. **Install Dependencies with Poetry**
   This project uses a dedicated PyTorch source index for CUDA 12.1 compatibility. Install all dependencies using:

   ```bash
   poetry install
   ```

   *(Note: This automatically creates a virtual environment and installs PyTorch with GPU support).*

3. **Verify GPU Availability**
   You can verify if PyTorch detects your NVIDIA GPU by running:

   ```bash
   poetry run python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

4. **Pre-trained Weights**
   Ensure your pre-trained YOLO weights (e.g., `yolo26n.pt`) are downloaded and placed in the `models/` directory (or at the root directory if specified in the scripts).

---

## Running Inference

Ensure you run all commands prefixed with `poetry run` to use the correct virtual environment.

### 1. Image Detection

To run object detection on a single image:

```bash
poetry run python src/detect_image.py
```

*Note: You can modify `src/detect_image.py` to change the target image path or the model weights.*

### 2. Webcam Detection (Real-time)

To run real-time detection using your system webcam:

```bash
poetry run python src/detect_webcam.py
```

*Note: Press `ESC` (Escape key) in the webcam window to exit.*

---

## Model Training & Custom Dataset Pipeline

Training a custom YOLO model involves transfer learning: taking a pre-trained model (like `yolo26n.pt`) and fine-tuning its weights on your own domain-specific dataset (in this case, vehicle detection).

This guide details the complete end-to-end pipeline from preparing labels to monitoring training.

---

### 1. Dataset Annotation & Preparation

Your custom dataset must be annotated in the **YOLO format**:

- Each image must have a corresponding `.txt` label file with the same name (e.g., `image1.jpg` and `image1.txt`).
- The coordinates in label files must be normalized between `0` and `1`.
- The format of each line in the `.txt` file is:

  ```text
  <class_id> <x_center> <y_center> <width> <height>
  ```

  *(Example: `0 0.521 0.312 0.124 0.245` representing a class at index 0).*

#### Annotation Tools

You can use the following popular tools to label your data:

- [Roboflow](https://roboflow.com/) (Recommended: easily exports directly in YOLO format).
- [Label Studio](https://labelstud.io/) (Open-source multi-purpose tool).
- [CVAT](https://www.cvat.ai/) (Computer Vision Annotation Tool).

---

### 2. Directory Structure

Organize your dataset files into `train`, `valid` (validation), and `test` splits inside the `datasets/vehicles/` directory:

```text
datasets/
└── vehicles/
    ├── data.yaml          # Dataset configuration file
    ├── train/
    │   ├── images/        # Training images (e.g., img1.jpg)
    │   └── labels/        # Training annotations (e.g., img1.txt)
    ├── valid/
    │   ├── images/        # Validation images
    │   └── labels/        # Validation annotations
    └── test/
        ├── images/        # Test images
        └── labels/        # Test annotations
```

---

### 3. Dataset Configuration (`data.yaml`)

The `data.yaml` file links the model to your data files and maps class IDs to labels. Verify yours looks like this:

```yaml
# Paths can be relative to the directory containing this data.yaml
train: ../train/images
val: ../valid/images
test: ../test/images

# Number of classes
nc: 5

# Class names mapping (matches <class_id> 0 to 4)
names: ['Bus', 'Car', 'Motorcycle', 'Pickup', 'Truck']
```

---

### 4. Training Execution & Parameters

The training process is defined in `src/train/vehicles_dataset.py`. It initializes a base YOLO model and trains it using custom parameters.

#### Key Hyperparameters (in `vehicles_dataset.py`)

- `data`: Path to `data.yaml`.
- `epochs`: Number of iterations over the whole dataset (currently set to `100`).
- `imgsz`: Input image size (typically `640` pixels). Large sizes improve detail detection but require more VRAM.
- `batch`: Batch size (`16`). Reduce if you get GPU "Out Of Memory" (OOM) errors.
- `workers`: Number of CPU threads for loading data (`2`).
- `device`: Compute hardware (use `0` for NVIDIA GPU, or `"cpu"`).

#### Run the Training Script

```bash
poetry run python src/train/vehicles_dataset.py
```

---

### 5. Monitoring & Outputs

Once training starts, YOLO automatically outputs progress logs to the console and generates visualization graphs under:

```text
runs/detect/train/
```

#### Key Output Files to Check

- `weights/best.pt`: The model weights that achieved the highest accuracy/mAP on validation data. **Use this for deployment.**
- `weights/last.pt`: The weights from the very last epoch.
- `results.png` & `results.csv`: Progress plots showing loss metrics (Box, Cls, DFL) and evaluation metrics (mAP50, mAP50-95) decreasing/increasing over epochs.
- `confusion_matrix.png`: Matrix showing class-by-class classification performance.
- `val_batch0_labels.jpg`: Sample visualization of validation ground truths and predictions.

---

### 6. Evaluating / Testing Trained Model

Test your newly trained model (`runs/detect/train/weights/best.pt`) on custom video files using the following pre-built test scripts.

> [!NOTE]
> These scripts are provided as templates. You must edit them (e.g., path to your weights `best.pt`, source video paths like `images/test-2.mp4`, confidence thresholds, or device settings) to match your custom setup.

#### Option A (Simple prediction with default Ultralytics viewer)

Ideal for standard video inference and auto-saving output results.

```bash
poetry run python src/train/test_vehicles_1.py
```

#### Option B (OpenCV viewer with custom controls)

Ideal if you need to adjust window sizes and exit on demand by pressing `q`.

```bash
poetry run python src/train/test_vehicles_2.py
```
