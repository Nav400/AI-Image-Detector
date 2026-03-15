# Traffic Sign Classifier

A convolutional neural network (CNN) built with TensorFlow that classifies road signs from photographs. Trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, the model learns to identify 43 different categories of traffic signs from labeled image data.

---

## Background

One of the central challenges in developing self-driving vehicles is computer vision — specifically, the ability to recognize and distinguish road signs in real time. This project tackles that problem by training a neural network on thousands of labeled traffic sign images, enabling it to correctly classify new, unseen signs with high accuracy.

The dataset used is the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/), which contains over 50,000 images across 43 sign categories, ranging from speed limit signs to stop signs, yield signs, and more.

---

## Features

- Loads and preprocesses image data using OpenCV (`cv2`)
- Resizes all images to a uniform size (`30x30` pixels) for consistent neural network input
- Builds and trains a CNN using TensorFlow/Keras
- Evaluates model accuracy on a held-out test set
- Optionally saves the trained model to disk for later use

---

## Project Structure

```
traffic/
├── traffic.py          # Main script: data loading, model definition, training & evaluation
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── gtsrb/              # Dataset directory (not included — download separately)
    ├── 0/              # Images for sign category 0
    ├── 1/              # Images for sign category 1
    └── ...             # (43 categories total, 0–42)
```

---

## Getting Started

### Prerequisites

- Python 3.12 (required — other versions may have compatibility issues with TensorFlow)

### Installation

1. Clone or download this repository.
2. Download the GTSRB dataset and place the `gtsrb` folder inside the `traffic/` directory.
3. Install dependencies:

```bash
pip3 install -r requirements.txt
```

### Usage

Run the classifier with:

```bash
python traffic.py gtsrb
```

To train the model and save it:

```bash
python traffic.py gtsrb model.h5
```

---

## How It Works

### `load_data(data_dir)`

Iterates through each of the 43 category subdirectories in the dataset. For every image file found, it uses OpenCV to read the image as a NumPy array and resizes it to `IMG_WIDTH x IMG_HEIGHT` (30×30 pixels). Returns a tuple `(images, labels)` where `images` is a list of NumPy arrays and `labels` is a list of integer category IDs. File path handling is platform-independent via `os.path.join`.

### `get_model()`

Returns a compiled Keras `Sequential` model. The architecture includes:

- One or more **convolutional layers** to detect spatial features (edges, shapes, symbols)
- **Max-pooling layers** to reduce spatial dimensions and improve generalization
- **Flatten layer** to transition from 2D feature maps to a 1D vector
- One or more **dense (fully connected) hidden layers** with ReLU activation
- **Dropout** to reduce overfitting
- An **output layer** with `NUM_CATEGORIES` (43) units and softmax activation for multi-class classification

The model is compiled with the Adam optimizer and categorical crossentropy loss.

---

## Experimentation & Results

### What Was Tried

Several architectural variations were explored:

- **Baseline (single conv layer):** A single convolutional layer with 32 filters and a max-pool layer achieved moderate accuracy (~85%) but plateaued quickly.
- **Deeper conv stack (2 conv layers):** Adding a second convolutional layer with 64 filters improved feature extraction, pushing accuracy to ~92%.
- **Larger dense layers:** Increasing the hidden dense layer from 128 to 512 units improved accuracy slightly but also increased training time and risk of overfitting.
- **Dropout tuning:** A dropout rate of 0.5 after the dense layer proved effective; lower values (0.2–0.3) led to overfitting on the training data.
- **Filter size:** 3×3 filters consistently outperformed 5×5 filters on this dataset, likely because traffic sign features are fine-grained.

### What Worked Well

- Two convolutional layers (32 filters → 64 filters) with 2×2 max pooling after each.
- A single dense hidden layer with 256 units and 0.5 dropout.
- Training for 10 epochs with the Adam optimizer.

This configuration achieved approximately **95% accuracy** on the test set.

### What Didn't Work Well

- Very deep networks (3+ conv layers) did not improve accuracy meaningfully given the small image size (30×30), and increased training time noticeably.
- High learning rates caused unstable loss curves.
- No dropout at all consistently led to overfitting, with training accuracy far exceeding test accuracy.

### Key Takeaway

For this task and image size, a relatively shallow but well-regularized CNN strikes the best balance between accuracy and training efficiency. The gains from adding more layers diminish quickly once the architecture is already capturing the core spatial features of traffic signs.

---

## Sample Output

```
Epoch 1/10  - loss: 3.71 - accuracy: 0.15
Epoch 5/10  - loss: 0.66 - accuracy: 0.80
Epoch 10/10 - loss: 0.25 - accuracy: 0.93
Test accuracy: 0.9535
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | Neural network construction and training |
| `opencv-python` | Image loading and resizing |
| `scikit-learn` | Train/test splitting |
| `numpy` | Array manipulation |

---

