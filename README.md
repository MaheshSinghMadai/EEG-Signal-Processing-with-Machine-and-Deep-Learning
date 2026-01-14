# EEG Signal Processing with Machine and Deep Learning

A comprehensive implementation of EEG signal processing techniques combined with machine learning and deep learning models for brain-computer interface applications and neurological signal analysis.

## 📋 Overview

This project provides a complete pipeline for processing and analyzing EEG (Electroencephalogram) signals using modern machine learning approaches. It includes signal preprocessing, feature extraction, and classification using both traditional ML and deep learning methods.

## ✨ Features

- **Signal Preprocessing**
  - Bandpass and notch filtering
  - Artifact removal (EOG, EMG)
  - Epoch segmentation
  - Data normalization

- **Feature Extraction**
  - Time-domain features
  - Frequency-domain features (FFT, PSD)
  - Wavelet transform
  - Common Spatial Patterns (CSP)

- **Machine Learning Models**
  - Random Forest
  - Logistic Regression

- **Deep Learning Architectures**
  - Convolutional Neural Networks (CNN)
  - Recurrent Neural Networks (LSTM/GRU)
  - EEGNet
  - Chrononet
  - Hybrid architectures

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/MaheshSinghMadai/EEG-Signal-Processing-with-Machine-and-Deep-Learning.git
cd EEG-Signal-Processing-with-Machine-and-Deep-Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```python
# Load and preprocess EEG data
from src.preprocessing import preprocess_eeg
from src.models import train_model

# Preprocess your data
X_train, y_train = preprocess_eeg('data/train/')

# Train a model
model = train_model(X_train, y_train, model_type='eegnet')

# Evaluate
accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}%")
```

## 📁 Project Structure

```
├── EEG Datasets/                   # Data directory
├── notebooks/              # Jupyter notebooks for analysis
├── Epilepsy Classification/                    # Source code
│   ├── Datasets/      # Signal preprocessing
│   ├── Notebook files           # Feature extraction
├── requirements.txt        # Python dependencies
└── README.md
```

## 🔧 Usage

### Preprocessing

```python
from src.preprocessing import apply_bandpass_filter, create_epochs

# Filter EEG signals
filtered_data = apply_bandpass_filter(raw_data, lowcut=0.5, highcut=50)

# Create epochs
epochs = create_epochs(filtered_data, duration=4.0)
```

### Training Models

```python
# Train traditional ML model
from src.models import SVMClassifier

svm_model = SVMClassifier()
svm_model.fit(X_train, y_train)
accuracy = svm_model.score(X_test, y_test)

# Train deep learning model
from src.models import EEGNet

model = EEGNet(n_channels=64, n_classes=4)
model.train(X_train, y_train, epochs=100, batch_size=32)
```
<!--
## 📊 Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| SVM + CSP | 78.5% | 0.76 |
| Random Forest | 75.2% | 0.73 |
| CNN | 82.3% | 0.81 |
| EEGNet | 85.7% | 0.84 |
| LSTM | 80.1% | 0.79 |
-->
## 📦 Dependencies

- NumPy
- SciPy
- scikit-learn
- TensorFlow/Keras or PyTorch
- MNE-Python
- Pandas
- Matplotlib
- Seaborn

See `requirements.txt` for complete list.

## 🗃️ Datasets

This project supports various EEG datasets:

- PhysioNet Motor Movement/Imagery Dataset
- BCI Competition datasets
- Custom EEG recordings
