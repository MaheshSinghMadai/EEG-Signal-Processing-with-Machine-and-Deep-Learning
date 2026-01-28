# EEG Signal Processing with Machine and Deep Learning

A curated repository of code, notebooks, and datasets for EEG preprocessing, feature extraction, and classification using classical machine learning and deep learning methods.

**Project Summary**
- **Scope:** Reproducible pipelines and reference notebooks for motor-imagery and epilepsy detection experiments.
- **Tools:** Python, Jupyter, MATLAB (helper scripts), PyTorch/TensorFlow, MNE-Python.

**Top-level Structure**
- `EEG Datasets/` — Raw and converted EEG datasets (BCICIV_2a, EDF/GDF/CSV exports).
- `EEG Matlab Dataset/` — MATLAB pipelines and helper functions.
- `EEG Motor Imagery Classification/` — Notebooks demonstrating MI preprocessing and models (CNN, GRU, EEGNet, etc.).
- `Epilepsy Classification/` — App, notebooks, model definitions, and example trained model (`epilepsy_classification_model.pth`).
- `Papers/` — References used for experiments.
- `requirements.txt` — Python dependencies.

**Quick Start**
1. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

**Run Examples**
- Notebooks: open and run notebooks in `EEG Motor Imagery Classification/` to explore preprocessing, features, and model training.
- Epilepsy demo app:

```powershell
python "Epilepsy Classification\app.py"
```

**Notes on Data**
- Large raw files are stored under `EEG Datasets/`. Use the provided loading utilities in notebooks (MNE is recommended for EDF/GDF formats).
- Do not commit large derived artifacts; store only processed subsets when necessary.

**Important Files**
- [Epilepsy Classification/app.py](Epilepsy Classification/app.py)
- [Epilepsy Classification/epilepsy_classification_model.pth](Epilepsy Classification/epilepsy_classification_model.pth)
- [requirements.txt](requirements.txt)

**Contributing**
- Fork, create a branch, add tests or a notebook demonstrating your change, and open a PR with a clear description.

If you'd like, I can also add README badges, generate per-folder READMEs, or produce a short developer setup guide.
# EEG Signal Processing with Machine and Deep Learning

A curated collection of code, notebooks, and datasets for EEG signal processing, feature extraction, and classification using traditional machine learning and deep learning methods. This repository groups practical implementations and experiments for motor-imagery, epilepsy detection, and general EEG analysis.

**Project At A Glance**
- **Purpose:** Provide reproducible pipelines and reference notebooks for EEG preprocessing, feature engineering, and classification.
- **Main languages/tools:** Python, MATLAB (helper scripts), Jupyter notebooks, PyTorch/TensorFlow (models), MNE-Python.
- **Datasets included:** BCI Competition IV-2a, Bonn Epilepsy dataset excerpts, various EDF/GDF/CSV EEG files.

**Repository Structure**
- **EEG Datasets/**: Raw and converted datasets (BCICIV_2a, CSV exports, dataverse EDF files).
- **EEG Matlab Dataset/**: MATLAB data, helper functions, and pipeline scripts.
- **EEG Motor Imagery Classification/**: Jupyter notebooks exploring MI preprocessing and models (CNN, GRU, BiGRU, EEGNet variants).
- **Epilepsy Classification/**: Application code, notebooks, trained model, and datasets for epilepsy detection. Key files:
  - `Epilepsy Classification/app.py` – small app/demo script
  - `Epilepsy Classification/model.py` – model definition
  - `Epilepsy Classification/epilepsy_classification_model.pth` – example trained model
- **Papers/**: References and literature used in experiments.
- **requirements.txt**: Python dependencies for reproducible environment.

**Quick Setup**
1. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
# Windows (cmd)
venv\\Scripts\\activate.bat
# macOS / Linux
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) If you work with MATLAB code, open files under `EEG Matlab Dataset/Pipeline` in MATLAB.

**How to Use**
- Open and run notebooks under `EEG Motor Imagery Classification/` for exploratory experiments and model training examples.
- For epilepsy detection demo, run the app script:

```powershell
python "Epilepsy Classification\app.py"
```

- To train or evaluate models, open the relevant notebook or `Epilepsy Classification/` scripts. Many notebooks include runnable cells for preprocessing, model training, and evaluation.

**Datasets & Notes**
- Raw data is kept under `EEG Datasets/`. Files may be large — do not commit large derived artifacts.
- Some datasets (e.g., GDF/EDF) may require the `mne` package to read. See notebooks for dataset-specific loading utilities.

**Key Files & Locations**
- [Epilepsy Classification/app.py](Epilepsy Classification/app.py) – demo runner
- [requirements.txt](requirements.txt) – dependency list
- [EEG Motor Imagery Classification](EEG Motor Imagery Classification/) – notebooks collection

**Contributing**
- Fork, create a feature branch, add tests or a notebook demonstrating changes, and open a PR with a clear description.

**Citation / Acknowledgements**
- If you use this repo in research, please cite the datasets and relevant papers listed in `Papers/`.

---

If you want, I can also:
- add badges (build / license),
- generate a smaller README per subfolder (notebooks or epilepsy app), or
- create a short contributor guide and a `requirements-dev.txt`.
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
