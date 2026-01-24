# app.py
import streamlit as st
import numpy as np
import torch
import os
from model import EEGNet  # make sure your EEGNet class is in model.py

# 1. Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Change this path to your trained model weights
MODEL_PATH = "epilepsy_classification_model.pth"  

# Initialize EEGNet with same parameters used during training
model = EEGNet(num_classes=5, chans=1, samples=4097, F1=16, D=2, F2=32, dropout_rate=0.3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Class labels
labels_map = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}
simplified_map = {
'A': 'Healthy (Eyes Open)',
'B': 'Healthy (Eyes Closed)',
'C': 'Epilepsy (Focus Side)',
'D': 'Epilepsy (Opposite Side)',
'E': 'Seizure'
}

# 2. Streamlit UI
st.title("🧠 Epilepsy EEG Classification")
st.write("Upload a single-channel EEG `.txt` file (4097 samples) to predict its class.")

uploaded_file = st.file_uploader("Upload EEG .txt file", type=["txt"])
if uploaded_file is not None:
    try:
        # Load EEG signal
        data = np.loadtxt(uploaded_file)
        if data.shape[0] != 4097:
            st.warning("⚠️ File length must be exactly 4097 samples.")
        else:
            # Preprocess (same as training)
            x = (data - data.mean()) / (data.std() + 1e-8)
            x = x[np.newaxis, np.newaxis, np.newaxis, :]  # shape (1,1,1,4097)
            x_tensor = torch.FloatTensor(x).to(device)

            # Predict
            with torch.no_grad():
                output = model(x_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_class = np.argmax(probs)

            # Display results
            st.success(f"Predicted Class: **{labels_map[pred_class]}**")
            st.info(f"Predicted State: **{simplified_map[labels_map[pred_class]]}**")

            st.write("Class Probabilities:")
            for i, label in labels_map.items():
                st.write(f"{label}: {probs[i]*100:.2f}%")

            # Plot EEG waveform
            st.line_chart(data)

    except Exception as e:
        st.error(f"Error processing file: {e}")

# Footer
st.write("---")