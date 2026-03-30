# QGAN-Classifier-task4

# Quantum GAN for High Energy Physics Data Analysis

This repository contains a implementation of a **Quantum Generative Adversarial Network (QGAN)** Discriminator designed to separate **Signal** events from **Background** events in High Energy Physics data using **TensorFlow Quantum (TFQ)** and **Google Cirq**.

## Overview
- **Data**: Simulated events in NumPy NPZ format (100 training, 100 test samples).
- **Quantum Circuit**: A Variational Quantum Circuit (VQC) using RX angle encoding, RY/RZ rotations, and CNOT entanglement layers.
- **Framework**: TensorFlow Quantum for the hybrid quantum-classical model.

## Performance
The model achieves a stable Area Under Curve (AUC) of approximately **0.80**, demonstrating strong discriminative power even with a small dataset.

### Visual Results
| Receiver Operating Characteristic | Signal vs Background Separation |
| :---: | :---: |
| ![ROC Curve](images/roc_curve.png) | ![Separation Plot](images/separation_plot.png) |

## Setup and Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
