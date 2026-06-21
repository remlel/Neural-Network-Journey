# Neural Network Journey

Welcome to my Deep Learning logbook. This repository documents my **progressive and empirical** learning path, starting from the fundamental concepts of neural networks to designing advanced industrial architectures applied to digital signal processing.

The purpose of this project is to track my progression through concrete problems of increasing complexity, focusing on physical intuition, tensor manipulation, and GPU performance optimization.

---

## 🚀 Projects Overview

### 1. Topological Binary Classification (Two Moons)
* **Objective:** Trace a non-linear decision boundary on a two-dimensional asymmetric dataset (red and blue points shaped like interlocking half-moons).
* **Architecture:** Classic Deep Multilayer Perceptron (MLP) with non-linear activations (ReLU).
* **Validated Concepts:** Gradient backpropagation, weight initialization, decision boundary intuition.

| Initial Data Distribution (Input) | Discovered Decision Boundary (Output) |
| :---: | :---: |
| ![Initial Distribution](images/moons_input.png) | ![Found Boundary](images/moons_output.png) |

---

### 2. Human Activity Recognition (UCI HAR Dataset)
* **Objective:** Classify 6 distinct human activities (walking, walking upstairs, walking downstairs, sitting, standing, laying) using inertial sensors (tri-axial accelerometer and gyroscope) from a smartphone.
* **Approach 1 (Statistical Features):** Training a dense linear network on a 2D tensor containing 561 pre-computed statistical features (means, standard deviations, etc.).
    * **Performance:** **96.4% accuracy** on the test set.
* **Approach 2 (Raw Temporal Signals):** Designing 1D convolutional architectures (1D CNN, 1D Inception, and 1D ResNet) processing raw time-series directly (Tensor shape: `[Batch, 9, 128]`).
    * **Performance:** **97.3% accuracy** achieved with the 1D ResNet architecture.

| Confusion Matrix - Statistical Features (96.4%) | Confusion Matrix - Raw Signals 1D ResNet (97.3%) |
| :---: | :---: |
| ![HAR Matrix 1](images/mc_HAR1.png) | ![HAR Matrix 2](images/mc_HAR2.png) |

---

### 3. Radar Target Classification (Micro-Doppler Signatures)
* **Objective:** Identify 4 types of moving targets (vehicle, pedestrian, cyclist, UAV/drone) from real-world radar acquisitions (Open Radar Initiative).
* **Technical Pipeline:** Temporal slicing of variable-length trajectories (*tracks*) into fixed windows of 10 spectra ($1008 \times 10$) and full pipeline migration to GPU CUDA.
* **Representation Learning:** Comparative evaluation of input signal geometry:
    1. *Power (dB)* ➔ Standard logarithmic approach (Top: 87.7% with ResNet).
    2. *Raw (Real & Imaginary)* ➔ Linear algebraic form $z = a + jb$ (Top: 72%).
    3. *Physics-Informed (Magnitude & Phase)* ➔ Polar form $z = r e^{j\phi}$ directly exposing spatial-temporal velocity gradients $\Delta \phi / \Delta t$.
* **Champion Performance:** **89% accuracy** achieved by combining the **Magnitude & Phase** representation with a **2D ResNet architecture with 4 residual blocks**, optimized using asymmetric rectangular filters tailored to the signal physics.

| Confusion Matrix - 2D ResNet (89%) |
| :---: |
| ![Confusion Matrix](images/mc_Radar.png) |
