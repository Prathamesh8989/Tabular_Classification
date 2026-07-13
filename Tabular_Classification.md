# Rice Morphological Classification using Custom Deep Feedforward Neural Networks

[![PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Preprocessing-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.68%25-success?style=for-the-badge)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An end-to-end Machine Learning pipeline built with PyTorch and Scikit-Learn to classify rice varieties based on their spatial and structural morphological features. The pipeline implements maximum absolute scaling normalization, deterministic data splits, and a custom feedforward neural network architecture optimized via an Adam optimizer to achieve a **98.68% testing accuracy**.

---

##  Table of Contents
1. [Project Summary](#1-project-summary)
2. [Results at a Glance](#2-results-at-a-glance)
3. [Data Pipeline & Normalization Strategy](#3-data-pipeline--normalization-strategy)
4. [Model Architecture](#4-model-architecture)
5. [Mathematical Foundations](#5-mathematical-foundations)
6. [Training Progress & Convergence Performance](#6-training-progress--convergence-performance)
7. [Inference & Production Deployment Playbook](#7-inference--production-deployment-playbook)
8. [Installation & Reproduction Guide](#8-installation--reproduction-guide)

---

## 1. Project Summary

This project addresses the task of classifying grain structures using precise morphological criteria. Starting from structural measurement files, the pipeline handles feature parsing, data cleansing, and vector normalization. It feeds these vectors into a multi-layer linear network graph to isolate structural variations between different categories. 

The entire process runs seamlessly: `CSV ingestion → Null cleaning & ID stripping → Max-Absolute feature scaling → 70/15/15 deterministic validation split → Custom PyTorch Tensor DataLoaders → Network forward pass → Binary Cross-Entropy loss estimation → Adam backpropagation → Production single-sample inference simulation`.

---

## 2. Results at a Glance

| Metric | Operational Strategy / Output Value |
| :--- | :--- |
| **Target Task** | Binary Tabular Grain Classification (`Class 0` vs `Class 1`) |
| **Dataset Footprint** | 18,185 Individual Samples (10 Input Metrics, 1 Target Category) |
| **Data Constraints** | 0 Missing Values (Dropped via strict null filtration) |
| **Network Configuration** | Multi-Layer Perceptron (10 Features → 10 Hidden Neurons → 1 Output) |
| **Trainable System Parameters** | 121 Learnable Weights and Biases |
| **Batch Processing Volume** | 32 Samples Per Step |
| **Final Generalization Accuracy** | **98.68%** (Evaluated on unseen test partition) |

---

## 3. Data Pipeline & Normalization Strategy

### 3.1 Feature Extraction Matrix
The dataset uses 10 distinct morphological measurements to track variations in sample area and contour lines:

* `Area`: Total pixel count enclosed within the grain boundaries.
* `MajorAxisLength`: The longest distance across the structural envelope.
* `MinorAxisLength`: The shortest structural coordinate axis perpendicular to the major axis.
* `Eccentricity`: Elliptical variance tracking divergence from circular profiles.
* `ConvexArea`: Pixel count defining the smallest convex polygon enclosing the boundary.
* `EquivDiameter`: Diameter of a perfect circle sharing the identical area baseline.
* `Extent`: Ratio of pixel area to the bounding box dimension.
* `Perimeter`: Continuous boundary perimeter length.
* `Roundness`: Geometric circularity factor tracking surface roughness.
* `AspectRation`: Length-to-width ratio calculated across the primary axis profiles.

### 3.2 Max-Absolute Feature Scaling
To balance features with widely differing scales (e.g., small decimal ratios versus large area pixel values), the pipeline applies a Max-Absolute scaling transformation across every input row vector:

$$X_{normalized} = \frac{X}{|X|_{max}}$$

This binds feature activations between $0$ and $1$, preventing variables with large raw ranges from dominating the model's gradients.

### 3.3 Data Partitioning Breakdown
Splits are computed using a deterministic random seed to guarantee exactly reproducible data slices across different runs:

| Partition Split | Percentage allocation | Dataset Rows | Intended Mathematical Purpose |
| :--- | :---: | :---: | :--- |
| **Training Split** | 70% | 12,729 | Computes loss gradients and updates network parameters. |
| **Validation Split** | 15% | 2,728 | Monitors out-of-sample loss convergence per epoch. |
| **Testing Split** | 15% | 2,728 | Unbiased final verification score for the model. |

---

## 4. Model Architecture

The custom model (`MyModel`) is structured as a compact, highly optimized Feedforward Neural Network containing an input projection layer, a hidden layer, and a non-linear probability transformation head.

   [Input Vector: 10 Features]
                │
                ▼ 
 Linear Layer (10 Inputs -> 10 Hidden)   ---> [110 Weight/Bias Parameters]
                │
                ▼ 
 Linear Layer (10 Hidden -> 1 Output)    ---> [11 Weight/Bias Parameters]
                │
                ▼ 
     Sigmoid Scaling Function
                │
                ▼ 
[Output Scalar Probability: Interval (0, 1)]


### Explicit Architecture Trace Summary

| Layer Block | Structural Input Dimension | Structural Output Dimension | Total Learnable Parameters | Layer Transformation Type |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Layer 1** | `[-1, 10]` | `[-1, 10]` | 110 | Input to Hidden Projection |
| **Linear Layer 2** | `[-1, 10]` | `[-1, 1]` | 11 | Hidden to Single Logit Mapping |
| **Sigmoid Non-Linearity** | `[-1, 1]` | `[-1, 1]` | 0 | Probability Activation Function |

* **Total Model Complexity**: 121 Parameters (100% Trainable)

---

## 5. Mathematical Foundations

### 5.1 Fully Connected Transformation Model
Each linear network layer applies a matrix multiplication operation to its input vectors, adding a bias vector to shift the output space:

$$\mathbf{Y} = \mathbf{X}\mathbf{W}^T + \mathbf{b}$$

Where $\mathbf{W}$ represents the learnable parameter weight matrix and $\mathbf{b}$ represents the local layer offset bias vector.

### 5.2 Sigmoid Logistic Activation
The final output layer uses a Sigmoid function to compress raw numerical logits into a valid $[0, 1]$ probability range for binary classification:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### 5.3 Binary Cross-Entropy Loss (BCELoss)
The optimization loop calculates prediction error using standard Binary Cross-Entropy Loss:

$$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

Where $y_i$ is the actual binary target label ($0$ or $1$) and $p_i$ is the predicted probability output by the model's Sigmoid layer.

---

## 6. Training Progress & Convergence Performance

The pipeline was run for 10 epochs using a fixed learning rate of `1e-3` via the Adam optimizer. The scaled validation loss steadily followed the training loss curve down to convergence, confirming clean generalization without overfitting.

### Chronological Epoch Step Values

| Epoch ID | Scaled Training Loss | Training Accuracy (%) | Scaled Validation Loss | Validation Accuracy (%) |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.2552 | 74.66% | 0.0459 | 98.06% |
| **2** | 0.1565 | 97.99% | 0.0226 | 98.94% |
| **3** | 0.0747 | 98.25% | 0.0113 | 98.86% |
| **5** | 0.0308 | 98.49% | 0.0055 | 98.83% |
| **7** | 0.0224 | 98.55% | 0.0040 | 98.97% |
| **9** | 0.0194 | 98.60% | 0.0035 | 99.05% |
| **10** | 0.0187 | **98.59%** | 0.0034 | **99.08%** |

### Final Benchmark Performance Score
* **Unseen Evaluation Test Partition Accuracy**: **`98.68%`**

---

## 7. Inference & Production Deployment Playbook

For production inference, new real-world feature vectors must match the model's training data distribution. Raw inputs are normalized on the fly by dividing each feature by the maximum absolute value observed during training before being passed to the model:

[Raw Production User Inputs]
│
▼
[Divide by original_df[column].abs().max() Vector]
│
▼
[Normalized Input Tensor]
│
▼
[Model Forward Calculation]
│
▼
[Round Code Output Float]  ───► Class Output (0 or 1)


---

## 8. Installation & Reproduction Guide

### 1. Environment Preparation
Ensure you have Python 3.10+ and virtualenv configured locally:
```bash
git clone [https://github.com/your-username/rice-tabular-classification.git](https://github.com/your-username/rice-tabular-classification.git)
cd rice-tabular-classification
python3 -m venv ml_env
source ml_env/bin/activate
2. Dependency Ingestion
Install the required machine learning and visualization packages via pip:

Bash
pip install torch torchsummary scikit-learn pandas numpy matplotlib
3. Pipeline Ingestion & Invocations
Ensure your feature source asset file (riceClassification.csv) is present in the workspace root directory, then execute the notebook:

Bash
jupyter notebook Tabular_Classification.ipynb
