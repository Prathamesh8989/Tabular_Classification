Rice Morphological Classification using Custom Deep Feedforward Neural Networks

[![PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Preprocessing-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.68%25-success?style=for-the-badge)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An end-to-end Machine Learning pipeline built with PyTorch and Scikit-Learn to classify rice varieties based on their spatial and structural morphological features. The pipeline implements maximum absolute scaling normalization, deterministic data splits, and a custom feedforward neural network architecture optimized via an Adam optimizer to achieve a **98.68% testing accuracy**.

---

## 📌 Table of Contents
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

$$X_{normalized} = \frac{X}{\vert{}X\vert{}_{max}}$$

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
| **Sigmoid Non-Linearity** | `[-1, 1]` | `[-1, 1]` | 0 | Probability Activation Function