# Brain Tumor Classification via Deep Transfer Learning and Interpretability frameworks

This repository presents a specialized Deep Learning framework for the multi-class classification of brain tumors from MRI imagery. By integrating **ResNet50** architectures with **Explainable AI (XAI)**, this system provides clinically relevant spatial justifications for its diagnostic outputs.

---

### Technical Methodology

The architecture is built on a Transfer Learning paradigm, optimized for high-dimensional medical imaging data:

* **Feature Extraction:** Utilizes a pre-trained **ResNet50** backbone (ImageNet weights) to identify complex textural patterns.
* **Global Average Pooling (GAP):** Implemented to minimize spatial dimensions while preserving vital feature maps, reducing the risk of overfitting.
* **Optimization:** Stochastic Gradient Descent (SGD) with automated **class-weight balancing** to address inherent dataset skew across Glioma, Meningioma, and Pituitary categories.



---

### Quantitative Results

The model underwent rigorous validation to ensure statistical reliability:

| Metric | Training Set | Validation Set | Test Set (Unseen) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 93.2% | 93.5% | **88.0%** |
| **AUC Score** | 0.993 | 0.994 | **0.970** |

> **Note:** The high AUC (Area Under Curve) score indicates superior separation power between pathological classes, a critical requirement for medical diagnostic support systems.

---

### Interpretability (XAI) via Grad-CAM

In clinical environments, "Black-Box" models lack transparency. This project implements **Gradient-weighted Class Activation Mapping (Grad-CAM)** to localize the diagnostic focus.

* **Target Layer:** `conv5_block3_out` (The final residual block)
* **Visualization:** Heatmaps represent the spatial importance of pixels contributing to the final softmax probability.

---

### System Deployment

The system is deployed as an interactive inference tool using **Streamlit**.

1. **Environment Setup:** `pip install -r requirements.txt`
2. **Inference Execution:** `streamlit run app.py`

---
