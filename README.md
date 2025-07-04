# 🫀 DL-MI-ECG: Deep Learning-Based Myocardial Infarction Detection from ECG Images

This repository presents a deep learning pipeline for the automatic classification of ECG images to detect **myocardial infarction (MI)** and other cardiac conditions using advanced CNN architectures and attention mechanisms.

---

## 📁 Dataset Description

The dataset used is the **[ECG Images Dataset of Cardiac Patients](https://data.mendeley.com/datasets/6y5dyrj9v2/1)**, consisting of **3951 pre-segmented ECG images** across 4 clinical categories:

| Class                | Training Images | Testing Images | Total |
|---------------------|------------------|----------------|-------|
| Myocardial Infarction (MI) | 956              | 239            | 1195 |
| Abnormal Heartbeat  | 699              | 233            | 932   |
| History of MI       | 516              | 172            | 688   |
| Normal ECG          | 852              | 284            | 1136  |

- Format: Preprocessed ECG images from individual leads.
- Compatibility: Supports explainability tools like Grad-CAM and Integrated Gradients.
- Purpose: Enables multi-class ECG classification targeting acute and chronic cardiac conditions.

---

## 🧠 Model Architectures

### 🔹 Baseline Models

| Model           | Accuracy | Precision | Recall | F1 Score |
|----------------|----------|-----------|--------|----------|
| Baseline CNN   | 0.72     | 0.70      | 0.68   | 0.69     |
| MobileNet      | 0.48     | 0.39      | 0.43   | 0.38     |
| ResNet50       | 0.68     | 0.65      | 0.64   | 0.62     |
| DenseNet121    | 0.90     | 0.91      | 0.91   | 0.91     |
| ConvNeXt-Tiny  | 0.26     | 0.06      | 0.25   | 0.10     |

### 🔸 Modified Models (Enhanced Performance)

| Model                          | Accuracy | Precision | Recall | F1 Score |
|-------------------------------|----------|-----------|--------|----------|
| DenseNet + Bottleneck         | 0.91     | 0.90      | 0.91   | 0.91     |
| DenseNet + Attention          | 0.91     | 0.93      | 0.99   | 0.96     |
| EfficientNet-B0 + MSA (Best)  | 0.96     | 0.94      | 0.97   | 0.95     |

---

## ⚙️ Architectural Modifications

1. **DenseNet121 + Bottleneck Block**  
   Adds a Conv2D → BatchNorm → ReLU → Conv2D block after the final dense layer to deepen the model without drastically increasing complexity.

2. **DenseNet121 + Channel Attention**  
   Integrates a squeeze-and-excitation (SE) block after the final convolutional feature map to recalibrate channels and suppress noise.

3. **EfficientNet-B0 + Multi-Scale Attention (MSA)** 🔥  
   Incorporates hierarchical attention blocks capturing:
   - **Block 3**: Intermediate spatial details  
   - **Block 4**: Deeper semantic features  
   - **Block 6**: High-level abstract representations  
   Best model across all metrics, showcasing superior performance in clinical ECG classification.

---

## 📈 Results Summary

> All modified models surpass the baseline with **accuracy >91%** and strong F1 scores.  
> EfficientNet-B0 + MSA delivers state-of-the-art performance with **96% accuracy** and **0.95 F1-score**, proving the value of multi-scale abstraction.

---

## 🧪 Training & Evaluation

- **Loss Function**: Categorical Crossentropy  
- **Optimizer**: Adam (LR = 1e-4)  
- **Metrics**: Accuracy, Precision, Recall, F1 Score  
- **Validation Strategy**: Stratified train-test split  
- **Hardware**: NVIDIA GPU (Recommended)

---

## 🔍 Explainability

- **Tools Used**: Grad-CAM, Integrated Gradients  
- Used to visualize class activation maps and enhance interpretability in clinical environments.

---

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/DL-MI-ECG.git
cd DL-MI-ECG
pip install -r requirements.txt
# Train a model
python train_model.py --model efficientnet_msa --epochs 30

# Evaluate model
python evaluate_model.py --model efficientnet_msa

# Visualize explanations
python explainability.py --method gradcam

