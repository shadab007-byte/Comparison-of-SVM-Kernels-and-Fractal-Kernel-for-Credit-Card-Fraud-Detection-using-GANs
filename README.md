# Comparison-of-SVM-Kernels-and-Fractal-Kernel-for-Credit-Card-Fraud-Detection-using-GANs
# Credit Card Fraud Detection using GANs and Fractal Kernel SVM

**Department of Mathematics, IIT Madras**  
**Modelling Workshop (MA5770) Project**  
**Supervisor:** Prof. A.K.B. Chand  
**Author:** Mohd Shadab (MA24M015)

---

## 🚀 Overview

This project addresses the challenge of detecting fraudulent credit card transactions by leveraging advanced machine learning techniques: **Generative Adversarial Networks (GANs)** for data augmentation and **Support Vector Machines (SVMs)** enhanced with novel **fractal-based kernels** for classification.

---

## 🔍 Problem Statement

Credit card fraud detection is a classic imbalanced classification problem, with fraudulent transactions constituting less than 1% of all transactions. Traditional methods struggle with such imbalance and fail to capture complex, nonlinear fraud patterns. This project systematically compares traditional SVM kernels with custom fractal-based kernels, using GAN-augmented data for robust training and evaluation.

---

## 🛠️ Methodology

### 1. **Data Augmentation with CTGAN**
- **Original Dataset:** European credit card dataset (284,807 transactions, 492 frauds)
- **CTGAN Synthesis:** Generated synthetic fraud samples to balance the dataset (2,000 legitimate vs 984 fraudulent transactions)
- **Result:** Significantly improved model training and evaluation fairness

### 2. **SVM Kernel Comparison**
- **Traditional Kernels:** Linear, Polynomial, RBF (Gaussian), Sigmoid
- **Fractal-Based Kernels:** Fractal RBF, Mercer Fractal RBF, Alpha-Fractal Kernel
- **Implementation:** Custom kernel matrices for fractal variants

### 3. **Model Evaluation**
- **Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, Matthews Correlation Coefficient
- **Visualization:** Confusion matrices, ROC curves, kernel response plots

---

## 💡 Key Innovations

- **Fractal Kernel Design:** Introduced recursive and Hermite-based fractal kernels to capture self-similar and complex patterns in transaction data.
- **CTGAN Integration:** Used Conditional Tabular GAN to generate realistic synthetic fraud samples, addressing class imbalance without oversampling bias.
- **Comprehensive Benchmarking:** Compared traditional and fractal kernels, highlighting the strengths and trade-offs of each approach.

---

## 📊 Results

| Kernel Type         | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Linear              | 0.9754   | 0.9891    | 0.9349 | 0.9599   | 0.9877  |
| Polynomial          | 0.9576   | 1.0000    | 0.8699 | 0.9304   | 0.9869  |
| RBF (Gaussian)      | 0.9710   | 1.0000    | 0.9110 | 0.9534   | 0.9848  |
| Sigmoid             | 0.9263   | 0.9346    | 0.8322 | 0.8804   | 0.8694  |
| Fractal RBF         | 0.9699   | 0.9752    | 0.9322 | 0.9532   | —       |
| Mercer Fractal RBF  | 0.9688   | 0.9652    | 0.9390 | 0.9519   | —       |
| Alpha-Fractal       | 0.76     | 0.99/0.58 | 0.66/0.98 | 0.79/0.73 | —       |

> **Note:** Alpha-Fractal kernel prioritizes fraud recall (0.98) at the cost of some false positives, which is critical for financial security.

---

## 🎯 Key Takeaways

- **GAN-based augmentation** effectively balances imbalanced datasets, enabling robust classifier training.
- **Fractal kernels** offer a mathematically grounded way to model complex, irregular patterns in fraud data.
- **Linear and RBF kernels** perform strongly on balanced data, but fractal kernels provide unique advantages in capturing intricate decision boundaries.

---

## 🛠️ Getting Started

### Requirements

- **Python 3.10**
- **Libraries:** scikit-learn, CTGAN, NumPy, pandas, matplotlib, seaborn
- **Execution Platforms:** Local machine (Intel Core i3, 12GB RAM) or Google Colab Pro (NVIDIA Tesla T4 GPU)

### Usage

1. **Clone the repository:**

2. **Install dependencies:**

3. **Mount your Google Drive (if using Colab):**

4. **Run the Jupyter notebook:**

5. **Follow the notebook steps:**
- Data loading and preprocessing
- CTGAN-based data augmentation
- Feature scaling and selection
- SVM kernel training and evaluation (traditional and fractal)
- Performance visualization

---

## 📝 Project Report

For a detailed explanation of the methodology, mathematical foundations, and experimental results, refer to the Project Report attached folder.

---

## 🤝 Acknowledgements

I express my sincere gratitude to Prof. A.K.B. Chand for his guidance and support. I also thank the Department of Mathematics, IIT Madras, for providing the resources and environment that made this project possible.

---

## 📚 References

- **Alfaiz, N. S. and Fati, S. M. (2022):** Enhanced credit card fraud detection model using machine learning.
- **Alshawi, B. (2024):** Comparison of SVM kernels in credit card fraud detection using GANs.
- **Kumar, D., Chand, A. K. B. and Massopust, P. R. (2023):** Multivariate zipper fractal functions.
- **Xu, L., Skoularidou, M., Cuesta-Infante, A. and Veeramachaneni, K. (2019):** Modeling tabular data using conditional GAN.

---

## 🌟 Future Work

- **Hyperparameter optimization** for fractal kernels
- **Integration with explainable AI** techniques (e.g., SHAP)
- **Generalization to other anomaly detection tasks** (healthcare, cybersecurity)

---

**© 2025 | Department of Mathematics, IIT Madras**  
**Project Author:** Mohd Shadab

