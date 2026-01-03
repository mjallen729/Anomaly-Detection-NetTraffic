# Network Intrusion Detection Using Unsupervised Learning (NSL-KDD)

A comparative study of deep learning, ensemble, and clustering approaches for network anomaly detection using the NSL-KDD dataset. The Autoencoder-based deep learning model achieved **92% attack detection rate** with balanced precision and recall.

---

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Approach |
|-------|----------|-----------|--------|----------|---------|----------|
| **Autoencoder** | **0.908** | **0.919** | **0.921** | **0.920** | **0.951** | Deep Learning |
| Isolation Forest | 0.787 | 0.792 | 0.849 | 0.819 | 0.876 | Ensemble |
| K-Means | 0.771 | 0.979 | 0.611 | 0.752 | 0.905 | Clustering |

**Key Finding**: The Autoencoder outperformed traditional unsupervised ML methods by 10% in F1 score, demonstrating superior balanced performance between precision and recall.

**Performance Highlights**:
- 92.1% attack detection rate
- 89.2% normal detection rate
- 94x higher reconstruction error for attacks vs. normal traffic

---

## Comparison with Existing Research

This project's autoencoder performance compares favorably with published research on the NSL-KDD dataset:

| Research/Model | F1 Score | Precision | Recall | Notes |
|----------------|----------|-----------|--------|-------|
| **This Project** | **92.0%** | **91.9%** | **92.1%** | PyTorch, trained on normal traffic only |
| Various Autoencoders [2] | 89.3-92.3% | - | - | Multiple architectures tested |
| Denoising Autoencoder [1] | 89.3% | - | - | Dropout-based regularization |
| VAE + XGBoost [3] | 94.7% | 99.7% | 89.4% | Hybrid supervised approach |
| Stacked Sparse AE [1] | - | - | - | With SVM classifier |

This project's autoencoder F1 score of 92% places it at the upper range of purely unsupervised/semi-supervised autoencoder approaches, demonstrating effective reconstruction error-based detection without attack-specific training.

---

## Dataset & Problem Formulation

### NSL-KDD Dataset
- **Training**: 125,973 samples (53% normal, 47% attacks)
- **Testing**: 22,544 samples (43% normal, 57% attacks)
- **Original**: 23 different attack categories across network traffic
- **Reframed as binary classification**: Normal vs. Anomaly (all attack types combined)

### Preprocessing Pipeline
1. **Log transformations** on skewed features (src_bytes, dst_bytes, duration, etc.)
2. **One-hot encoding** for categorical features (protocol_type)
3. **Frequency encoding** for high-cardinality features (service, flag)
4. **PCA dimensionality reduction**: 41 features → 18 principal components (95% variance retained)
5. **Feature removal**: Zero-variance and redundant features eliminated

### Why Unsupervised Learning?

Despite having labeled data, unsupervised approaches were prioritized to:
- **Simulate real-world scenarios** where attack labels are unavailable during deployment
- **Enable detection of novel attacks** not present in training data (zero-day threats)
- **Test generalization** beyond known attack patterns
- **Evaluate models' ability** to learn "normal" behavior and flag deviations autonomously

This approach validates whether models can detect intrusions without attack-specific supervision, making them more robust to evolving threats.

---

## Models Implemented

### 1. Autoencoder (Deep Learning - PyTorch)

**Architecture**: 18 → 14 → 12 (bottleneck) → 14 → 18

**Approach**: Semi-supervised learning trained exclusively on normal traffic. Anomaly detection based on reconstruction error threshold.

```python
mse_loss = F.mse_loss(reconstructed, input)
is_anomaly = mse_loss > 0.020  # Optimized threshold
```

**Results**:
- Accuracy: **0.908**
- F1 Score: **0.920**
- ROC-AUC: **0.951**
- Normal traffic median error: 0.0034
- Attack traffic median error: 0.3209 (94x higher)

**Key Strength**: Best overall performance with balanced precision/recall. Effective at detecting novel attacks by learning normal patterns.

---

### 2. Isolation Forest (Ensemble)

**Architecture**: 512 isolation trees with 90% subsampling

**Approach**: Unsupervised anomaly detection using isolation depth. Hyperparameters tuned via grid search optimizing F1 score.

**Results**:
- F1 Score: 0.819
- ROC-AUC: 0.876
- Detection rate: 84.9% of attacks flagged

**Key Strength**: Fast training, no labeled data required, good for real-time detection scenarios.

---

### 3. K-Means Clustering

**Architecture**: 20 clusters (optimized via silhouette analysis, score: 0.5796)

**Approach**: Unsupervised clustering with majority-vote labeling (clusters with ≥50% attacks marked as anomalous).

**Results**:
- F1 Score: 0.752
- ROC-AUC: 0.905
- Precision: **0.979** (highest)
- Recall: 0.611 (misses 39% of attacks)

**Key Strength**: Highest precision (few false positives), interpretable cluster compositions.

**Limitation**: Low recall makes it unreliable for security-critical applications where missing attacks is costly.

---

## Technical Implementation

### Tech Stack
- **Deep Learning**: PyTorch 2.9.1
- **ML Models**: scikit-learn 1.7.2
- **Data Processing**: pandas 2.3.3, NumPy 2.3.4
- **Visualization**: matplotlib 3.10.7, seaborn 0.13.2

### Project Structure
```
├── data/              # NSL-KDD dataset (raw and preprocessed)
├── notebooks/         # Data exploration and preprocessing
│   ├── preprocess.ipynb
│   ├── pca.ipynb
│   └── explore.ipynb
├── models/            # Model implementations and training
│   ├── autoencoder.ipynb
│   ├── isolation_forest.ipynb
│   └── kmeans.ipynb
└── scripts/           # Data setup utilities
    └── setup_data.py
```

---

## Key Insights

### Performance Analysis
- Autoencoder achieved **92.1%** attack detection with **91.9%** precision
- Deep learning outperformed traditional unsupervised methods by **10% in F1 score**
- PCA reduced dimensionality by **56%** while retaining **95% variance**
- Reconstruction error clearly separates normal (0.0034) vs. attack (0.3209) traffic

### Model Trade-offs
- **K-Means**: Highest precision (97.9%) but misses 39% of attacks — too conservative for security applications
- **Isolation Forest**: Balanced approach, faster training, suitable for real-time detection with moderate accuracy
- **Autoencoder**: Best overall results but requires more computational resources for training and inference

### Unsupervised Learning Validation
- Models successfully distinguished normal vs. anomalous patterns **without attack-specific training**
- Autoencoder's semi-supervised approach (trained only on normal traffic) proved most effective
- Results demonstrate **viability of unsupervised methods for detecting unknown threats**, crucial for real-world deployments where novel attacks emerge constantly

---

## Conclusions

- **Deep learning approach (Autoencoder) demonstrated superior anomaly detection performance**, achieving the highest balanced F1 score (0.920) and ROC-AUC (0.951) among all models tested

- **Unsupervised methods successfully detected attacks without attack-specific training**, validating their use for novel threat detection where labeled attack data is unavailable or incomplete

- **Preprocessing and dimensionality reduction (PCA) are crucial** for computational efficiency, reducing features by 56% while maintaining 95% variance and enabling faster model training without sacrificing accuracy

---

## Repository Contents

- **Notebooks**: Complete implementations of data preprocessing, PCA, and all three anomaly detection models
- **Models**: Autoencoder (PyTorch), Isolation Forest, K-Means with hyperparameter tuning
- **Visualizations**: ROC curves, reconstruction error distributions, cluster analysis, feature correlations
- **Scripts**: Data preparation utilities

---

**Dataset Source**: NSL-KDD (improved version of KDD Cup 1999 dataset)

---

## References

1. [Improving Performance of Autoencoder-Based Network Anomaly Detection on NSL-KDD Dataset (IEEE 2021)](https://ieeexplore.ieee.org/document/9552882/)
2. [Analysis of Autoencoders for Network Intrusion Detection (MDPI 2021)](https://www.mdpi.com/1424-8220/21/13/4294)
3. [Intrusion Detection in NSL-KDD Dataset Using Hybrid Self-Organizing Map Model (2025)](https://www.techscience.com/CMES/v143n1/60475/html)
