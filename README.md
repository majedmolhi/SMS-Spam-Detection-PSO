# SMS Spam Detection Using PSO-Based Feature Selection

A machine learning project that detects SMS spam messages using Particle Swarm Optimization (PSO) for feature selection combined with Naive Bayes and Logistic Regression classifiers.

---

## Dataset

SMS Spam Collection — UCI Machine Learning Repository  
Link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

| Property       | Value              |
|----------------|--------------------|
| Total messages | 5,572              |
| Ham            | 4,825 (86.6%)      |
| Spam           | 747 (13.4%)        |
| Language       | English            |
| Format         | CSV                |

---

## Methodology

1. **Text Preprocessing** — lowercasing, removing punctuation, stopword removal, stemming (Porter Stemmer)
2. **Class Imbalance Handling** — SMOTE applied to training data only
3. **Feature Extraction** — TF-IDF with 1,500 features, bigrams, sublinear TF
4. **Feature Selection** — Binary PSO run separately for each classifier
5. **Classification** — Naive Bayes and Logistic Regression
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## Results

| Model         | Features | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|----------|-----------|--------|----------|
| NB Before PSO | 1,500    | 96.05%   | 81.07%    | 91.95% | 86.16%   |
| NB After PSO  | 723      | 97.04%   | 88.16%    | 89.93% | 89.04%   |
| LR Before PSO | 1,500    | 96.95%   | 86.62%    | 91.28% | 88.89%   |
| LR After PSO  | 762      | 97.49%   | 89.54%    | 91.95% | 90.73%   |

> PSO reduced features by **51.8%** for Naive Bayes and **49.2%** for Logistic Regression while improving performance for both classifiers.

---

## PSO Configuration

| Parameter            | Value |
|----------------------|-------|
| Number of particles  | 40    |
| Number of iterations | 60    |
| Inertia weight (w)   | 0.5   |
| C1 — cognitive       | 2.0   |
| C2 — social          | 2.0   |
| Fitness function     | 0.7 × Accuracy + 0.29 × F1 + 0.01 × (1 - features/total) |

---

## Project Structure

```
SMS-Spam-Detection-PSO/
│
├── README.md
├── requirements.txt
├── notebook/
│   └── spam_detection.ipynb
└── figures/
    ├── Fig1_class_distribution.png
    ├── Fig2_avg_length.png
    ├── Fig3_smote.png
    ├── Fig_pso_convergence.png
    ├── Fig_feature_reduction.png
    ├── Fig_performance_comparison.png
    └── Fig_confusion_matrices.png
```

---

## Technologies

| Library          | Purpose                              |
|------------------|--------------------------------------|
| Python 3.10      | Programming language                 |
| Scikit-learn     | TF-IDF, classifiers, metrics         |
| NLTK             | Stopwords removal and stemming       |
| Imbalanced-learn | SMOTE implementation                 |
| NumPy / Pandas   | Data processing                      |
| Matplotlib / Seaborn | Visualization                   |

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/SMS-Spam-Detection-PSO
cd SMS-Spam-Detection-PSO
```

**2. Install requirements**
```bash
pip install -r requirements.txt
```

**3. Open the notebook**
```bash
jupyter notebook notebook/spam_detection.ipynb
```

**4. Enter your Kaggle credentials when prompted to download the dataset**

---

## Citation

If you use this work, please cite:  
DOI: *Add Zenodo DOI here after publishing*