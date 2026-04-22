# requirements.txt

```
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
imbalanced-learn
opendatasets
jupyter
```

---

# README.md

```markdown
# SMS Spam Detection Using PSO-Based Feature Selection

A machine learning project that detects SMS spam messages using 
Particle Swarm Optimization (PSO) for feature selection combined 
with Naive Bayes and Logistic Regression classifiers.

---

## Dataset
SMS Spam Collection — UCI Machine Learning Repository  
Link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

- Total messages: 5,572
- Ham (legitimate): 4,825 (86.6%)
- Spam: 747 (13.4%)

---

## Methodology

1. Text Preprocessing — lowercasing, removing punctuation, stopword removal, stemming
2. Class Imbalance Handling — SMOTE
3. Feature Extraction — TF-IDF (1,500 features)
4. Feature Selection — Binary PSO (separate run for each classifier)
5. Classification — Naive Bayes and Logistic Regression
6. Evaluation — Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## Results

| Model         | Features | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|----------|-----------|--------|----------|
| NB Before PSO | 1,500    | 96.05%   | 81.07%    | 91.95% | 86.16%   |
| NB After PSO  | 723      | 97.04%   | 88.16%    | 89.93% | 89.04%   |
| LR Before PSO | 1,500    | 96.95%   | 86.62%    | 91.28% | 88.89%   |
| LR After PSO  | 762      | 97.49%   | 89.54%    | 91.95% | 90.73%   |

**PSO reduced features by 51.8% for NB and 49.2% for LR  
while improving performance for both classifiers.**

---

## PSO Configuration

| Parameter            | Value |
|----------------------|-------|
| Number of particles  | 40    |
| Number of iterations | 60    |
| Inertia weight (w)   | 0.5   |
| C1 (cognitive)       | 2.0   |
| C2 (social)          | 2.0   |

---

## Technologies

- Python 3.10
- Scikit-learn
- NLTK
- Imbalanced-learn
- NumPy, Pandas, Matplotlib, Seaborn

---

## How to Run

1. Clone the repository
```
git clone https://github.com/your-username/SMS-Spam-Detection-PSO
```

2. Install requirements
```
pip install -r requirements.txt
```

3. Open the notebook
```
jupyter notebook notebook/spam_detection.ipynb
```

4. When prompted, enter your Kaggle credentials to download the dataset

---

## Citation

If you use this work, please cite:  
DOI: [Add Zenodo DOI here after publishing]
```

---
