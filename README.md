# 120-Years-of-Olympic-History-Athletes
Global AI Hub &amp; Akbank: Introduction to Machine Learning Bootcamp Project

# 🏅 Predicting Olympic Medal Winners with Machine Learning

This repository contains a machine learning project developed for the **Akbank & Global AI Hub: Introduction to Machine Learning Bootcamp**. The goal was to apply supervised learning techniques to predict Olympic medal winners based on physical and categorical attributes, and to optionally apply unsupervised learning to explore clusters among athletes.

---

## 📌 Problem Statement

Can we predict whether an athlete will win a medal based on their **age**, **physical attributes**, **sport**, **team**, and **year** of participation?

I framed this as a **binary classification** problem:
- **Target variable**: `Medal_Won` (1 = Medal, 0 = No Medal)

The dataset used was the [**120 Years of Olympic History: Athletes and Results**](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results) dataset from Kaggle, which includes over 200,000 athlete records (athlete_events.csv).

---

## 📊 Dataset Preprocessing

- Removed irrelevant columns (e.g., name, event, city)
- Created a binary label `Medal_Won` from the original `Medal` column
- Dropped rows with missing `Age`, `Height`, or `Weight` values
- Label-encoded categorical features (`Sex`, `Team`, `Sport`)
- Final dataset: **206,165 records** and 7 features

---

## 🤖 Supervised Learning Workflow

### 🔍 Models Evaluated
- Logistic Regression
- Decision Tree ✅ *(Selected)*
- Naive Bayes
- Random Forest 

### 📈 Model Selection & Tuning
Using 5-fold **Stratified Cross-Validation** and **F1-score** as the main metric due to class imbalance.

📉 **Class Imbalance Note**:
- Only **~14.6%** of the athletes in the dataset won a medal.
- This makes it an imbalanced classification task, where precision, recall, and F1-score are more meaningful than raw accuracy.
  
Model Evaluation (F1-score - 5-fold cross validation):

- Logistic Regression: 0.0008
- Decision Tree: 0.5175
- Random Forest: 0.4452
- Naive Bayes: 0.0645

- The **Decision Tree Classifier** gave the best F1-score.
- Applied **GridSearchCV** for hyperparameter tuning.

```python
# Best Parameters
{'max_depth': None, 'min_samples_split': 10}
