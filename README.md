# 📊 Telco Customer Churn Prediction using Machine Learning

## 🔍 Project Overview
This project focuses on predicting customer churn in a telecom company using Machine Learning techniques. Customer churn refers to customers who stop using a company’s service. By predicting churn in advance, companies can take steps to retain customers and reduce revenue loss.

This project uses multiple ML models and applies proper data preprocessing, feature scaling, class balancing, and hyperparameter tuning to achieve reliable and accurate results.

---

## 🎯 Objectives
- Analyze customer behavior using data analysis
- Preprocess and clean the dataset
- Handle class imbalance using SMOTE
- Train multiple machine learning models
- Optimize Random Forest using GridSearchCV
- Evaluate performance using confusion matrix and classification report
- Visualize important insights using graphs

---

## 📁 Dataset Information
- Dataset: Telco Customer Churn Dataset
- Target Variable: `Churn`
- Features include:
  - Customer demographics
  - Contract type
  - Internet services
  - Monthly & total charges
  - Payment method

---

## ⚙️ Technologies & Tools Used
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Jupyter Notebook / Google Colab

---

## 🧠 Machine Learning Models Used
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

---

## 🔁 Project Workflow

1. Load Dataset
2. Drop Unnecessary Columns (`customerID`)
3. Handle Missing Values
4. Encode Categorical Features
5. Split Data into Training & Testing
6. Balance Classes using SMOTE
7. Feature Scaling using StandardScaler
8. Train Multiple Models
9. Hyperparameter Tuning using GridSearchCV
10. Final Model Evaluation
11. Data Visualization

---

## ✅ Final Model
- Best Model: **Random Forest Classifier**
- Optimized using **GridSearchCV**
- Evaluation Metrics:
  - Accuracy
  - Confusion Matrix

---

## 📊 Graphs & Visualizations

### 1️⃣ Confusion Matrix
Displays true positives, false positives, true negatives, and false negatives of the churn prediction.
<img width="782" height="625" alt="image" src="https://github.com/user-attachments/assets/e40db264-3a4e-4c70-bfa0-9c19ec7bddae" />


### 2️⃣ Feature Importance Graph
Shows which features contribute the most to customer churn prediction.

### 3️⃣ Contract vs Churn Count Plot
Visualizes how churn is distributed across different contract types.

### 4️⃣ Monthly Charges vs Churn Boxplot
Shows how monthly charges differ between churned and non-churned customers.

---

## 📌 Results
- Random Forest performed the best among all models.
- Customers with **month-to-month contracts** show higher churn.
- **Higher monthly charges** are correlated with higher churn.
- Contract type, tenure, and charges are major churn factors.


## ⭐ Conclusion
This project successfully predicts customer churn using Machine Learning and provides valuable insights into customer behavior, helping businesses improve retention strategies.

