Loan Approval Prediction using Machine Learning
📌 Project Overview

This project implements an end-to-end machine learning pipeline to predict loan approval outcomes based on applicant demographics, financial attributes, and loan characteristics. The primary objective is to build a realistic, reliable, and interpretable model while addressing real-world challenges such as class imbalance, missing data, and data leakage.

Rather than focusing only on high accuracy, this project emphasizes correct ML methodology, robust evaluation, and alignment with real-world lending logic.

📊 Dataset Description

Total records: ~148,000 loan applications

Features: 30+ numerical and categorical variables

Target variable: Status

0 → Loan Rejected

1 → Loan Approved

Class distribution:

~75% rejected

~25% approved (moderately imbalanced dataset)

The dataset includes applicant financial information (income, credit score), loan details (loan amount, LTV), and categorical attributes such as loan type, region, and occupancy.

🔍 Exploratory Data Analysis (EDA)

Initial analysis revealed:

Presence of both numerical and categorical features

Missing values in several financial attributes

Moderate class imbalance in the target variable

EDA was performed to understand feature distributions, missing-value patterns, and target imbalance before model development.

🧹 Data Preprocessing

The following preprocessing steps were applied:

1. Missing Value Handling

Numerical features: Median imputation (robust to outliers)

Categorical features: Mode (most frequent category) imputation

2. Encoding

Categorical variables were converted to numerical form using One-Hot Encoding

Feature transformations were applied using scikit-learn Pipelines to prevent data leakage

3. Data Leakage Detection & Prevention

During experimentation, unrealistically perfect model performance was observed. This led to the identification of data leakage caused by:

Preprocessing performed before train–test splitting

Inclusion of post-decision features (e.g., interest rates, charges, debt ratios)

These leaky features were removed, and all preprocessing steps were refit only on training data, ensuring realistic generalization.

🤖 Models Implemented
1. Logistic Regression (Baseline Model)

Used as an interpretable baseline

Class imbalance handled using class weighting

Achieved strong recall for approved loans

2. Random Forest Classifier (Final Model)

Captures non-linear relationships

More conservative in approving loans

Demonstrated robust performance and stability

📈 Model Performance
Logistic Regression

ROC–AUC ≈ 0.87

Higher recall for approved loans (minority class)

Suitable for scenarios prioritizing customer inclusion

Random Forest

Accuracy: ~0.88

ROC–AUC: ~0.88

Recall (Approved loans): ~0.61

Conservative model prioritizing risk minimization

These results represent realistic performance for a real-world loan approval problem.

🔑 Feature Importance Analysis

Feature importance analysis using the Random Forest model showed that the most influential factors included:

Credit score

Loan amount

Applicant income

Loan-to-value (LTV) ratio

Property and occupancy characteristics

The importance rankings align well with real-world lending and underwriting practices.

🧠 Key Learnings

Perfect accuracy is often an indicator of data leakage

Correct ordering of preprocessing and train–test split is critical

Evaluation metrics beyond accuracy are essential for imbalanced datasets

Model choice depends on business objectives (risk vs inclusion)

🚀 Future Improvements

Decision-threshold tuning to improve minority-class recall

Hyperparameter optimization

Experimentation with gradient boosting models (XGBoost, LightGBM)

Optional deployment using a lightweight web interface (Streamlit)

🏁 Conclusion

This project demonstrates a methodologically sound machine learning workflow, from data exploration and leakage detection to model evaluation and interpretation. The focus on correctness, realism, and explainability makes the solution suitable for real-world deployment and further academic exploration.

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn
