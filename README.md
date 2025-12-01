Loan Risk Assessment

**Overview**

This project performs a complete loan risk assessment analysis using a synthetic dataset. It includes data exploration, preprocessing, visualization, and machine-learning model development to predict loan approval outcomes and understand key risk factors.
The notebook walks through a full data-science workflow, making it useful for students, analysts, and machine-learning beginners.

**Dataset Description**

The dataset contains detailed applicant, financial, and loan-related information such as:

ApplicationDate – Date of the loan application

LoanAmount, InterestRate, LoanPurpose, LoanStatus

AnnualIncome, CreditScore, EmploymentLength

State, City, Address

DTI (Debt-to-Income), OpenAccounts, DelinquentAccounts

Other demographic & financial indicators

These features support both exploratory analysis and predictive modeling.

**Methodology**

1. Data Exploration

Summary statistics, missing value detection

Univariate & multivariate analysis

Outlier identification

Distribution plots, count plots, boxplots

2. Data Visualization

Heatmaps for correlation

Categorical visualizations

Financial metric comparisons

Loan approval trend analysis

3. Data Preprocessing

Handling missing values

Encoding categorical features

Scaling numerical variables

Detecting and removing outliers

Train-test splitting

4. Model Development

ML models implemented include:

Logistic Regression

Decision Tree Classifier

Random Forest

XGBoost / LightGBM (if installed)

Models are evaluated using:

Accuracy, Precision, Recall, F1-Score

Confusion Matrix

5. Insights & Interpretation

Key factors influencing loan approval

Applicant attributes linked to high risk

Feature importance rankings

Trends across credit score, income, DTI, loan purpose, etc.

**Key Insights (Example)**

Higher CreditScore and AnnualIncome significantly increase approval chances.

Higher DTI and DelinquentAccounts strongly correlate with loan rejection.

Certain loan purposes (e.g., debt consolidation) show higher risk patterns.

Random Forest provided the best predictive performance in this analysis.

**Conclusion**

This project demonstrates a complete workflow for building a loan approval prediction system, starting from raw data, performing in-depth analysis, and developing machine-learning models.
The approach can be extended to real-world datasets for financial analytics, credit risk scoring, or automated loan decision systems.
