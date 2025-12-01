from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load models and scaler
model = pickle.load(open("final_model.pkl", "rb"))      # Classification model
model2 = pickle.load(open("final_model2.pkl", "rb"))    # Regression model
scaler = pickle.load(open("scaler.pkl", "rb"))          # Scaler for regression model


@app.route('/')
def home():
    return render_template('index2.html',
                           loan_Approval_status='',
                           risk_score='')


@app.route('/predict', methods=['POST'])
def predict():
    # Collect user input
    Age = float(request.form['Age'])
    AnnualIncome = float(request.form['AnnualIncome'])
    CreditScore = float(request.form['CreditScore'])
    EducationLevel = request.form['EducationLevel']
    LoanAmount = float(request.form['LoanAmount'])
    LoanDuration = float(request.form['LoanDuration'])
    BankruptcyHistory = float(request.form['BankruptcyHistory'])
    PreviousLoanDefaults = float(request.form['PreviousLoanDefaults'])
    LengthOfCreditHistory = float(request.form['LengthOfCreditHistory'])
    TotalAssets = float(request.form['TotalAssets'])
    MonthlyIncome = float(request.form['MonthlyIncome'])
    NetWorth = float(request.form['NetWorth'])
    InterestRate = float(request.form['InterestRate'])
    MonthlyLoanPayment = float(request.form['MonthlyLoanPayment'])

    # Create DataFrame
    input_data = pd.DataFrame([[
        Age, AnnualIncome, CreditScore, EducationLevel, LoanAmount, LoanDuration,
        BankruptcyHistory, PreviousLoanDefaults, LengthOfCreditHistory, TotalAssets,
        MonthlyIncome, NetWorth, InterestRate, MonthlyLoanPayment
    ]], columns=[
        'Age', 'AnnualIncome', 'CreditScore', 'EducationLevel', 'LoanAmount',
        'LoanDuration', 'BankruptcyHistory', 'PreviousLoanDefaults',
        'LengthOfCreditHistory', 'TotalAssets', 'MonthlyIncome', 'NetWorth',
        'InterestRate', 'MonthlyLoanPayment'
    ])

    # Encode Education level for classification model
    input_data['EducationLevel'] = input_data['EducationLevel'].map({
        'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4
    })

    # Prepare scaled data for regression model
    input_data_scaled = scaler.transform(input_data)

    # Classification prediction
    result_cls = model.predict(input_data)[0]

    # Regression prediction (risk score)
    risk_score = round(float(model2.predict(input_data_scaled)[0]), 2)

    # Convert classification output to text
    if result_cls == 0:
        loan_Approval_status = "Loan Will NOT Approve"
    else:
        loan_Approval_status = "Loan Will Approve"

    # Send data to UI
    return render_template(
        'index2.html',
        loan_Approval_status=loan_Approval_status,
        risk_score=risk_score
    )


if __name__ == "__main__":
    app.run(debug=True)
