from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("final_model.pkl", "rb"))
model2 = pickle.load(open("final_model2.pkl", "rb"))  # Regression
scaler = pickle.load(open("scaler.pkl","rb"))
@app.route('/')
def home():
    return render_template('index2.html', prediction='')


@app.route('/predict', methods=['POST'])
def predict():
    Age = float(request.form['Age'])
    AnnualIncome= float(request.form['AnnualIncome'])
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


    # Create dataframe for prediction
    input_data = pd.DataFrame([[
    Age,AnnualIncome, CreditScore, EducationLevel, LoanAmount,LoanDuration,BankruptcyHistory
    ,PreviousLoanDefaults,LengthOfCreditHistory,TotalAssets,MonthlyIncome,NetWorth,InterestRate,MonthlyLoanPayment]],
    columns=['Age', 'AnnualIncome', 'CreditScore', 'EducationLevel',
    'LoanAmount', 'LoanDuration','BankruptcyHistory','PreviousLoanDefaults','LengthOfCreditHistory','TotalAssets','MonthlyIncome',
    'NetWorth','InterestRate','MonthlyLoanPayment'])
    
    #scaling
    input_data_reg=scaler.transform(input_data)
    input_data_reg
    
    #encoding
    input_data['EducationLevel']=input_data['EducationLevel'].map({'High School':0,'Associate':1,'Bachelor':2,'Master':3,'Doctorate':4})
    

    # classification Predict
    result_cls = model.predict(input_data)
    # regression Predict
    result_reg = model2.predict(input_data_reg)
    
    if result_cls == 0:
        Loan_Approval_status = 'Loan Will Not Approve'
    elif result_cls == 1 :
        Loan_Approval_status = 'Loan Will Approve'
    else: 
        Loan_Approval_status = 'Invalid' 
        
    prediction_text = f" Predicted Loan Status: {Loan_Approval_status} \n RiskScore: {result_reg[0]}"
    return render_template('index2.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
