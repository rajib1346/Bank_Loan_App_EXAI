from flask import Flask, render_template, request
import pickle
import numpy as np
import lime
import lime.lime_tabular
import pandas as pd

app = Flask(__name__)

# Load the model
model_path = "C:/Users/acer/Bank Loan/Models/model.pkl"
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the training data for LIME explainer background
train_data_path = "C:/Users/acer/Bank Loan/Models/train.csv"  # Adjust the path accordingly
train_data = pd.read_csv(train_data_path)

# LIME explainer setup
explainer = lime.lime_tabular.LimeTabularExplainer(
    train_data[['Dependents', 'Education', 'LoanAmount', 'Credit_History', 'Property_Area', 'Total_Income']].values,
    feature_names=['Dependents', 'Education', 'LoanAmount', 'Credit_History', 'Property_Area', 'Total_Income'],
    class_names=['Rejected', 'Approved'],
    mode='classification'
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collecting input data from the form
        name = request.form['Name']
        gender = request.form['Gender']
        age = int(request.form['Age'])
        dependents = int(request.form['Dependents'])
        education = int(request.form['Education'])
        loan_amount = float(request.form['LoanAmount'])
        credit_history = int(request.form['Credit_History'])
        property_area = int(request.form['Property_Area'])
        total_income = float(request.form['Total_Income'])

        # Model input
        input_features = np.array([[dependents, education, loan_amount, credit_history, property_area, total_income]])

        # Model prediction
        prediction = model.predict(input_features)[0]

        # Convert prediction result to binary and map to "Approved" or "Rejected"
        result = 'Approved' if prediction == 1 else 'Rejected'

        # LIME explanation
        exp = explainer.explain_instance(input_features[0], model.predict_proba, num_features=5)
        
        # Generate the LIME explanation HTML
        explanation_html = exp.as_html()

        # Pass the additional inputs (name, gender, age) to the result page
        return render_template('result.html', result=result, explanation_html=explanation_html, name=name, gender=gender, age=age)
    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(debug=False)
