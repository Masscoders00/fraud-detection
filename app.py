# app.py

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("fraud_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form
        features = [float(x) for x in request.form.values()]
        final_input = np.array([features])
        
        prediction = model.predict(final_input)[0]
        result = "Fraudulent" if prediction == 1 else "Not Fraudulent"
    except Exception as e:
        result = f"Error in prediction: {str(e)}"
    
    return render_template('index.html', prediction_text=f'Transaction is likely: {result}')

if __name__ == "__main__":
    app.run(debug=True)
