
# 💳 Online Payments Fraud Detection using Machine Learning

This project aims to detect fraudulent online transactions using a supervised machine learning model trained on transaction data.

---

## 🚀 Features

- ✅ **Random Forest Classifier** for accurate predictions  
- ✅ **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance  
- ✅ **Feature Importance Visualization** to understand model behavior  
- ✅ **Flask Web App** for real-time prediction  
- ✅ Easy Deployment using Render or localhost

---

## 🧾 Dataset Columns

- `step` – Time step in the simulation  
- `type` – Type of transaction (e.g., PAYMENT, TRANSFER)  
- `amount` – Amount of the transaction  
- `nameOrig`, `nameDest` – Origin and destination account names  
- `oldbalanceOrg`, `newbalanceOrig` – Originator's balance before/after transaction  
- `oldbalanceDest`, `newbalanceDest` – Receiver's balance before/after transaction  
- `isFraud` – Target column (1 = fraud, 0 = not fraud)

---

## 🛠️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model and save it
python main.py

# Launch the web app (after model is saved)
python app.py
```

Once running, open your browser and go to:  
`http://localhost:5000`

---

## 🌐 Deployment (Optional)

If you want to deploy this on Render:

1. Push your files to a public GitHub repo.
2. Go to [https://render.com](https://render.com) and create a new Web Service.
3. Set:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
4. Done! You'll get a live URL like `https://fraud-detector.onrender.com`

---

## 👩‍💻 Author

**Preethi Kethireddy**  
Raghu Engineering College  
[GitHub: @Masscoders00](https://github.com/Masscoders00)
