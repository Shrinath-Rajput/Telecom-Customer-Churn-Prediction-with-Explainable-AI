# 📊 Telecom Customer Churn Prediction with Explainable AI

## 🚀 Overview

This project predicts whether a telecom customer will **stay or churn** using Machine Learning.
It is a complete **end-to-end ML + Full Stack application**.

---

## 🧠 Problem Statement

Telecom companies face customer churn.
This system helps predict which customers are likely to leave so companies can take preventive action.

---

## ⚙️ Tech Stack

* **Frontend:** HTML, CSS, JavaScript, Bootstrap
* **Backend:** Node.js (Express)
* **Machine Learning:** Python, Scikit-learn
* **Experiment Tracking:** MLflow
* **Deployment:** Docker

---

## 🏗️ Project Structure

```id="x7m1c3"
.
├── .venv/
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── raw.csv
│   ├── train.csv
│   └── test.csv
│
├── Dataset/
│   └── telo.csv
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── model_api/
│   └── predict.py
│
├── server/
│   └── index.js
│
├── src/
│   ├── Components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── Pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── mlruns/
├── logs/
├── Dockerfile
├── mlflow.db
└── requirements.txt
```

---

## 🔄 Workflow

1. Data is collected and stored in Dataset
2. Data ingestion splits data into train/test
3. Data transformation applies preprocessing (scaling, encoding)
4. Model training selects the best model
5. MLflow tracks experiments
6. Frontend sends user input to backend
7. Backend calls Python model
8. Model predicts churn (Yes/No)

---

## 📊 Machine Learning

* Models Used:

  * Logistic Regression ✅ (Best)
  * Random Forest
* Metrics:

  * Accuracy
  * Precision
  * Recall
  * F1 Score

---

## ▶️ Run Locally

### 1️⃣ Install dependencies

```id="n2xgdr"
pip install -r requirements.txt
cd server
npm install
```

### 2️⃣ Run Backend

```id="k8rb5k"
node index.js
```

### 3️⃣ Open in Browser

```id="9oyx3q"
http://localhost:3000
```

---

## 🐳 Run with Docker

```id="mttzwh"
docker build -t churn-app .
docker run -p 3000:3000 churn-app
```

---

## 📈 Features

* End-to-end ML pipeline
* Real-time prediction
* MLflow experiment tracking
* Full-stack integration
* Docker support

---

## 🧾 Resume Highlight

Developed an end-to-end **Telecom Customer Churn Prediction system** using Machine Learning and full-stack technologies with real-time prediction capability.

---

## 🙌 Author

**Shrinath Rajput**

---

## ⭐ Star this repo if you like it!
