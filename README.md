# DDOS-ATTACK-DETECTOR
A Distributed Denial-of-Service (DDoS) Attack Detector is a sophisticated network security system designed to continuously monitor inbound and outbound network traffic to identify, characterize, and flag malicious activity associated with a DDoS attack. The core function is to distinguish between legitimate, high-volume traffic 
# DDoS Attack Detection using Machine Learning

## 📌 Overview

This project implements a Machine Learning-based DDoS Attack Detection System using the **CICIDS2017** dataset. The model performs data preprocessing, feature selection, class balancing using SMOTE, and trains a Random Forest classifier to classify network traffic as different attack types or benign traffic.

The goal is to accurately identify malicious network traffic and improve intrusion detection performance.

---

## 🚀 Features

- Data cleaning and preprocessing
- Duplicate record removal
- Correlation-based feature reduction
- Label encoding for attack classes
- Class balancing using SMOTE
- Random Forest classification
- Model evaluation using Accuracy and Classification Report

---

## 📂 Dataset

**Dataset Used:** CICIDS2017

The dataset should be stored locally and loaded as a Parquet file.

Example path used in the project:

```python
C:\DATASET\DATA\cicids2017_combined.parquet
```

You can download the CICIDS2017 dataset from the Canadian Institute for Cybersecurity:

https://www.unb.ca/cic/datasets/ids-2017.html

---

## 🛠 Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)

---

## 📦 Installation

Clone the repository

```bash
git clone https://github.com/your-username/ddos-attack-detection.git

cd ddos-attack-detection
```

Install dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn pyarrow
```

---

## ▶️ Running the Project

Simply run

```bash
python cic4.py
```

The program will

1. Load the dataset
2. Remove duplicate records
3. Remove highly correlated features
4. Encode attack labels
5. Split training/testing data
6. Apply SMOTE
7. Train Random Forest
8. Predict attack classes
9. Display Accuracy and Classification Report

---

## 🔄 Workflow

```
Load Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Remove Duplicate Rows
      │
      ▼
Correlation Analysis
      │
      ▼
Feature Selection
      │
      ▼
Label Encoding
      │
      ▼
Train-Test Split
      │
      ▼
SMOTE Oversampling
      │
      ▼
Random Forest Training
      │
      ▼
Prediction
      │
      ▼
Performance Evaluation
```

---

## 📊 Model

**Algorithm:** Random Forest Classifier

Parameters

- Number of Trees: 100
- Random State: 42
- Parallel Processing Enabled

---

## 📈 Evaluation Metrics

The model evaluates performance using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Classification Report

---

## 📁 Project Structure

```
DDoS-Attack-Detection/
│
├── cic4.py
├── README.md
└── dataset/
    └── cicids2017_combined.parquet
```

---

## 🔮 Future Improvements

- Deep Learning (LSTM/CNN)
- XGBoost and LightGBM comparison
- Real-time network traffic detection
- Flask/Django Web Dashboard
- Live packet capture using Scapy
- Model deployment using Docker

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Aayush Arya**

Student | Machine Learning & Cybersecurity Enthusiast

GitHub: https://github.com/aayusharya1202

---

## ⭐ If you found this project useful

Give the repository a ⭐ on GitHub.
