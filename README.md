# ❤️ Heart Disease Prediction using Logistic Regression

🚑 This project predicts whether a person has a **defective heart (1)** or a **healthy heart (0)** using **Machine Learning (Logistic Regression)**.

---

## ⚙️ Workflow

1️⃣ **Heart Data** 📊 – Load dataset with 303 rows & 14 columns.
2️⃣ **Data Preprocessing** 🧹 – Clean and prepare the data.
3️⃣ **Train-Test Split** ✂️ – Split into training and testing sets.
4️⃣ **Logistic Regression Model** 🧠 – Train the model to learn patterns.
5️⃣ **Prediction** 🔮 – Feed new data to the model → it predicts heart condition.

---

## 📦 Importing Dependencies

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

## 📊 Dataset Overview

```python
# number of rows and columns
heart_data.shape
# Output: (303, 14)

# checking distribution of target variable
heart_data['target'].value_counts()
# Output:
# 1 --> 165 (Defective Heart ❤️‍🩹)
# 0 --> 138 (Healthy Heart 💚)
```

---

## 🎯 Model Accuracy

```python
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)
# Output: 0.85 ✅

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)
# Output: 0.82 ✅
```

---

## 🛠️ How to Run this Project

### 🔹 Clone the Repository

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

### 🔹 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Run the Code

```bash
python main.py
```

---

## ⭐ Support

If you found this repo useful, **don’t forget to star ⭐ the repository!** 🚀

---
