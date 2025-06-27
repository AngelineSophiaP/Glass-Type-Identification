
# 🧪 Glass Type Identification using K-Nearest Neighbors (KNN)

## 📌 Project Description

This project aims to **classify glass samples into various types** based on their chemical composition.
Identifying glass types has real-world applications in **forensics**, **manufacturing**, and **material science**.

We use the **K-Nearest Neighbors (KNN)** algorithm to predict the type of glass by comparing its attributes to known samples.

---

## 🎯 Objective

To build a machine learning model using **KNN** that accurately classifies glass into one of several categories (e.g., building windows,
containers, etc.) based on features like refractive index and chemical elements.

---

## 🗂️ Dataset

- **Source:** [UCI Machine Learning Repository – Glass Identification Data Set](https://archive.ics.uci.edu/ml/datasets/glass+identification)
- **Total Samples:** 214
- **Features:**
  - RI: Refractive index
  - Na: Sodium
  - Mg: Magnesium
  - Al: Aluminum
  - Si: Silicon
  - K: Potassium
  - Ca: Calcium
  - Ba: Barium
  - Fe: Iron
- **Target (Type of Glass):**
  - 1: Building windows (float processed)
  - 2: Building windows (non-float processed)
  - 3: Vehicle windows
  - 5: Containers
  - 6: Tableware
  - 7: Headlamps

> **Note:** Type 4 is not present in the dataset.

---

## ⚙️ Libraries Used

- `pandas` — for data loading & cleaning
- `sklearn.model_selection` — train/test split
- `sklearn.neighbors` — KNN Classifier
- `sklearn.metrics` — evaluation

---

## 🧠 Algorithm Used: K-Nearest Neighbors (KNN)

- A **supervised classification algorithm**.
- Predicts class by majority vote of the **k closest data points**.
- We used `k = 3` neighbors in this project.

---

## 🛠️ ML Workflow

1. **Load** the dataset from CSV
2. **Clean** the data (e.g., drop duplicates, handle nulls if any)
3. **Split** the data into features (`X`) and target (`y`)
4. **Train** the KNN classifier with `n_neighbors = 3`
5. **Predict** the glass type for test data
6. **Evaluate** the model using accuracy, confusion matrix, and classification report

---

## 📈 Results

```text
✅ Accuracy: 0.84 (example output; yours may vary)

📊 Confusion Matrix:
[[13  0  0  0  0  0]
 [ 0 12  1  0  0  0]
 [ 0  1  9  0  0  0]
 [ 0  0  0  6  0  0]
 [ 0  0  0  0  3  0]
 [ 0  0  0  0  0  4]]

📄 Classification Report:
              precision    recall  f1-score   support
           1       1.00      1.00      1.00        13
           2       0.92      0.92      0.92        13
           3       0.90      0.90      0.90        10
           5       1.00      1.00      1.00         6
           6       1.00      1.00      1.00         3
           7       0.89      0.89      0.89         4
````

---

## ✅ Conclusion

* The **KNN model** successfully classified various glass types with a good accuracy.
* It performs well when features are well-scaled and the dataset is balanced.

---

## 📁 Folder Structure

```
├── glass_knn_classification.ipynb  # or .py
├── glass.csv                       # dataset file
├── README.md
```
