
# Diabetes Prediction Model using SVM

A machine learning project that predicts the likelihood of diabetes using a Support Vector Machine (SVM) classifier trained on the PIMA Indians Diabetes Dataset.

---

## 🧪 Project Overview

This project applies a Support Vector Machine (SVM) with a linear kernel to predict whether an individual is diabetic or not, based on medical features. The model is trained using the well-known **PIMA Diabetes Dataset**, which contains 768 patient records with 8 diagnostic features.

---

## 📂 Dataset

- **Source**: PIMA Indians Diabetes Data Set (UCI / Kaggle)
- **Samples**: 768
- **Features**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
- **Target**:  
  - `0` = Non‑Diabetic  
  - `1` = Diabetic

---

## 🚀 Project Workflow

1. Load and explore the dataset using **Pandas**
2. Preprocess:
   - Handle missing values (zeros)
   - Feature scaling with `StandardScaler`
3. Train–test split with **stratification**
4. SVM model training (linear kernel)
5. Evaluate on training and test sets (accuracy)
6. Test predictions on sample data inputs
7. (Optional) Serialize the model for external use

---

## 🛠️ Implementation Details

- **Environment**: Python 3.x
- **Libraries**:
  - pandas, numpy
  - scikit-learn (StandardScaler, train_test_split, SVC)
  - Matplotlib / Seaborn (exploratory visualization, optional)

---

## 📈 Model Results

- Training Accuracy: around **75–80%**
- Test Accuracy: comparable range on unseen data
- The model’s performance aligns with benchmarks reported in similar research projects.

---

## ⚙️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/hacker-knight/Diabetes-Prediction-Model-using-SVM.git
cd Diabetes-Prediction-Model-using-SVM
```

1. Setup a Python virtual environment  
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
2. Install requirements  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or script  
   ```bash
   jupyter notebook diabetes_prediction.ipynb
   ```
   Or execute a prediction script:
   ```bash
   python predict.py
   ```

---

## 🖥️ Usage Example

In a Python shell or script:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pickle

# Load trained scaler and classifier
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("svm_classifier.pkl", "rb"))

# Sample patient data
sample = np.array([[5, 166, 72, 19, 175, 25.8, 0.587, 51]])
scaled = scaler.transform(sample)
prediction = model.predict(scaled)

print("Diabetic" if prediction[0] == 1 else "Non‑Diabetic")
```

---

## 📊 Evaluation Metrics

- **Accuracy** on training and test datasets
- (Optional: Add confusion matrix, precision, recall, F1‑score if implemented)

---

## 🔧 Future Improvements

- Hyperparameter tuning using `GridSearchCV`
- Cross‑validation for robust evaluation
- Comparison with other classifiers (Logistic Regression, Random Forest, XGBoost)
- Save and deploy the model via a web app (e.g. Streamlit or Flask)
- GUI or web-based form for real-time input

---

## 👤 Author & Acknowledgments

- Developed by **hacker‑knight**
- Based on the UCI PIMA Diabetes Dataset and widely used SVM implementations

---

## 📃 License

This project is shared under the **MIT License**. See the `LICENSE` file for details.

---

Made with ❤️ to help healthcare prediction and ML enthusiasts.
