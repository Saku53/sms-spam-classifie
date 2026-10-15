An end-to-end Machine Learning pipeline built using Python, Scikit-Learn, and Pandas to classify SMS messages as either **Spam** or **Ham** (Legitimate).

## 🚀 Project Overview

This repository hosts an NLP-based spam detector utilizing a **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer** for text representation, paired with a **Logistic Regression** model.

## 🔄 Project Pipeline

```text
spam.csv
   ↓
train_model.py
   ↓
Text normalization + TF-IDF + handcrafted features
   ↓
SVM / Logistic Regression / Ensemble
   ↓
Evaluation
   ↓
backend/model.joblib + backend/metrics.json
   ↓
FastAPI /predict endpoint
```

## 🤖 Models and Training Choices

### Model 1: Calibrated SVM

```python
svm_cal = CalibratedClassifierCV(
    LinearSVC(class_weight="balanced"),
    cv=3
)
```

The base classifier is `LinearSVC`. SVMs are often strong for text classification because text produces very high-dimensional sparse vectors. A linear SVM finds a boundary separating ham and spam in that feature space.

#### Why calibration is used

A normal `LinearSVC` does not naturally provide probabilities; it provides decision scores. However, this project needs probabilities for:

- ROC curves
- Threshold selection
- Displaying spam probabilities through the API

`CalibratedClassifierCV` converts the SVM decision scores into probability-like values.

#### `class_weight="balanced"`

The dataset contains more ham messages than spam messages. `class_weight="balanced"` gives more importance to the minority spam class during training.

### Model 2: Logistic Regression

```python
LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)
```

Logistic Regression is a standard and effective baseline for text classification. It produces probabilities naturally, which makes it convenient for threshold-based classification.

The code uses:

```python
max_iter=2000
```

because TF-IDF data can be large and the model may require more iterations to converge.

### Model 3: Soft-Voting Ensemble

```python
VotingClassifier([
    ("svm", CalibratedClassifierCV(...)),
    ("lr", LogisticRegression(...)),
], voting="soft")
```

This combines the SVM and Logistic Regression probabilities:

1. The SVM produces a spam probability.
2. Logistic Regression produces a spam probability.
3. The ensemble combines both probabilities.

The idea is that the models may make different mistakes. Combining them can sometimes produce a more stable result.

## 🔧 Hyperparameter Tuning

The code uses `GridSearchCV`:

```python
gs = GridSearchCV(
    pipeline,
    grid or {},
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True
)
```

The cross-validation object is:

```python
cv = StratifiedKFold(
    5,
    shuffle=True,
    random_state=SEED
)
```

This is five-fold stratified cross-validation. The training set is divided into five parts, and the model is trained and validated five times:

```text
Round 1: train on parts 2–5, validate on part 1
Round 2: train on parts 1, 3, 4, 5, validate on part 2
Round 3: train on parts 1, 2, 4, 5, validate on part 3
Round 4: train on parts 1, 2, 3, 5, validate on part 4
Round 5: train on parts 1–4, validate on part 5
```

This provides a more reliable estimate than using only one validation split.

### Hyperparameter values

For the calibrated SVM:

```python
{"clf__estimator__C": [0.3, 1, 3]}
```

For Logistic Regression:

```python
{"clf__C": [3, 10, 30]}
```

`C` controls regularization:

- Smaller `C` means stronger regularization.
- Larger `C` allows the model to fit the training data more closely.

### What metric selects the model?

The code uses:

```python
scoring="f1"
```

Therefore, the best hyperparameter/model combination is selected according to F1-score rather than accuracy. This is sensible because the dataset is imbalanced: ham messages greatly outnumber spam messages, so accuracy alone can be misleading.

## 🏆 Choosing the Final Model

After evaluating all three model types, the code chooses:

```python
best_name = max(results, key=lambda n: results[n]["cv_f1"])
```

The final model is selected using the best cross-validation F1-score.

The test set is not used to select the model:

```python
# Choose the model by cross-validation F1 (NOT by looking at the test set)
```

This is important because the test set should be held back until the final evaluation. Choosing a model by its test-set result could cause test-set overfitting and make the reported performance less trustworthy.

## 🎚️ Threshold Selection

Normally, a binary classifier labels a message as spam when:

```text
spam probability >= 0.5
```

This project does not blindly use `0.5`. Instead, it calculates out-of-fold predictions on the training data:

```python
oof = cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]
```

It then chooses a threshold using:

```python
thr = pick_threshold(
    y_train,
    oof,
    min_recall=0.97
)
```

The threshold function searches for the highest-precision threshold that still maintains at least 97% recall on the training data:

```python
if ri >= min_recall and pi > best_prec:
    best_prec, best_thr = pi, ti
```

In simple terms:

> Find a threshold that catches at least 97% of spam messages while producing as few false positives as possible.

This is important because incorrectly labeling a legitimate message as spam can be harmful. The API later uses the selected threshold:

```python
"spam" if prob >= THRESHOLD else "ham"
```

Therefore, the final decision is based on the selected threshold and not necessarily on `0.5`.

## 📏 Evaluation Metrics

The `metrics_at()` function calculates:

```python
accuracy
precision
recall
f1
mcc
balanced_acc
roc_auc
pr_auc
```

### Accuracy

```text
Accuracy = correct predictions / all predictions
```

For example, if 1,080 out of 1,115 messages are correctly classified:

```text
accuracy = 1,080 / 1,115
```

Accuracy is easy to understand, but it can be misleading when the classes are imbalanced.

### Precision

For spam:

```text
Precision = correctly predicted spam /
            all messages predicted as spam
```

High spam precision means that when the model says “spam”, it is usually correct. Precision is related to avoiding false positives.

### Recall

For spam:

```text
Recall = correctly detected spam /
         all actual spam messages
```

High recall means that the model catches most spam messages. A model can have high precision but lower recall if it only labels a message as spam when it is very confident.

### F1-score

F1 combines precision and recall:

```text
F1 = 2 × precision × recall /
     (precision + recall)
```

F1 is useful when both false positives and false negatives matter. This project uses F1 to choose the best model during cross-validation.

### Balanced accuracy

Balanced accuracy averages performance across the two classes:

```text
Balanced accuracy = (ham recall + spam recall) / 2
```

This is more informative than normal accuracy when the classes are imbalanced.

### Matthews correlation coefficient

Matthews correlation coefficient, or MCC, is a balanced classification metric that considers all four confusion-matrix values: true positives, true negatives, false positives, and false negatives. It is useful when one class is much more common than the other.

### ROC-AUC

ROC-AUC measures how well the model ranks spam messages above ham messages across different classification thresholds. An AUC close to `1.0` indicates excellent separation, while an AUC close to `0.5` is similar to random guessing.

### PR-AUC

PR-AUC means the area under the Precision-Recall curve. It is especially useful for this project because spam is the minority class and the curve focuses on the positive class.

## 📊 Model Performance

The classifier was validated using **5-Fold Cross-Validation** and evaluated on a holdout test dataset (20% of the original dataset):

- **5-Fold Cross-Validation Mean Accuracy:** ~95.69%
- **Test Set Accuracy:** **96.77%**

### Classification Metrics

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Ham** (Legitimate) | 0.96 | 1.00 | 0.98 | 966 |
| **Spam** | 1.00 | 0.76 | 0.86 | 149 |

*Key takeaway: The classifier achieved **100% Precision** for spam detection, meaning it is exceptionally safe from producing false positives (falsely labeling legitimate messages as spam).* 

### Visual Representations

To better understand the model's reliability, view our evaluation plots below:

#### 1. Confusion Matrix
Looks at the exact prediction layout. Out of 149 true spam items, 113 were correctly identified, and crucially, 0 legitimate ham messages were misclassified as spam.

![Confusion Matrix](images/confusion_matrix.png)

#### 2. ROC Curve
Displays performance trade-offs across all thresholds, achieving an excellent Area Under the Curve (AUC) score.

![ROC Curve](images/roc_curve.png)

## 🛠️ How to Use

You can easily use the predictive pipeline via the following Python snippet:

```python
# Sample inference function
def predict_message(message):
    prediction = model_pipeline.predict([message])[0]
    probability = model_pipeline.predict_proba([message])[0]
    label = 'Spam' if prediction == 1 else 'Ham'
    confidence = probability[prediction] * 100
    return f"Message: '{message}' -> Classified as: {label} ({confidence:.2f}% confidence)"
```

## 📂 Repository Structure
- `spam.csv`: The raw message dataset.
- `train_model.py`: Training script and pipeline builder.
- `backend/`: FastAPI backend, saved model, and metrics.
- `images/`: Exported performance evaluation plots.
- `README.md`: Project summary and performance log.
