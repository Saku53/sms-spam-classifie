updated_readme_content = """# SMS Spam Classifier 📱

An end-to-end Machine Learning pipeline built using Python, Scikit-Learn, and Pandas to classify SMS messages as either **Spam** or **Ham** (Legitimate).

## 🚀 Project Overview

This repository hosts an NLP-based spam detector utilizing a **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer** for text representation, paired with a **Logistic Regression** model. The pipeline achieves state-of-the-art predictive performance, making it lightweight and highly reliable for real-time inference.

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
- `images/`: Exported performance evaluation plots.
- `README.md`: Project summary and performance log.
"""

with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(updated_readme_content)

print("README.md successfully updated with visual representations!")
