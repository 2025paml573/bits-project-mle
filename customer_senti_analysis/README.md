**Customer Sentiment Preprocessing and Feature Extraction**

**Overview**

This project processes customer review data for sentiment analysis. It reads customer sentiment data from a CSV file, cleans and normalizes the review text, extracts NLP-based features, and prepares the data for training and testing a sentiment classification model.

**The code uses:**

* pandas for reading and manipulating CSV data
* numpy for numerical operations
* nltk for natural language processing
* re for regular expression-based text cleaning
* random for shuffling the dataset

**Short Summary** - 
This notebook preprocesses customer review text for sentiment analysis. It cleans hashtags, handles, URLs, emoticons, punctuation, and repeated characters, then converts each review into NLP features such as word presence, positive word indicators, and negation-based features. The processed data is shuffled and split into training and test sets using a 5-fold style split, preparing it for a future sentiment classification model.


# Training the Sentiment Analysis Model

## Overview

After preprocessing and feature extraction, the next step is to train a machine learning model that can classify customer reviews as **Positive**, **Negative**, or **Neutral** sentiment.

The training process uses the preprocessed review text and extracted NLP features to learn patterns associated with each sentiment category.

---

## Training Workflow

### 1. Load Processed Data

Load the preprocessed and shuffled review dataset where each review is associated with a sentiment label.

**Input:**
- Preprocessed customer reviews
- Sentiment labels (Positive, Negative, Neutral)

---

### 2. Extract Features

Convert each review into a feature dictionary using the following feature extraction techniques:

#### Word Presence Features
Captures important words present in the review.

Example:

```python
{
    'has(price)': 1,
    'has(delivery)': 1
}
```

#### Negation Features

Identifies negation words and their influence on nearby words.

Example:

```python
{
    'neg_l(worth)': 0.9,
    'neg_l(price)': 0.8
}
```

#### Positive Sentiment Features

Identifies strongly positive words.

Example:

```python
{
    'pos(great)': 1,
    'pos(amazing)': 1
}
```

---

### 3. Split Dataset

The dataset is divided into training and testing subsets using a 5-fold partitioning approach.

Configuration:

```python
K = 5
k = 1
```

Resulting split:

- Training Data: ~80%
- Test Data: ~20%

Example:

```text
length of train reviews 20001
length of test reviews 5000
```

---

### 4. Train the Classifier

Train a machine learning classifier using the extracted features.

Recommended baseline model:

- NLTK Naive Bayes Classifier

Example:

```python
from nltk import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(train)
```

Alternative models may include:

- Maximum Entropy Classifier
- Decision Tree Classifier
- Support Vector Machine (SVM)
- Logistic Regression

---

### 5. Evaluate Model Performance

Evaluate the trained model using the test dataset.

Example:

```python
accuracy = nltk.classify.accuracy(classifier, test)
print("Accuracy:", accuracy)
```

Common evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics help determine how well the model predicts customer sentiment.

---

### 6. Predict New Reviews

Once trained, the model can predict the sentiment of new customer reviews.

Examples:

| Review | Predicted Sentiment |
|----------|-------------------|
| Fast delivery and great packaging. | Positive |
| Not worth the price. | Negative |
| Delivery was fine, product is decent. | Neutral |

Example prediction:

```python
features = extract_features(review_words)
classifier.classify(features)
```

---

## Business Value

The trained sentiment analysis model can be used to:

- Monitor customer satisfaction
- Analyze product feedback
- Identify recurring customer complaints
- Track sentiment trends over time
- Improve customer support processes
- Automatically categorize customer reviews

---

## Deliverables

The training task produces:

### Trained Model
A machine learning model capable of predicting customer sentiment.

### Test Results
Performance metrics including:

- Accuracy
- Precision
- Recall
- F1-Score

### Feature Dataset

Feature vectors generated from:

- Word presence
- Negation detection
- Positive sentiment indicators

### Prediction Capability

The ability to classify new customer reviews as:

- Positive
- Negative
- Neutral

---

## Example Training Code

```python
from nltk import NaiveBayesClassifier
import nltk

# Train model
classifier = NaiveBayesClassifier.train(train)

# Evaluate model
accuracy = nltk.classify.accuracy(classifier, test)

print("Model Accuracy:", accuracy)

# Display most informative features
classifier.show_most_informative_features(20)
```

---

## End-to-End Flow

```text
Customer Reviews
        ↓
Preprocessing
        ↓
Feature Extraction
        ↓
Train/Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Sentiment Prediction
        ↓
Business Insights
```
