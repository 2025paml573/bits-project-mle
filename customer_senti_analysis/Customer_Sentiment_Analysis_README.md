# Customer Sentiment Analysis — AI/ML Certification Project

## Project Overview

This project implements an end-to-end **Customer Sentiment Analysis** solution for automatically classifying customer text into sentiment categories.

The solution covers the complete ML lifecycle:

- Raw customer sentiment data ingestion
- Text preprocessing and normalization
- Tokenization and stemming
- Processed and shuffled datasets
- Feature engineering
- Multiple classification models
- 5-fold evaluation
- Model serialization
- FastAPI REST inference
- Browser/Postman testing
- Docker packaging and deployment
- Prediction logging
- Concept-drift monitoring design
- Retraining trigger design

The project was developed using **Anaconda and JupyterLab**.

---

## 1. Problem Statement

A customer support or e-commerce platform receives a large volume of customer text. Manually analyzing every review is time-consuming and difficult to scale.

The objective is to build an ML pipeline that ingests raw customer text, engineers NLP features, trains classification models, exposes the trained models through REST APIs, and provides a monitoring/retraining design for changing customer language and topics.

---

## 2. Dataset

### Source

Kaggle Customer Sentiment Dataset:

https://www.kaggle.com/datasets/kundanbedmutha/customer-sentiment-dataset

Raw file:

```text
Customer_Sentiment.csv
```

The dataset is processed and stored under the `data/` directory.

---

## 3. Architecture

```text
Raw Customer Sentiment CSV
          |
          v
+-------------------------+
|      Preprocessing      |
| normalize / tokenize    |
|         / stem          |
+-----------+-------------+
            |
            v
+-------------------------+
| Processed + Shuffled    |
| Dataset                 |
+-----------+-------------+
            |
            v
+-------------------------+
| Feature Engineering     |
| word / positive /       |
| negation                |
+-----------+-------------+
            |
            v
+-------------------------+
| Model Training          |
| 5-fold evaluation       |
+-----------+-------------+
            |
            v
+-------------------------+
| Model Artifacts         |
| Serialized .pkl models  |
+-----------+-------------+
            |
    +-------+-------+-------+-------+-------+-------+
    |       |       |       |       |       |       |
    v       v       v       v       v       v
  Naive   MaxEnt    SVM  Decision   RTE    BERT*
  Bayes                 Tree
    |       |       |       |       |       |
    +-------+-------+-------+-------+-------+-------+
                            |
                            v
                   +------------------+
                   |    FastAPI       |
                   | REST Service     |
                   | Pydantic/Uvicorn |
                   +--------+---------+
                            |
                    +-------+-------+
                    |               |
                    v               v
              Local Python       Docker
              python api.py    sentiment-api
                    |               |
                    +-------+-------+
                            |
                            v
                   Browser / Postman
                            |
                            v
                    Prediction Logs
                            |
                            v
                    Drift Monitoring
                            |
                            v
                    Retraining Trigger
                            |
                            v
                 Review → Retrain → Validate

* BERT applies where implemented in the final submitted code.
```

---

## 4. Project Structure

```text
CustSentiAnalysis/
|
+-- api.py
+-- Dockerfile
+-- requirements.txt
|
+-- preprocess.ipynb
+-- preprocess_and_train.ipynb
+-- load_model_and_test.ipynb
|
+-- data/
|   +-- Customer_Sentiment.csv
|   +-- processed datasets
|   +-- shuffled datasets
|
+-- models/
|   +-- NaiveBayesClassifier_pkl_model.pkl
|   +-- MaxentClassifier_pkl_model.pkl
|   +-- SvmClassifier_pkl_model.pkl
|   +-- DecisiontreeClassifier_pkl_model.pkl
|   +-- RTEClassifier_pkl_model.pkl
|
+-- logs/
|
+-- README.md
```

Exact filenames in the final repository should be treated as authoritative.

---

## 5. Preprocessing — `preprocess.ipynb`

`preprocess.ipynb` performs the initial NLP preprocessing.

### Processing Flow

```text
Raw CSV
  |
  v
Load Dataset
  |
  v
Text Normalization
  |
  v
Tokenization
  |
  v
Stemming
  |
  v
Processed Dataset
  |
  v
Shuffling
  |
  v
Processed + Shuffled Dataset
```

### Main Activities

1. Load the original CSV dataset.
2. Normalize customer text.
3. Tokenize text.
4. Stem tokens.
5. Generate processed data.
6. Shuffle the processed records.
7. Store the resulting datasets under `data/`.

The purpose is to produce a consistent text representation for downstream feature engineering and model training.

---

## 6. Feature Engineering

The project uses sentiment-related lexical features including:

### Word Presence

Identifies words/tokens occurring in customer text.

### Positive Terms

Captures positive sentiment-related terms.

### Negation

Considers negation because the meaning of a sentiment term can change when it appears in a negated context.

### TF-IDF

TF-IDF was identified as an additional feature enhancement. If it is present in the final implementation, the fitted vectorizer and its exact configuration should be documented and reused during inference. If it is not in the submitted code, it should be treated as a future enhancement.

---

## 7. Model Training — `preprocess_and_train.ipynb`

`preprocess_and_train.ipynb` performs the training workflow.

### Training Flow

```text
Processed + Shuffled Data
          |
          v
Feature Engineering
          |
          v
5-Fold Evaluation
          |
   +------+------+------+------+------+
   |      |      |      |      |      |
   v      v      v      v      v      v
  NB    MaxEnt   SVM    DT     RTE   BERT*
   |      |      |      |      |      |
   +------+------+------+------+------+
          |
          v
Model Comparison
          |
          v
Serialized Model Artifacts
```

The notebook:

1. Loads processed/shuffled data.
2. Creates feature representations.
3. Trains multiple classifiers.
4. Performs 5-fold evaluation.
5. Compares model performance.
6. Records training/experiment information.
7. Serializes trained models.

### Models

| Model | Purpose |
|---|---|
| Naive Bayes | Probabilistic classification baseline |
| MaxEnt | Maximum Entropy classifier |
| SVM | Support Vector Machine classifier |
| Decision Tree | Tree-based classifier |
| RTE | RTE classifier |
| BERT | Transformer-based model path |

BERT should be described as implemented only where the final submitted source code contains its training and inference implementation.

---

## 8. Why Shuffle the Data?

The processed dataset is shuffled before training so the model is not influenced by the original record ordering.

Shuffling helps create a more randomized training input.

> Shuffling alone does not prevent overfitting. Overfitting must also be addressed through validation, model complexity controls, regularization, feature selection, and other appropriate techniques.

---

## 9. Five-Fold Evaluation

The project uses 5-fold evaluation.

```text
Dataset
   |
   +---- Fold 1
   +---- Fold 2
   +---- Fold 3
   +---- Fold 4
   +---- Fold 5
             |
             v
      Model Evaluation
```

Each fold can serve as the evaluation partition while the remaining folds are used for training. This provides repeated evaluation across multiple data partitions.

---

## 10. Model Artifacts

Trained models are serialized under:

```text
models/
```

Established artifacts include:

```text
NaiveBayesClassifier_pkl_model.pkl
MaxentClassifier_pkl_model.pkl
SvmClassifier_pkl_model.pkl
DecisiontreeClassifier_pkl_model.pkl
RTEClassifier_pkl_model.pkl
```

The serialized models bridge offline training and online inference.

### Important

Once the models have been generated, **`preprocess_and_train.ipynb` does not need to be rerun every time the API starts**.

The API loads the existing model artifacts.

---

## 11. FastAPI REST Service — `api.py`

`api.py` exposes the trained models through REST endpoints.

The service uses:

- FastAPI
- Pydantic
- Uvicorn
- Serialized model artifacts
- NLP feature extraction

### API Flow

```text
REST Client
    |
    v
Request Validation
    |
    v
Feature Extraction
    |
    v
Selected Model
    |
    v
Prediction
    |
    v
JSON Response
```

---

## 12. API Endpoints

Root endpoint:

```text
GET http://localhost:8000/
```

Established prediction endpoints:

```text
POST /predict/naivebayes
POST /predict/maxent
POST /predict/svm
POST /predict/decisiontree
POST /predict/rte
```

If the final `api.py` contains a BERT endpoint, document the exact path defined by that implementation.

---

## 13. Run the API Locally

Navigate to the project directory:

```bash
cd /Users/rairakesh/work/BitsPillaniProject/CustSentiAnalysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
python api.py
```

Open:

```text
http://localhost:8000/
```

Expected response structure:

```json
{
  "status": "API is running",
  "available_endpoints": [
    "/predict/naivebayes",
    "/predict/maxent",
    "/predict/svm",
    "/predict/decisiontree",
    "/predict/rte"
  ]
}
```

The exact response should match the submitted `api.py`.

---

## 14. API Prediction Example

### Request

```text
POST http://localhost:8000/predict/naivebayes
```

### JSON Payload

```json
{
  "text": "great value for the money"
}
```

### Flow

```text
Customer Text
     |
     v
Pydantic Validation
     |
     v
Feature Extraction
     |
     v
Naive Bayes Model
     |
     v
Sentiment Prediction
     |
     v
JSON Response
```

The same pattern applies to the other model-specific endpoints.

---

## 15. Input Validation and Edge Cases

The API should handle invalid input in a controlled way.

Examples:

### Empty text

```json
{
  "text": ""
}
```

### Missing field

```json
{}
```

### Malformed JSON

Invalid JSON should result in an HTTP error rather than terminating the application.

### Very short text

Examples:

```text
good
bad
```

These are valid inputs, although they contain limited textual information.

The exact validation behavior should follow the submitted `api.py`.

---

## 16. Docker Deployment

Docker packages the application into a portable runtime.

### Image

```text
sentiment-api
```

### Container

```text
sentiment-app-container
```

### Deployment Flow

```text
Developer Machine
       |
       v
   Dockerfile
       |
       v
 Docker Image
 sentiment-api
       |
       v
 Docker Container
 sentiment-app-container
       |
       v
 Port Mapping
 8000:8000
       |
       v
 Browser / Postman
```

---

## 17. Build Docker Image

Ensure Docker is running.

```bash
docker build -t sentiment-api .
```

The Docker image packages the runtime, dependencies, API code, NLP resources, and required model artifacts according to the submitted Dockerfile.

---

## 18. Run Docker Container

```bash
docker run -d -p 8000:8000 --name sentiment-app-container sentiment-api
```

Port mapping:

```text
Host:      8000
             |
             v
Container: 8000
```

Open:

```text
http://localhost:8000/
```

The same REST contract is intended to be used for local and containerized execution.

---

## 19. Local Python vs Docker

| Capability | Local Python | Docker |
|---|---|---|
| Runtime | Local Anaconda/Python | Container |
| Start | `python api.py` | `docker run ...` |
| API | FastAPI | FastAPI |
| Port | 8000 | 8000 mapped to host |
| Models | `.pkl` files | Packaged artifacts |
| Client | Browser/Postman | Browser/Postman |
| Portability | Local environment dependent | More portable/reproducible |

---

## 20. Prediction Logging

Prediction activity provides operational evidence for monitoring.

```text
FastAPI
   |
   v
Customer Request
   |
   v
Prediction
   |
   v
Prediction Logs
   |
   v
Monitoring
```

Potential monitoring signals include:

- Prediction volume
- Sentiment distribution
- Vocabulary changes
- Feature-distribution changes
- Incoming text characteristics
- Prediction error rates
- Model performance when labels become available
- API latency
- Throughput

The exact logging implementation should match the final submitted code.

---

## 21. Concept Drift

Customer language can evolve after model training.

Examples:

- New slang
- New product names
- New topics
- New abbreviations
- New customer terminology
- Changes in how customers express sentiment

A model trained on historical language may therefore become less effective over time.

---

## 22. Drift Monitoring Design

```text
Prediction Logs
       |
       v
Drift Monitoring
       |
       +-------------------------+
       |                         |
       v                         v
No Sustained Drift       Sustained Drift /
       |                  Performance Decline
       v                         |
Continue Monitoring              v
                            Review Trigger
                                  |
                                  v
                           Collect New Labels
                                  |
                                  v
                                Retrain
                                  |
                                  v
                               Validate
                                  |
                                  v
                            Promote Model
```

Potential signals:

1. Sustained feature-distribution drift.
2. Significant changes in sentiment distribution.
3. Performance degradation when labeled data is available.
4. Persistent prediction errors.
5. Vocabulary/topic changes.

A production implementation should establish thresholds using validation data and business requirements.

---

## 23. Retraining Strategy

Retraining should be controlled rather than triggered by a single unusual prediction.

```text
Production Monitoring
        |
        v
Drift / Performance Signal
        |
        v
Retraining Review
        |
        v
Collect New Labeled Data
        |
        v
Preprocess
        |
        v
Feature Engineering
        |
        v
Train Candidate Models
        |
        v
Evaluate
        |
        v
Validate Candidate
        |
        v
Promote if Approved
```

---

## 24. End-to-End ML Lifecycle

```text
Raw Customer Data
       |
       v
Preprocessing
       |
       v
Feature Engineering
       |
       v
Model Training
       |
       v
5-Fold Evaluation
       |
       v
Model Artifacts
       |
       v
FastAPI / Docker
       |
       v
Online Inference
       |
       v
Prediction Logging
       |
       v
Drift Monitoring
       |
       v
Retraining Trigger
       |
       v
Retrain / Validate
```

---

## 25. Technology Stack

| Area | Technology |
|---|---|
| Development | Anaconda |
| Notebook | JupyterLab |
| Language | Python |
| NLP | NLTK |
| Classical ML | Naive Bayes, MaxEnt, SVM, Decision Tree, RTE |
| Transformer | BERT where implemented |
| Serialization | `.pkl` model artifacts |
| REST API | FastAPI |
| Validation | Pydantic |
| Server | Uvicorn |
| API Testing | Postman / REST client |
| Containerization | Docker |
| Source Control | Git / GitHub / GitLab |
| Dataset | Kaggle Customer Sentiment Dataset |

---

## 26. Key Design Decisions

### Multiple Models

Multiple classifiers provide a meaningful comparison of different approaches.

### Shuffled Dataset

Shuffling avoids dependence on the original ordering of records.

### Five-Fold Evaluation

Provides repeated evaluation across multiple partitions.

### Serialized Models

Allows inference without rerunning model training.

### FastAPI

Provides a lightweight REST interface for online inference.

### Docker

Provides a portable and reproducible execution environment.

### Prediction Logging

Creates operational evidence for monitoring.

### Drift Monitoring

Provides a mechanism for detecting changes in customer language and model behavior.

### Controlled Retraining

Retraining follows review, retraining, and validation rather than blindly replacing the production model.

---

## 27. Reproducibility

```text
1. Obtain the Kaggle dataset.
2. Place Customer_Sentiment.csv in data/.
3. Run preprocess.ipynb.
4. Verify processed data.
5. Verify shuffled data.
6. Run preprocess_and_train.ipynb.
7. Verify models/ artifacts.
8. Verify logs/.
9. Install requirements.txt.
10. Run python api.py.
11. Test REST endpoints.
12. Build Docker image.
13. Run Docker container.
14. Repeat API testing through Docker.
```

---

## 28. Training vs Inference

### Training

```text
preprocess.ipynb
       |
       v
preprocess_and_train.ipynb
       |
       v
Processed Data
+
Model Artifacts
+
Training Logs
```

### Inference

Once models exist:

```bash
python api.py
```

The API loads the existing model artifacts.

Training does not need to be repeated for every API startup.

### Docker Inference

```bash
docker build -t sentiment-api .
```

```bash
docker run -d -p 8000:8000 --name sentiment-app-container sentiment-api
```

---

## 29. Recommended 5–7 Minute Demo

### 1. Project Introduction

Explain:

- Business problem
- Dataset
- Objective
- Architecture
- Offline training versus online inference

### 2. Preprocessing

Open:

```text
preprocess.ipynb
```

Show:

- Dataset loading
- Normalization
- Tokenization
- Stemming
- Processed data
- Shuffling

### 3. Feature Engineering

Explain:

- Word presence
- Positive terms
- Negation
- TF-IDF if implemented

### 4. Model Training

Open:

```text
preprocess_and_train.ipynb
```

Show:

- Training
- 5-fold evaluation
- Model comparison
- Experiment logs
- Serialization

### 5. Model Artifacts

Show:

```text
models/
```

Explain that the API consumes these files.

Key statement:

> The training notebook does not need to be rerun every time the API starts.

### 6. Start API

```bash
python api.py
```

Open:

```text
http://localhost:8000/
```

### 7. Test Prediction

Use Postman:

```text
POST /predict/naivebayes
```

```json
{
  "text": "great value for the money"
}
```

### 8. Demonstrate Docker

```bash
docker build -t sentiment-api .
```

```bash
docker run -d -p 8000:8000 --name sentiment-app-container sentiment-api
```

Repeat the API test.

### 9. Explain Monitoring

```text
Prediction Logs
       |
       v
Drift Monitoring
       |
       v
Retraining Trigger
       |
       v
Review → Retrain → Validate
```

---

## 30. Submission Checklist

| No. | Deliverable | Project Evidence |
|---:|---|---|
| 1 | Versioned dataset and pipeline code | Dataset, notebooks, Git repository and incremental commit history |
| 2 | Experiment tracking and model comparison | `logs/`, 5-fold evaluation and model comparison |
| 3 | Deployed model and working API | `api.py`, REST endpoints and Postman tests |
| 4 | Monitoring, drift and retraining | Prediction logging, drift-monitoring design and retraining trigger |
| 5 | README, architecture and demo | README, architecture diagrams and 5–7 minute demonstration |

---

## 31. Evaluation Rubric Alignment

### Data Engineering & Versioning — 20%

Addresses:

- Data ingestion
- Validation/processing
- Text normalization
- Tokenization
- Stemming
- Dataset processing
- Dataset shuffling
- Feature engineering
- Version-controlled code

### Experimentation & Reproducibility — 20%

Addresses:

- Multiple classification models
- 5-fold evaluation
- Experiment/training logs
- Serialized model artifacts
- Repeatable notebook workflow

### Model Packaging & Deployment — 20%

Addresses:

- Model serialization
- FastAPI REST endpoints
- Input validation
- Local execution
- Docker packaging
- Port mapping
- Postman/browser testing

### Monitoring, Drift & Retraining — 20%

Addresses:

- Prediction logging
- Monitoring design
- Changing vocabulary/topics
- Drift signals
- Retraining triggers
- Candidate-model validation

### Documentation & Presentation — 20%

Addresses:

- Architecture
- Project structure
- Setup instructions
- API examples
- Docker commands
- Demo flow
- Design decisions
- Monitoring/retraining design

---

## 32. Troubleshooting

### API Does Not Start

```bash
pip install -r requirements.txt
python api.py
```

Confirm port `8000` is available.

### Model Not Found

Verify required `.pkl` files exist under:

```text
models/
```

If artifacts have not been generated, run:

```text
preprocess_and_train.ipynb
```

### Docker Build Fails

Verify:

- Docker is running.
- `Dockerfile` exists.
- Required files are in the Docker build context.
- Required model artifacts exist.
- Required NLP resources are included according to the Dockerfile.

### Docker Container Does Not Respond

```bash
docker ps
```

```bash
docker logs sentiment-app-container
```

Verify:

```text
8000:8000
```

---

## 33. Production Considerations

The certification project demonstrates the end-to-end ML lifecycle. A production implementation could additionally include:

- MLflow model registry
- DVC dataset versioning
- Automated CI/CD
- Automated model validation
- Centralized logging
- Monitoring dashboards
- Automated drift detection
- Automated retraining pipelines
- Model versioning and rollback
- Authentication and authorization
- HTTPS/TLS
- API rate limiting
- Container vulnerability scanning
- Dependency security scanning
- Health/readiness endpoints
- Latency and throughput monitoring
- PII protection

---

## 34. Future Enhancements

1. Complete and document BERT fine-tuning if not already implemented.
2. Add TF-IDF as an additional classical feature representation.
3. Add DVC for formal dataset versioning.
4. Add MLflow for experiment tracking and model registry.
5. Add automated drift detection.
6. Add automated retraining pipelines.
7. Add a dedicated holdout test dataset.
8. Add API latency and throughput measurements.
9. Add automated API tests.
10. Add CI/CD for Docker image creation.
11. Add model versioning and rollback.
12. Add production monitoring dashboards.

---

## 35. Conclusion

The Customer Sentiment Analysis project demonstrates the complete lifecycle:

```text
Raw Customer Data
       |
       v
Preprocessing
       |
       v
Feature Engineering
       |
       v
Model Training
       |
       v
5-Fold Evaluation
       |
       v
Serialized Model Artifacts
       |
       v
FastAPI REST Service
       |
       v
Local Python / Docker
       |
       v
Online Sentiment Prediction
       |
       v
Prediction Logging
       |
       v
Drift Monitoring
       |
       v
Retraining Trigger
```

The key architectural principle is the separation of **offline model development** from **online inference**.

The notebooks perform data preparation, feature engineering, training, and evaluation. The resulting model artifacts are consumed by the FastAPI service. Docker provides a portable runtime for the inference service.

The monitoring and retraining design extends the solution beyond initial deployment and addresses the real-world challenge that customer language, topics, and sentiment expressions can change over time.

---

# Quick Reference

### Install

```bash
pip install -r requirements.txt
```

### Preprocess

```text
preprocess.ipynb
```

### Train

```text
preprocess_and_train.ipynb
```

### Start API

```bash
python api.py
```

### API

```text
http://localhost:8000/
```

### Example Endpoint

```text
POST http://localhost:8000/predict/naivebayes
```

### Example Payload

```json
{
  "text": "great value for the money"
}
```

### Build Docker

```bash
docker build -t sentiment-api .
```

### Run Docker

```bash
docker run -d -p 8000:8000 --name sentiment-app-container sentiment-api
```

### Check Container

```bash
docker ps
```

### Container Logs

```bash
docker logs sentiment-app-container
```

---

**AI/ML Certification Project — Customer Sentiment Analysis**

**NLP → Preprocessing → Feature Engineering → Model Training → Model Artifacts → FastAPI → Docker → Monitoring → Retraining Design**
