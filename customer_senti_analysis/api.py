import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
import re
nltk.download('rte')
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle
from sentence_transformers import SentenceTransformer

# Initialize the FastAPI application
app = FastAPI(
    title="Sentiment Analysis API",
    description="REST endpoints serving 4 distinct NLTK text classifiers",
    version="1.0"
)

# -------------------------------------------------------------
# 1. Feature Extractor (PLACEHOLDER)
# -------------------------------------------------------------
def get_word_features(review):
    stemmer = nltk.stem.PorterStemmer()
    words = [word if (word[0:2] == '__') else word.lower() \
                     for word in review.split() \
                     if len(review) >= 3]
    stemmed_words = [stemmer.stem(w) for w in words] 
    print(stemmed_words)

    def get_word_features(words):
            bag = {}
            stop_words = set(stopwords.words('english'))
            filtered_words = [w for w in words if not w in stop_words]
            words_uni = ['has(%s)' % ug for ug in filtered_words]
            for f in words_uni:
                bag[f] = 1

            # bag = collections.Counter(words_uni+words_bi+words_tri)
            return bag

    negtn_regex = re.compile(r"""(?:
            ^(?:never|no|nothing|nowhere|noone|none|not|
                havent|hasnt|hadnt|cant|couldnt|shouldnt|
                wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint
            )$
          )
          |
          n't
          """, re.X)

    pos_regex = re.compile(r"""(?:
                    ^(?:excellent|wow|awesome|happy|cool|good|love|
                        wonderful|amazing|amaze|bliss|enjoy|fantastic|
                        beautiful|beauty|better|very good|fun|funny|arent|luck|lucky|
                        nice|super|great
                    )$
                )
                |
                n't
                """, re.X)

    def get_negation_features(words):
            INF = 0.0
            negtn = [bool(negtn_regex.search(w)) for w in words]

            left = [0.0] * len(words)
            prev = 0.0
            for i in range(0, len(words)):
                if (negtn[i]):
                    prev = 1.0
                left[i] = prev
                prev = max(0.0, prev - 0.1)

            right = [0.0] * len(words)
            prev = 0.0
            for i in reversed(range(0, len(words))):
                if (negtn[i]):
                    prev = 1.0
                right[i] = prev
                prev = max(0.0, prev - 0.1)

            return dict(zip(
                ['neg_l(' + w + ')' for w in words] + ['neg_r(' + w + ')' for w in words],
                left + right))

    def get_positive_features(words):

            bag={}
            for word in words:
                if bool(pos_regex.search(word)):
                    key = 'pos(' + word + ')'
                    bag[key] = 1
            return bag


    def extract_features(words):

            features = {}
            negation_features = get_negation_features(words)
            features.update(negation_features)
            postive_features = get_positive_features(words)
            features.update(postive_features)
            word_features = get_word_features(words)
            features.update(word_features)
            #sys.stderr.write('\rfeatures extracted for ' + str(extract_features.count) + ' reviews')
            return features

    word_features = extract_features(stemmed_words)
    print(word_features)
    return word_features

    


# -------------------------------------------------------------
# 2. Model Loading Utility
# -------------------------------------------------------------
# Path where your .pkl files are saved
MODEL_DIR = "/app/models/"
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("\nGenerating BERT embeddings for custom reviews...")

def load_classifier(model_name: str):
    """Safely loads a pickled classifier from the disk."""
    # Find files that start with the classifier name to handle your timestamp suffixes
    if not os.path.exists(MODEL_DIR):
        raise HTTPException(status_code=500, detail="Models directory not found.")
        
    for file in os.listdir(MODEL_DIR):
        if file.startswith(model_name) and file.endswith(".pkl"):
            try:
                with open(os.path.join(MODEL_DIR, file), "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error loading {model_name}: {str(e)}")
                
    raise HTTPException(status_code=404, detail=f"Model file for {model_name} not found.")


# -------------------------------------------------------------
# 3. Pydantic Request Structure
# -------------------------------------------------------------
class TextRequest(BaseModel):
    text: str


# -------------------------------------------------------------
# 4. REST Endpoints
# -------------------------------------------------------------
@app.post("/predict/naivebayes")
def predict_naive_bayes(request: TextRequest):
    classifier = load_classifier("NaiveBayesClassifier")
    features = get_word_features(request.text)
    prediction = classifier.classify(features)
    return {"classifier": "NaiveBayesClassifier", "text": request.text, "sentiment": prediction}


@app.post("/predict/maxent")
def predict_maxent(request: TextRequest):
    classifier = load_classifier("MaxentClassifier")
    features = get_word_features(request.text)
    prediction = classifier.classify(features)
    return {"classifier": "MaxentClassifier", "text": request.text, "sentiment": prediction}


@app.post("/predict/svm")
def predict_svm(request: TextRequest):
    classifier = load_classifier("SvmClassifier")
    features = get_word_features(request.text)
    prediction = classifier.classify(features)
    return {"classifier": "SvmClassifier", "text": request.text, "sentiment": prediction}


@app.post("/predict/decisiontree")
def predict_decision_tree(request: TextRequest):
    classifier = load_classifier("DecisiontreeClassifier")
    features = get_word_features(request.text)
    prediction = classifier.classify(features)
    return {"classifier": "DecisiontreeClassifier", "text": request.text, "sentiment": prediction}

@app.post("/predict/rte")
def predict_decision_tree(request: TextRequest):
    classifier = load_classifier("RTEClassifier")
    features = get_word_features(request.text)
    prediction = classifier.classify(features)
    return {"classifier": "RTEClassifier", "text": request.text, "sentiment": prediction}

@app.post("/predict/bert")
def predict_decision_tree(request: TextRequest):
    classifier = load_classifier("BertClassifier")
    sentiment_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
    reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
    # Load a pre-trained sentence transformer model
    # 'all-MiniLM-L6-v2' is a good general-purpose model
    new_reviews_embeddings = bert_model.encode([request.text], show_progress_bar=True)
    new_bert_predictions = classifier.predict(new_reviews_embeddings)
    sentiment_label = reverse_sentiment_mapping[new_bert_predictions[0]]
    return {"classifier": "BertClassifier", "text": request.text, "sentiment": sentiment_label}


# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "API is running", "available_endpoints": ["/predict/naivebayes", "/predict/maxent", "/predict/svm", "/predict/decisiontree","/predict/rte", "/predict/bert"]}

if __name__ == "__main__":
    import uvicorn
    # This keeps the server alive and listening for API requests
    print('Starting api via uvicorn')
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
