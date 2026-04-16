import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import re

# 1. Load models
print("Loading models...")
text_clf = joblib.load("models/text_model.pkl")
style_clf = joblib.load("models/style_model.pkl")
tfidf_clf = joblib.load("models/tfidf_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
embedder = joblib.load("models/embedding_model.pkl")
user_reliability_model = joblib.load("models/user_reliability_model.pkl")

# 2. Load TRAIN datasets
train_df = pd.read_csv("data/train.csv")
train_users = pd.read_csv("data/train_users.csv")

# 3. Import feature extraction
from train_user_model import extract_user_features

# 4. Generate user scores from TRAIN users only
print("Calculating user scores for training...")
all_user_scores = user_reliability_model.predict_proba(
    extract_user_features(train_users)
)[:, 1]

real_user_scores = all_user_scores[train_users['is_fake'] == 0]
fake_user_scores = all_user_scores[train_users['is_fake'] == 1]

def sample_user_score(label):
    if label == 0:
        return np.random.choice(real_user_scores)
    else:
        return np.random.choice(fake_user_scores)

# 5. Hindi Style Features
hindi_stopwords = set(["है","हैं","था","थे","की","का","के","को","में","पर","से","तक","और","या","लेकिन","अगर","तो","ही","भी","नहीं","क्या","क्यों","कौन","कब","यह","वह","इन","उन","उस","इस","कुछ","सब","बहुत","अब","जब","तब","जहाँ","वहाँ"])

def extract_style_features(text):
    words = text.split()
    sentences = text.split('.')
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0
    exclamations = text.count('!')
    questions = text.count('?')
    punctuation = len(re.findall(r'[^\w\s]', text))
    uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    ttr = len(set(words)) / word_count if word_count > 0 else 0
    stopword_ratio = sum(1 for w in words if w in hindi_stopwords) / word_count if word_count > 0 else 0
    return [word_count, sentence_count, avg_sentence_len, exclamations, questions, punctuation, uppercase_ratio, ttr, stopword_ratio]

# 6. Generate Meta Features
def generate_social_meta_features(df):
    X_embed = embedder.encode(df['content'].tolist(), show_progress_bar=True)
    X_style = np.array([extract_style_features(t) for t in df['content']])
    X_tfidf = tfidf_vectorizer.transform(df['content'])
    
    text_probs = text_clf.predict_proba(X_embed)[:, 1]
    style_probs = style_clf.predict_proba(X_style)[:, 1]
    tfidf_probs = tfidf_clf.predict_proba(X_tfidf)[:, 1]
    
    # Simulated user scores
    user_scores = np.array([sample_user_score(l) for l in df['label']])
    
    return np.column_stack((text_probs, style_probs, tfidf_probs, user_scores))

print("Generating Social Meta features (Train News + Train Users)...")
X_meta_train = generate_social_meta_features(train_df)
y_train = train_df['label']

# 7. Scale features
scaler = StandardScaler()
X_meta_train = scaler.fit_transform(X_meta_train)

# 🔥 8. Reduce user feature weight (IMPORTANT)
USER_WEIGHT = 0.3   # try 0.1 / 0.3 / 0.5
X_meta_train[:, 3] *= USER_WEIGHT

print("Training Social Meta Model...")
social_meta_model = LogisticRegression(max_iter=1000)
social_meta_model.fit(X_meta_train, y_train)

# Save
joblib.dump(social_meta_model, "models/social_meta_model.pkl")
joblib.dump(scaler, "models/social_meta_scaler.pkl")

print("✅ Social Meta Model trained and saved!")