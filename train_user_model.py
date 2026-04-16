import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def extract_user_features(df):
    features = pd.DataFrame()
    features['statuses_count'] = df['statuses_count'].fillna(0)
    features['followers_count'] = df['followers_count'].fillna(0)
    features['friends_count'] = df['friends_count'].fillna(0)
    features['favourites_count'] = df['favourites_count'].fillna(0)
    features['listed_count'] = df['listed_count'].fillna(0)
    
    features['verified'] = df['verified'].apply(lambda x: 1 if x == 'TRUE' or x is True else 0)
    features['default_profile'] = df['default_profile'].apply(lambda x: 1 if x == 'TRUE' or x is True else 0)
    
    features['name_len'] = df['name'].fillna('').apply(len)
    features['name_digits'] = df['name'].fillna('').apply(lambda x: sum(c.isdigit() for c in str(x)))
    features['screen_name_len'] = df['screen_name'].fillna('').apply(len)
    features['screen_name_digits'] = df['screen_name'].fillna('').apply(lambda x: sum(c.isdigit() for c in str(x)))
    
    features['has_location'] = df['location'].fillna('').apply(lambda x: 1 if str(x).strip() != "" and str(x).lower() != 'null' else 0)
    
    return features

# 1. Load ONLY train users
print("Loading train users...")
train_users = pd.read_csv("data/train_users.csv")

# 2. Extract Features
X_train = extract_user_features(train_users)
y_train = train_users['is_fake']

# 3. Train
print("Training User Reliability Model (on Train Users only)...")
user_model = LogisticRegression(random_state=42)
user_model.fit(X_train, y_train)

# 4. Save
joblib.dump(user_model, "models/user_reliability_model.pkl")
print("✅ User reliability model saved!")

# 5. Evaluate on Test Users (Self-Check)
test_users = pd.read_csv("data/test_users.csv")
X_test = extract_user_features(test_users)
y_test = test_users['is_fake']
y_pred = user_model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
