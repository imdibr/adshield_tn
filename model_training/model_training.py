import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv(r"C:\Users\ABISHEKRS\Downloads\twitter_human_bots_dataset.csv\twitter_human_bots_dataset.csv")

# Fill missing descriptions
data['description'] = data['description'].fillna('')

# Add a 'label' column if it doesn't exist
if 'label' not in data.columns:
    print("'label' column is missing. Adding default labels.")
    data['label'] = np.random.randint(0, 2, size=len(data))  # Random labels (0 or 1)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=1000)
text_features = vectorizer.fit_transform(data['description']).toarray()

# Select numeric features
numeric_columns = ['favourites_count', 'followers_count', 'friends_count', 
                   'statuses_count', 'average_tweets_per_day', 'account_age_days', 
                   'default_profile', 'default_profile_image', 'geo_enabled', 
                   'verified']
numeric_features = data[numeric_columns].values

# Combine features
X = np.hstack((text_features, numeric_features))
y = data['label']  # Target column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and preprocessing tools
joblib.dump(model, 'scam_vulnerability_model.pkl')
joblib.dump(vectorizer, 'text_vectorizer.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and tools have been saved successfully.")
