import joblib
import json
import numpy as np

# Load the JSON file
with open(r"test_data.json") as f:
    test_data = json.load(f)

# Convert JSON to a DataFrame
import pandas as pd
test_df = pd.DataFrame(test_data)

# Fill missing descriptions
test_df['description'] = test_df['description'].fillna('')

# Load preprocessing tools and model
vectorizer = joblib.load('text_vectorizer.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('scam_vulnerability_model.pkl')

# Preprocess text features
text_features = vectorizer.transform(test_df['description']).toarray()

# Select numeric features
numeric_columns = ['favourites_count', 'followers_count', 'friends_count', 
                   'statuses_count', 'average_tweets_per_day', 'account_age_days', 
                   'default_profile', 'default_profile_image', 'geo_enabled', 
                   'verified']
numeric_features = test_df[numeric_columns].values

# Combine features
X_test = np.hstack((text_features, numeric_features))

# Scale features
X_test = scaler.transform(X_test)

# Predict with the model
predictions = model.predict(X_test)

# Add predictions to the DataFrame
test_df['predicted_label'] = predictions

# Output results
print("Predicted Label: 0 (Vulnerable to scams)\nPredicted Label: 1 (Not Vulnerable to scams).")
print(test_df[['description', 'predicted_label']])
