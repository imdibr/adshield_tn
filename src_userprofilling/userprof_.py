import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import nltk
from nltk.corpus import stopwords

# Load dataset
file_path = r"C:\Users\ABISHEKRS\Downloads\twitter_human_bots_dataset.csv\twitter_human_bots_dataset.csv"
df = pd.read_csv(file_path)

# Inspect dataset
print(df.head())
print(df.info())

# Handling missing values
df.dropna(inplace=True)

# Convert categorical boolean values to 0 and 1
boolean_cols = ['default_profile', 'default_profile_image', 'geo_enabled', 'verified']
df[boolean_cols] = df[boolean_cols].astype(int)

# Download stopwords
nltk.download('stopwords')

# Text cleaning function for 'description' column
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@w+|\#','', text)  # Remove mentions and hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df['cleaned_description'] = df['description'].apply(clean_text)

# Define a custom function to label vulnerability (can be adjusted as needed)
def label_vulnerability(row):
    # Example vulnerability criteria
    if row['followers_count'] < 100 or row['account_age_days'] > 365:
        return 1  # Vulnerable
    else:
        return 0  # Not Vulnerable

# Apply the vulnerability labeling
df['vulnerable_label'] = df.apply(label_vulnerability, axis=1)

# Selecting features and target
X = df[['cleaned_description', 'favourites_count', 'followers_count', 'friends_count', 
        'statuses_count', 'average_tweets_per_day', 'account_age_days'] + boolean_cols]
y = df['vulnerable_label']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert cleaned text to numerical representation using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_text = vectorizer.fit_transform(X_train['cleaned_description']).toarray()
X_test_text = vectorizer.transform(X_test['cleaned_description']).toarray()

# Combine TF-IDF features with numerical features
X_train_final = np.hstack((X_train_text, X_train.drop(columns=['cleaned_description']).values))
X_test_final = np.hstack((X_test_text, X_test.drop(columns=['cleaned_description']).values))

# Normalize numerical features
scaler = StandardScaler()
X_train_final = scaler.fit_transform(X_train_final)
X_test_final = scaler.transform(X_test_final)

# Train the XGBoost model
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_final, y_train)

# Make predictions
y_pred = model.predict(X_test_final)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Get prediction probabilities (for profiling)
y_pred_proba = model.predict_proba(X_test_final)[:, 1]  # Probability of being vulnerable (class 1)

# Create user profile dataframe
user_profiles = X_test.copy()
user_profiles['predicted_label'] = y_pred
user_profiles['predicted_proba'] = y_pred_proba
user_profiles['predicted_vulnerability'] = np.where(y_pred == 1, 'Vulnerable', 'Not Vulnerable')

# Save the user profile data to CSV
output_file = r'C:\Users\ABISHEKRS\Downloads\twitter_human_bots_dataset.csv\cyber_scam_vulnerability_profiles_{time}.csv'
user_profiles.to_csv(output_file, index=False)

print(f"User profiles saved to {output_file}")
