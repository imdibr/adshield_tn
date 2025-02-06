import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout# type: ignore

# Load dataset
data = pd.read_csv(r"C:\Users\ABISHEKRS\Downloads\pseudo_facebook.csv\pseudo_facebook.csv")
data.columns = data.columns.str.strip()
if 'userid' in data.columns:
    data.drop(columns=['userid'], inplace=True)

# Drop unused columns (assuming 'userid' is not a predictive feature)
# data.drop(columns=['userid'], inplace=True)

# Feature Engineering - Encoding categorical data
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])  # Encode gender (male=1, female=0)

# Define target variable (engagement level)
def categorize_engagement(row):
    total_activity = row['friend_count'] + row['friendships_initiated']
    if total_activity == 0:
        return 'Inactive'
    elif total_activity < 50:
        return 'Low'
    elif total_activity < 200:
        return 'Moderate'
    else:
        return 'High'

data['engagement_level'] = data.apply(categorize_engagement, axis=1)

# Encode engagement level for classification
data['engagement_level'] = label_encoder.fit_transform(data['engagement_level'])  # Assign numerical labels

# Split features and target
X = data.drop(columns=['engagement_level'])
y = data['engagement_level']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build a neural network model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')  # 4 engagement categories: Inactive, Low, Moderate, High
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
model.save('user_engagement_model.h5')
print("  ")
print("Model training complete and saved as 'user_engagement_model.h5'.")
