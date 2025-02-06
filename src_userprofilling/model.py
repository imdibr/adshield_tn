import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
file_path = r"C:\Users\ABISHEKRS\Downloads\pseudo_facebook.csv\pseudo_facebook.csv"
data = pd.read_csv(file_path)

# Ensure no leading/trailing spaces in column names
data.columns = data.columns.str.strip()

# Check if 'gender' column exists
if 'gender' in data.columns:
    # Fit a LabelEncoder on the 'gender' column
    label_encoder = LabelEncoder()
    data['gender'] = label_encoder.fit_transform(data['gender'])

    # Save the fitted LabelEncoder as a .pkl file
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Label Encoder saved successfully as 'label_encoder.pkl'.")
else:
    print("'gender' column not found in the dataset.")
