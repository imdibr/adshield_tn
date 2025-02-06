from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model and preprocessing tools
model = joblib.load('scam_vulnerability_model.pkl')  # Model trained for scam detection
vectorizer = joblib.load('text_vectorizer.pkl')      # TF-IDF or similar vectorizer
scaler = joblib.load('scaler.pkl')                  # Scaler for numeric features

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Validate input
    required_fields = [
        'description', 'favourites_count', 'followers_count', 'friends_count',
        'statuses_count', 'average_tweets_per_day', 'account_age_days',
        'default_profile', 'default_profile_image', 'geo_enabled', 'verified'
    ]
    
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields in input'}), 400

    # Process input text
    text = vectorizer.transform([data['description']]).toarray()

    # Process numeric features
    numeric_features = np.array([
        data['favourites_count'], data['followers_count'], 
        data['friends_count'], data['statuses_count'], 
        data['average_tweets_per_day'], data['account_age_days'], 
        int(data['default_profile']), int(data['default_profile_image']), 
        int(data['geo_enabled']), int(data['verified'])
    ])

    # Combine text and numeric features
    combined_input = np.hstack((text, numeric_features.reshape(1, -1)))

    # Scale the combined input
    scaled_input = scaler.transform(combined_input)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    predicted_class = 'vulnerable' if prediction == 1 else 'not_vulnerable'

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
