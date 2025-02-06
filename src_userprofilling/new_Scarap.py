from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model and preprocessing tools
model = joblib.load('cyber_scam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Process input data
    text = vectorizer.transform([data['description']]).toarray()
    features = np.array([data['favourites_count'], data['followers_count'], 
                         data['friends_count'], data['statuses_count'], 
                         data['average_tweets_per_day'], data['account_age_days'], 
                         int(data['default_profile']), int(data['default_profile_image']), 
                         int(data['geo_enabled']), int(data['verified'])])
    
    combined_input = np.hstack((text, features.reshape(1, -1)))
    scaled_input = scaler.transform(combined_input)

    prediction = model.predict(scaled_input)[0]
    predicted_class = 'bot' if prediction == 1 else 'human'

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
