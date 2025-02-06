from flask import Flask, request, jsonify
import tweepy
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load saved model and preprocessing tools
model = joblib.load('cyber_scam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
scaler = joblib.load('scaler.pkl')

# Twitter API credentials (replace with your keys)
TWITTER_API_KEY = "your_api_key"
TWITTER_API_SECRET = "your_api_secret"
TWITTER_ACCESS_TOKEN = "your_access_token"
TWITTER_ACCESS_TOKEN_SECRET = "your_access_token_secret"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

@app.route('/predict_vulnerability', methods=['POST'])
def predict_vulnerability():
    try:
        # Get Twitter username from request
        data = request.json
        username = data.get('username')  # Twitter handle, e.g., "elonmusk"
        tweet_count = data.get('count', 20)  # Number of recent tweets to analyze (default: 20)

        # Fetch user profile information
        user = api.get_user(screen_name=username)
        tweets = api.user_timeline(screen_name=username, count=tweet_count, tweet_mode='extended')

        # Extract profile-based features
        profile_features = np.array([
            user.favourites_count,
            user.followers_count,
            user.friends_count,
            user.statuses_count,
            int(user.default_profile),
            int(user.default_profile_image),
            int(user.geo_enabled),
            int(user.verified),
            len(user.description),
            user.created_at.timestamp() / (60 * 60 * 24)  # Convert account age to days
        ]).reshape(1, -1)

        # Concatenate tweet text for analysis
        combined_text = " ".join(tweet.full_text for tweet in tweets)
        
        # Text preprocessing and feature extraction
        text_features = vectorizer.transform([combined_text]).toarray()

        # Combine textual and profile features
        combined_input = np.hstack((text_features, profile_features))
        scaled_input = scaler.transform(combined_input)

        # Predict scam vulnerability
        prediction = model.predict(scaled_input)[0]
        predicted_class = 'Vulnerable' if prediction == 1 else 'Not Vulnerable'

        return jsonify({
            'username': username,
            'followers': user.followers_count,
            'friends': user.friends_count,
            'account_age_days': round(user.created_at.timestamp() / (60 * 60 * 24)),
            'prediction': predicted_class
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
