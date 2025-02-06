import pandas as pd
import instaloader
import facebook
import tweepy

class UserProfiler:
    def __init__(self):
        self.insta_loader = instaloader.Instaloader()
        self.fb_api = self.authenticate_facebook()
        self.twitter_api = self.authenticate_twitter()

    # Instagram Data Fetching
    def fetch_instagram_data(self, username):
        """Fetch data from Instagram."""
        try:
            profile = self.insta_loader.check_profile_id(username)
            # Here you would define how to extract data like followers, posts, etc.
            insta_data = {
                'username': username,
                'followers': profile.followers,
                'following': profile.followees,
                'posts': profile.mediacount,
                'bio': profile.biography
            }
            return insta_data
        except Exception as e:
            print(f"Error fetching Instagram data: {e}")
            return None

    # Facebook Data Fetching
    def authenticate_facebook(self):
        """Authenticate Facebook Graph API."""
        access_token = 'your_facebook_access_token'  # Replace with your access token
        graph = facebook.GraphAPI(access_token)
        return graph

    def fetch_facebook_data(self, user_id):
        """Fetch Facebook data."""
        try:
            user_data = self.fb_api.get_object(user_id)
            fb_data = {
                'username': user_data['name'],
                'friends_count': user_data['friends']['summary']['total_count'],
                'posts_count': user_data['posts']['summary']['total_count']
            }
            return fb_data
        except Exception as e:
            print(f"Error fetching Facebook data: {e}")
            return None

    # Twitter (X) Data Fetching
    def authenticate_twitter(self):
        """Authenticate Twitter API."""
        consumer_key = 'your_consumer_key'
        consumer_secret = 'your_consumer_secret'
        access_token = 'your_access_token'
        access_token_secret = 'your_access_token_secret'
        
        auth = tweepy.OAuth1UserHandler(
            consumer_key, consumer_secret, access_token, access_token_secret)
        api = tweepy.API(auth)
        return api

    def fetch_twitter_data(self, username):
        """Fetch Twitter data."""
        try:
            user = self.twitter_api.get_user(screen_name=username)
            twitter_data = {
                'username': username,
                'followers_count': user.followers_count,
                'following_count': user.friends_count,
                'tweets_count': user.statuses_count
            }
            return twitter_data
        except Exception as e:
            print(f"Error fetching Twitter data: {e}")
            return None

    # Read data from CSV (Instagram, Facebook, Twitter)
    def read_csv_data(self, platform, file_path):
        """Read data from CSV based on platform."""
        try:
            if platform == 'instagram':
                return pd.read_csv(file_path)
            elif platform == 'facebook':
                return pd.read_csv(file_path)
            elif platform == 'twitter':
                return pd.read_csv(file_path)
            else:
                raise ValueError("Unknown platform")
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found. Please check the file path.")
            return None

    # User Profiling (Based on specific platform data)
    def profile_instagram_users(self, insta_df):
        """Profile Instagram users for scam vulnerability based on follower count."""
        insta_vulnerable = insta_df[insta_df['followers'] < 1000]  # Example condition
        return insta_vulnerable

    def profile_facebook_users(self, fb_df):
        """Profile Facebook users for scam vulnerability based on friends count."""
        fb_vulnerable = fb_df[fb_df['friends_count'] < 500]  # Example condition
        return fb_vulnerable

    def profile_twitter_users(self, twitter_df):
        """Profile Twitter users for scam vulnerability based on follower count."""
        twitter_vulnerable = twitter_df[twitter_df['followers_count'] < 1000]  # Example condition
        return twitter_vulnerable

# Example Usage
if __name__ == "__main__":
    profiler = UserProfiler()

    # Fetching Data for Instagram, Facebook, and Twitter
    insta_data = profiler.fetch_instagram_data('abishek_r_s_')
    fb_data = profiler.fetch_facebook_data('user_facebook_id')  # Replace with real Facebook user ID
    twitter_data = profiler.fetch_twitter_data('user_twitter_handle')  # Replace with real Twitter handle

    # Print the collected data
    print("Instagram Data:", insta_data)
    print("Facebook Data:", fb_data)
    print("Twitter Data:", twitter_data)

    # Read data from CSV files (Instagram, Facebook, Twitter)
    insta_df = profiler.read_csv_data('instagram', 'instagram_data.csv')
    fb_df = profiler.read_csv_data('facebook', 'facebook_data.csv')
    twitter_df = profiler.read_csv_data('twitter', 'twitter_data.csv')

    # Profile Users for Scam Vulnerability
    if insta_df is not None:
        profiled_insta = profiler.profile_instagram_users(insta_df)
        print("\nPotential scam-vulnerable Instagram users:")
        print(profiled_insta)

    if fb_df is not None:
        profiled_fb = profiler.profile_facebook_users(fb_df)
        print("\nPotential scam-vulnerable Facebook users:")
        print(profiled_fb)

    if twitter_df is not None:
        profiled_twitter = profiler.profile_twitter_users(twitter_df)
        print("\nPotential scam-vulnerable Twitter users:")
        print(profiled_twitter)
