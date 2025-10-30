from flask import Flask, jsonify
from pymongo import MongoClient
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bson import ObjectId

app = Flask(__name__)

# MongoDB setup
MONGO_URI = "mongodb+srv://haris:ZrC2J6w3085WTBpl@cluster0.44q94.mongodb.net/WordWave?retryWrites=true&w=majority"
DB_NAME = "WW"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def fetch_top_recent_posts(user_id):
    # Fetch posts excluding the user's own posts
    recent_posts = db.posts.find({
        "postedBy": {"$ne": ObjectId(user_id)}
    })

    # Calculate interaction score for each post
    interaction_scores = []
    for post in recent_posts:
        likes = len(post.get("likes", []))
        comments = post.get("replies", [])
        num_comments = sum(len(reply.get('replies', [])) + 1 for reply in comments)
        interaction_score = likes + num_comments
        interaction_scores.append((str(post["_id"]), interaction_score))

    # Sort posts by interaction score in descending order
    sorted_posts = sorted(interaction_scores, key=lambda x: x[1], reverse=True)

    # Get top 10 posts
    top_recent_posts = [post_id for post_id, _ in sorted_posts[:10]]
    
    return top_recent_posts

def analyze_user_sentiments(user_id):
    # Fetch user interactions from last 24 hours
    start_time = datetime.now() - timedelta(days=1)
    user_id_obj = ObjectId(user_id) 
    user_interactions = db.posts.find({
        "$or": [
            {"likes": user_id_obj},
            {"replies.userId": user_id_obj},
            # {"replies.replies.userId": user_id_obj},
            {"postedBy": user_id_obj}
        ],
        "createdAt": {"$gte": start_time}
    })

    # Perform sentiment analysis on user interactions
    total_sentiment_scores = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
    total_interactions = 0
    for interaction in user_interactions:
        text = interaction.get('text', '')  # Using 'text' from the schema
        sentiment_scores = analyzer.polarity_scores(text)
        total_sentiment_scores['neg'] += sentiment_scores['neg']
        total_sentiment_scores['neu'] += sentiment_scores['neu']
        total_sentiment_scores['pos'] += sentiment_scores['pos']
        total_sentiment_scores['compound'] += sentiment_scores['compound']
        total_interactions += 1

        # Include comments in sentiment analysis
        for comment in interaction.get('replies', []):
            comment_text = comment.get('text', '')
            comment_sentiment_scores = analyzer.polarity_scores(comment_text)
            total_sentiment_scores['neg'] += comment_sentiment_scores['neg']
            total_sentiment_scores['neu'] += comment_sentiment_scores['neu']
            total_sentiment_scores['pos'] += comment_sentiment_scores['pos']
            total_sentiment_scores['compound'] += comment_sentiment_scores['compound']
            total_interactions += 1

            # Include nested replies in sentiment analysis
            # for reply in comment.get('replies', []):
            #     reply_text = reply.get('text', '')
            #     reply_sentiment_scores = analyzer.polarity_scores(reply_text)
            #     total_sentiment_scores['neg'] += reply_sentiment_scores['neg']
            #     total_sentiment_scores['neu'] += reply_sentiment_scores['neu']
            #     total_sentiment_scores['pos'] += reply_sentiment_scores['pos']
            #     total_sentiment_scores['compound'] += reply_sentiment_scores['compound']
            #     total_interactions += 1

        # Include sentiment of liked posts
        if user_id_obj in interaction.get('likes', []):
            liked_post_text = interaction.get('text', '')  # Using 'text' from the schema
            liked_post_sentiment_scores = analyzer.polarity_scores(liked_post_text)
            total_sentiment_scores['neg'] += liked_post_sentiment_scores['neg']
            total_sentiment_scores['neu'] += liked_post_sentiment_scores['neu']
            total_sentiment_scores['pos'] += liked_post_sentiment_scores['pos']
            total_sentiment_scores['compound'] += liked_post_sentiment_scores['compound']
            total_interactions += 1

    # Calculate average sentiment scores
    if total_interactions > 0:
        average_sentiment_scores = {key: value / total_interactions for key, value in total_sentiment_scores.items()}
    else:
        average_sentiment_scores = total_sentiment_scores
    
    # Determine which sentiment is higher and which is lower
    max_sentiment = max(average_sentiment_scores, key=average_sentiment_scores.get)
    min_sentiment = min(average_sentiment_scores, key=average_sentiment_scores.get)
    
    return average_sentiment_scores, max_sentiment, min_sentiment


def filter_negative_posts(top_recent_posts):
    filtered_posts = []
    for post_id in top_recent_posts:
        post = db.posts.find_one({"_id": ObjectId(post_id)})
        text = post.get('text', '')  # Using 'text' from the schema
        sentiment_scores = analyzer.polarity_scores(text)
        if sentiment_scores['compound'] >= 0:  # Check if compound score is non-negative
            filtered_posts.append(post_id)
    return filtered_posts


@app.route('/recommendations/<user_id>', methods=['GET'])
def get_user_sentiments_and_top_posts(user_id):
    # Analyze user sentiments
    user_sentiments, max_sentiment, min_sentiment = analyze_user_sentiments(user_id)

    # Fetch top recent posts
    top_recent_posts = fetch_top_recent_posts(user_id)

    # Determine user emotion based on compound sentiment score
    if user_sentiments['compound'] >= 0.05:
        emotion = 'positive'
    elif user_sentiments['compound'] <= -0.05:
        emotion = 'negative'
    else:
        emotion = 'neutral'

    # Filter out negative emotion posts if user's emotion is negative
    if emotion == 'negative':
        top_recent_posts = filter_negative_posts(top_recent_posts)

    response = {
        'user_sentiments': user_sentiments,
        'emotion': emotion,
        'Recommendations': top_recent_posts
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, port=6000)