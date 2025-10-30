from flask import Flask, jsonify
from pymongo import MongoClient
from datetime import datetime, timedelta
from bson import ObjectId
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# MongoDB setup
MONGO_URI = "mongodb+srv://haris:ZrC2J6w3085WTBpl@cluster0.44q94.mongodb.net/WordWave?retryWrites=true&w=majority"
DB_NAME = "WW"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

def fetch_interacted_post_ids(user_id):
    user_id_obj = ObjectId(user_id)
    posts = db.posts.aggregate([
        {"$match": {"$or": [
            {"likes": user_id_obj},
            {"replies.userId": user_id_obj},
            {"postedBy": user_id_obj}
        ]}},
        {"$project": {"_id": 1}}
    ])
    return [str(post['_id']) for post in posts]

def fetch_top_recent_posts(user_id):
    start_date = datetime.now() - timedelta(days=1)
    interacted_ids = fetch_interacted_post_ids(user_id)
    
    recent_posts = db.posts.find({
        "createdAt": {"$gte": start_date},
        "postedBy": {"$ne": ObjectId(user_id)},
        "_id": {"$nin": [ObjectId(id) for id in interacted_ids]},
        "$or": [{"likes": {"$exists": True, "$ne": []}}, {"replies": {"$exists": True, "$ne": []}}]
    })

    interaction_scores = []
    for post in recent_posts:
        likes = len(post.get("likes", []))
        replies = len(post.get("replies", []))
        interaction_score = likes + replies
        interaction_scores.append((str(post["_id"]), interaction_score))

    sorted_posts = sorted(interaction_scores, key=lambda x: x[1], reverse=True)
    top_recent_posts = [post_id for post_id, _ in sorted_posts[:10]]
    
    return top_recent_posts

def analyze_user_sentiments(user_id):
    start_time = datetime.now() - timedelta(days=1)
    user_id_obj = ObjectId(user_id) 
    user_interactions = db.posts.find({
        "$or": [
            {"likes": user_id_obj},
            {"replies.userId": user_id_obj},
            {"postedBy": user_id_obj}
        ],
        "createdAt": {"$gte": start_time}
    })

    total_sentiment_scores = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
    total_interactions = 0
    for interaction in user_interactions:
        text = interaction.get('text', '')
        sentiment_scores = analyzer.polarity_scores(text)
        total_sentiment_scores['neg'] += sentiment_scores['neg']
        total_sentiment_scores['neu'] += sentiment_scores['neu']
        total_sentiment_scores['pos'] += sentiment_scores['pos']
        total_sentiment_scores['compound'] += sentiment_scores['compound']
        total_interactions += 1
        
        for reply in interaction.get('replies', []):
            reply_text = reply.get('text', '')
            reply_sentiment_scores = analyzer.polarity_scores(reply_text)
            total_sentiment_scores['neg'] += reply_sentiment_scores['neg']
            total_sentiment_scores['neu'] += reply_sentiment_scores['neu']
            total_sentiment_scores['pos'] += reply_sentiment_scores['pos']
            total_sentiment_scores['compound'] += reply_sentiment_scores['compound']
            total_interactions += 1

    if total_interactions > 0:
        average_sentiment_scores = {key: value / total_interactions for key, value in total_sentiment_scores.items()}
    else:
        average_sentiment_scores = total_sentiment_scores
    
    return average_sentiment_scores

def fetch_user_content(user_id):
    user_id_obj = ObjectId(user_id)
    user_posts = db.posts.find({"postedBy": user_id_obj})
    content = [post.get("text", "") for post in user_posts if "text" in post]

    liked_posts = db.posts.find({"likes": user_id_obj})
    content.extend(post.get("text", "") for post in liked_posts if "text" in post)

    replies = db.posts.find({"replies.userId": user_id_obj})
    for post in replies:
        for reply in post.get("replies", []):
            if reply.get("userId") == user_id_obj and "text" in reply:
                content.append(reply.get("text"))

    return content

def content_based_recommendations(user_id):
    user_id_obj = ObjectId(user_id)
    interacted_ids = fetch_interacted_post_ids(user_id)
    user_content = fetch_user_content(user_id)
    if not user_content:
        return []

    all_posts = list(db.posts.find({
        "postedBy": {"$ne": user_id_obj},
        "_id": {"$nin": [ObjectId(id) for id in interacted_ids]}
    }, {"text": 1}))

    all_texts = [post["text"] for post in all_posts if "text" in post]
    post_ids = [str(post["_id"]) for post in all_posts if "text" in post]

    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
    user_tfidf = tfidf_vectorizer.transform(user_content)

    cosine_similarities = linear_kernel(user_tfidf, tfidf_matrix)
    top_indices = cosine_similarities.sum(axis=0).argsort()[-10:][::-1]
    recommended_posts = [post_ids[index] for index in top_indices if index < len(post_ids)]

    return recommended_posts

def filter_negative_posts(post_ids):
    filtered_posts = []
    for post_id in post_ids:
        post = db.posts.find_one({"_id": ObjectId(post_id)})
        if post:
            text = post.get('text', '')
            sentiment_scores = analyzer.polarity_scores(text)
            if sentiment_scores['compound'] >= 0:  # Check if compound score is non-negative
                filtered_posts.append(post_id)
    return filtered_posts



@app.route('/recommendations/<user_id>', methods=['GET'])
def get_user_sentiments_and_top_posts(user_id):
    user_sentiments = analyze_user_sentiments(user_id)
    top_recent_posts = fetch_top_recent_posts(user_id)
    content_recommendations = content_based_recommendations(user_id)

    combined_posts = list(set(top_recent_posts + content_recommendations))

    if user_sentiments['compound'] >= 0.05:
        emotion = 'positive'
    elif user_sentiments['compound'] <= -0.05:
        emotion = 'negative'
    else:
        emotion = 'neutral'

    # final_recommendations = combined_posts if emotion != 'negative' else []

    if emotion == 'negative':
        final_recommendations = filter_negative_posts(combined_posts)
    else:
        final_recommendations = combined_posts

    response = {
        'user_sentiments': user_sentiments,
        'emotion': emotion,
        'Recommendations': final_recommendations
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, port=6000)
