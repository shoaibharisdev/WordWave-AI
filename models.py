from flask import Flask, request, jsonify
import torch
from torch.nn.functional import sigmoid
import pandas as pd
import pymongo
import certifi
import re
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim import corpora, models
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from bson import ObjectId
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask_cors import CORS
import nltk
import threading
import time
import gc

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ===== CONFIGURATION SETTINGS - EDIT THESE =====
TRENDING_UPDATE_FREQUENCY = 86400  # Seconds between auto-updates (86400 = 24 hours)
CACHE_CHECK_INTERVAL = 300         # Seconds between checks (300 = 5 minutes)
CACHE_EXPIRY_HOURS = 24            # Hours until cache expires (24 = 1 day)

# ===== CACHE SETUP =====
trending_cache = {
    'topics': [],
    'last_updated': None,
    'is_updating': False
}

# Load environment variables
load_dotenv()

# ===== MONGODB SETUP =====
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://haris:ZrC2J6w3085WTBpl@cluster0.44q94.mongodb.net/WordWave?retryWrites=true&w=majority")
DB_NAME = os.getenv("DB_NAME", "WW")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "posts")

# Initialize MongoDB client and database
try:
    client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    posts_collection = db[COLLECTION_NAME]
    cache_collection = db['trending_cache']
    print("✅ MongoDB connected successfully")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    db = None
    posts_collection = None
    cache_collection = None

# ===== HATE SPEECH DETECTION SETUP =====
# LAZY LOADING - Model only loads when needed
model = None
tokenizer = None

def load_hate_speech_model():
    """Lazy load the hate speech model only when needed"""
    global model, tokenizer
    if model is None or tokenizer is None:
        try:
            print("🔄 Loading hate speech model...")
            from transformers import BertTokenizer, BertForSequenceClassification
            tokenizer = BertTokenizer.from_pretrained("unitary/toxic-bert")
            model = BertForSequenceClassification.from_pretrained("unitary/toxic-bert")
            model.eval()
            print("✅ Hate speech model loaded successfully")
        except Exception as e:
            print(f"❌ Hate speech model loading failed: {e}")
            return None, None
    return tokenizer, model

# Define labels for toxic-bert
TOXIC_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ===== SENTIMENT ANALYSIS SETUP =====
analyzer = SentimentIntensityAnalyzer()

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# ===== NLP SETUP =====
# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Define a set of meaningless words to remove
meaningless_words = {
    'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 
    'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 
    'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 
    'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 
    'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 
    'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 
    'do', 'done', 'down', 'due', 'during', 'each', 'eg','come','came','coming','live', 'living', 'eight', 'either', 'eleven', 'else', 'elsewhere', 
    'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 
    'few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 
    'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 
    'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 
    'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 
    'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 
    'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 
    'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 
    'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 
    'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 
    'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 
    'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 
    'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 
    'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 
    'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 
    'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 
    'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 
    'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 
    'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 
    'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'want', 'used', 'using', 'use'
}

# ===== HELPER FUNCTIONS =====

# Hate Speech Detection Functions
def predict_hate_speech(text):
    """
    Predict if text contains hate speech using toxic-bert model.
    This model is multi-label, so we use sigmoid activation and thresholds.
    """
    tokenizer_local, model_local = load_hate_speech_model()
    
    if tokenizer_local is None or model_local is None:
        return {"error": "Hate speech model not available"}
    
    try:
        inputs = tokenizer_local(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        with torch.no_grad():
            outputs = model_local(**inputs)
            probabilities = sigmoid(outputs.logits)
        
        threshold = 0.5
        predictions = probabilities[0].tolist()
        detected_categories = []
        max_confidence = 0.0
        
        for i, label in enumerate(TOXIC_LABELS):
            if predictions[i] > threshold:
                detected_categories.append({
                    "label": label,
                    "confidence": round(predictions[i], 3)
                })
                max_confidence = max(max_confidence, predictions[i])
        
        is_hate = len(detected_categories) > 0
        
        if is_hate:
            primary_category = max(detected_categories, key=lambda x: x['confidence'])
            classification = primary_category['label']
        else:
            classification = "Not classified as any category"
        
        # Clean up
        del inputs, outputs, probabilities
        gc.collect()
        
        return {
            "hate_speech": is_hate,
            "classification": classification,
            "confidence": round(max_confidence, 3) if is_hate else 0.0,
            "detected_categories": detected_categories
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# Trending Topics Functions
def get_mongo_data(hours_back=730):
    """Get recent data from MongoDB for trending topics"""
    if posts_collection is None:
        return []
    try:
        start_time = datetime.now() - timedelta(hours=hours_back)
        return list(posts_collection.find({
            "createdAt": {"$gte": start_time}
        }).sort("createdAt", pymongo.DESCENDING).limit(10000))
    except Exception as e:
        print(f"Error fetching MongoDB data: {e}")
        return []

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    stop_words.update(meaningless_words)
    text = " ".join([word for word in word_tokenize(text) if word not in stop_words and len(word) > 3])
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])

def get_literal_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.add(lemma.name().replace('_', ' '))
    return synonyms

def create_pattern_from_top_words_with_synonyms(lda_model, topic_id, topn=1):
    top_words = [word[0] for word in lda_model.show_topic(topic_id, topn)]
    all_words = set(top_words)
    for word in top_words:
        all_words.update(get_literal_synonyms(word))
    pattern = '|'.join([f'\\b{word}\\b' for word in all_words])
    return pattern

def calculate_trending_topics():
    """Calculate trending topics (the expensive operation)"""
    print("🔄 Calculating trending topics...")
    start_time = time.time()
    
    recent_data = get_mongo_data(hours_back=17520)
    if not recent_data:
        print("❌ No data available for trending topics")
        return []
    
    df_twitter_recent = pd.DataFrame(recent_data)
    
    if 'text' not in df_twitter_recent.columns:
        print("❌ No 'text' column in data")
        return []
    
    df_twitter_recent['processed_text'] = df_twitter_recent['text'].apply(preprocess_text)
    df_twitter_recent = df_twitter_recent[df_twitter_recent['processed_text'].str.len() > 0]
    
    if len(df_twitter_recent) == 0:
        print("❌ No valid text data after preprocessing")
        return []

    try:
        texts = df_twitter_recent['processed_text'].str.split().tolist()
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        lda_model = models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)

        topics_info = []
        for i in range(10):
            topic_words = lda_model.show_topic(i, topn=1)
            if topic_words:
                topic_words = [word for word in topic_words if len(word[0]) > 3]  
                if topic_words:
                    pattern = create_pattern_from_top_words_with_synonyms(lda_model, i)
                    regex = re.compile(pattern)

                    related_posts = [post for post in df_twitter_recent['processed_text'] if regex.search(post)]
                    
                    if len(related_posts) > 1:
                        topics_info.append({
                            "Topic": topic_words[0][0],
                            "Number of Related Posts": len(related_posts),
                            "Related Posts": related_posts[:5]
                        })

        # Clean up memory
        del df_twitter_recent, texts, dictionary, corpus, lda_model
        gc.collect()

        processing_time = time.time() - start_time
        print(f"✅ Trending topics calculated in {processing_time:.2f}s. Found {len(topics_info)} topics.")
        return topics_info
    except Exception as e:
        print(f"❌ Error in topic modeling: {str(e)}")
        return []

def save_topics_to_cache(topics):
    """Save topics to MongoDB cache"""
    if cache_collection is None:
        return
        
    try:
        cache_doc = {
            'type': 'trending_topics',
            'topics': topics,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(hours=CACHE_EXPIRY_HOURS)
        }
        cache_collection.replace_one(
            {'type': 'trending_topics'},
            cache_doc,
            upsert=True
        )
        print("✅ Topics saved to cache")
    except Exception as e:
        print(f"❌ Error saving to cache: {e}")

def load_topics_from_cache():
    """Load topics from MongoDB cache"""
    if cache_collection is None:
        return None
        
    try:
        cache_doc = cache_collection.find_one({'type': 'trending_topics'})
        if cache_doc and cache_doc.get('expires_at', datetime.now()) > datetime.now():
            print("✅ Topics loaded from cache")
            return cache_doc.get('topics', [])
    except Exception as e:
        print(f"❌ Error loading from cache: {e}")
    return None

def update_trending_topics():
    """Update trending topics and cache"""
    if trending_cache['is_updating']:
        print("⏳ Trending update already in progress...")
        return
        
    trending_cache['is_updating'] = True
    try:
        topics = calculate_trending_topics()
        if topics:
            trending_cache['topics'] = topics
            trending_cache['last_updated'] = datetime.now()
            save_topics_to_cache(topics)
            print(f"🔄 Trending topics updated at {datetime.now()}")
        else:
            print("⚠️  No topics generated, keeping previous cache")
    except Exception as e:
        print(f"❌ Error updating trending topics: {e}")
    finally:
        trending_cache['is_updating'] = False

def background_trending_updater():
    """Background job to update trending topics"""
    print("🔄 Background trending updater started")
    while True:
        try:
            current_time = datetime.now()
            if (trending_cache['last_updated'] is None or 
                (current_time - trending_cache['last_updated']).total_seconds() >= TRENDING_UPDATE_FREQUENCY):
                
                if not trending_cache['is_updating']:
                    print("🔄 Auto-updating trending topics...")
                    update_trending_topics()
            
            time.sleep(CACHE_CHECK_INTERVAL)
            
        except Exception as e:
            print(f"❌ Background updater error: {e}")
            time.sleep(CACHE_CHECK_INTERVAL)

def initialize_background_services():
    """Initialize background services on startup"""
    print("🔄 Initializing background services...")
    
    cached_topics = load_topics_from_cache()
    if cached_topics:
        trending_cache['topics'] = cached_topics
        trending_cache['last_updated'] = datetime.now()
        print("✅ Loaded trending topics from cache")
    else:
        update_trending_topics()
    
    scheduler_thread = threading.Thread(target=background_trending_updater, daemon=True)
    scheduler_thread.start()
    print("✅ Background services started")

# Recommendation Functions
def fetch_interacted_post_ids(user_id):
    """Fetch post IDs that the user has interacted with"""
    if posts_collection is None:
        return []
    
    try:
        user_id_obj = ObjectId(user_id)
        posts = posts_collection.aggregate([
            {"$match": {"$or": [
                {"likes": user_id_obj},
                {"replies.userId": user_id_obj},
                {"postedBy": user_id_obj}
            ]}},
            {"$project": {"_id": 1}}
        ])
        return [str(post['_id']) for post in posts]
    except Exception as e:
        print(f"Error fetching interacted posts: {e}")
        return []

def fetch_top_recent_posts(user_id):
    """Fetch top recent posts based on interactions"""
    if posts_collection is None:
        return []
    
    try:
        start_date = datetime.now() - timedelta(days=1)
        interacted_ids = fetch_interacted_post_ids(user_id)
        
        recent_posts = posts_collection.find({
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
    except Exception as e:
        print(f"Error fetching top recent posts: {e}")
        return []

def analyze_user_sentiments(user_id):
    """Analyze user sentiments based on their interactions"""
    if posts_collection is None:
        return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
    
    try:
        start_time = datetime.now() - timedelta(days=1)
        user_id_obj = ObjectId(user_id) 
        user_interactions = posts_collection.find({
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
    except Exception as e:
        print(f"Error analyzing user sentiments: {e}")
        return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}

def fetch_user_content(user_id):
    """Fetch content created or interacted with by the user"""
    if posts_collection is None:
        return []
    
    try:
        user_id_obj = ObjectId(user_id)
        content = []
        
        user_posts = posts_collection.find({"postedBy": user_id_obj})
        content.extend(post.get("text", "") for post in user_posts if post.get("text"))
        
        liked_posts = posts_collection.find({"likes": user_id_obj})
        content.extend(post.get("text", "") for post in liked_posts if post.get("text"))
        
        replies = posts_collection.find({"replies.userId": user_id_obj})
        for post in replies:
            for reply in post.get("replies", []):
                if reply.get("userId") == user_id_obj and reply.get("text"):
                    content.append(reply.get("text"))
        
        return content
    except Exception as e:
        print(f"Error fetching user content: {e}")
        return []

def content_based_recommendations(user_id):
    """Generate content-based recommendations"""
    if posts_collection is None:
        return []
    
    try:
        user_id_obj = ObjectId(user_id)
        interacted_ids = fetch_interacted_post_ids(user_id)
        user_content = fetch_user_content(user_id)
        
        if not user_content:
            return []

        all_posts = list(posts_collection.find({
            "postedBy": {"$ne": user_id_obj},
            "_id": {"$nin": [ObjectId(id) for id in interacted_ids]}
        }, {"text": 1}))

        all_texts = [post["text"] for post in all_posts if post.get("text")]
        post_ids = [str(post["_id"]) for post in all_posts if post.get("text")]

        if not all_texts:
            return []

        tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
        user_tfidf = tfidf_vectorizer.transform(user_content)

        cosine_similarities = linear_kernel(user_tfidf, tfidf_matrix)
        top_indices = cosine_similarities.sum(axis=0).argsort()[-10:][::-1]
        recommended_posts = [post_ids[index] for index in top_indices if index < len(post_ids)]

        return recommended_posts
    except Exception as e:
        print(f"Error generating content recommendations: {e}")
        return []

def filter_negative_posts(post_ids):
    """Filter out posts with negative sentiment"""
    if posts_collection is None:
        return []
    
    filtered_posts = []
    for post_id in post_ids:
        try:
            post = posts_collection.find_one({"_id": ObjectId(post_id)})
            if post:
                text = post.get('text', '')
                sentiment_scores = analyzer.polarity_scores(text)
                if sentiment_scores['compound'] >= 0:
                    filtered_posts.append(post_id)
        except Exception as e:
            print(f"Error filtering post {post_id}: {e}")
            continue
    return filtered_posts

# ===== ROUTES =====

@app.route('/api/hate-speech/predict', methods=['POST'])
def predict_hate_speech_endpoint():
    """ API endpoint for hate speech prediction """
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Text field is required'}), 400
    
    result = predict_hate_speech(text)
    return jsonify(result)

@app.route('/api/hate-speech/health', methods=['GET'])
def hate_speech_health():
    """ Health check endpoint for hate speech detection """
    status = "ready" if model is None else "loaded"
    return jsonify({'status': status, 'model': 'toxic-bert (lazy-loaded)'})

@app.route('/api/trending', methods=['GET'])
def get_trending_topics():
    """Get trending topics from cache - instant response"""
    try:
        force_refresh = request.args.get('refresh', '').lower() == 'true'
        
        if force_refresh and not trending_cache['is_updating']:
            update_thread = threading.Thread(target=update_trending_topics, daemon=True)
            update_thread.start()
            return jsonify(trending_cache['topics'])
        
        return jsonify(trending_cache['topics'])
        
    except Exception as e:
        return jsonify({"error": f"Error fetching trending topics: {str(e)}"}), 500

@app.route('/api/trending/refresh', methods=['POST'])
def refresh_trending_topics():
    """Manually trigger trending topics refresh"""
    if trending_cache['is_updating']:
        return jsonify({"message": "Update already in progress"}), 409
    
    update_thread = threading.Thread(target=update_trending_topics, daemon=True)
    update_thread.start()
    
    return jsonify({
        "message": "Trending topics refresh started",
        "current_topics": trending_cache['topics']
    })

@app.route('/api/recommendations/<user_id>', methods=['GET'])
def get_user_sentiments_and_top_posts(user_id):
    """Get recommendations based on user sentiments and interactions"""
    if posts_collection is None:
        return jsonify({"error": "Database not available"}), 500
    
    try:
        if not ObjectId.is_valid(user_id):
            return jsonify({"error": "Invalid user ID format"}), 400
        
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

        if emotion == 'negative':
            final_recommendations = filter_negative_posts(combined_posts)
        else:
            final_recommendations = combined_posts

        response = {
            'user_sentiments': user_sentiments,
            'emotion': emotion,
            'recommendations': final_recommendations
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Error generating recommendations: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """ Overall health check endpoint """
    db_status = "connected" if db else "disconnected"
    model_status = "loaded" if model else "ready (lazy-load)"
    
    return jsonify({
        'status': 'ok', 
        'services': {
            'database': db_status,
            'hate_speech_detection': model_status,
            'trending_topics': 'active', 
            'recommendations': 'active'
        },
        'trending_cache': {
            'last_updated': trending_cache['last_updated'].isoformat() if trending_cache['last_updated'] else None,
            'topic_count': len(trending_cache['topics']),
            'is_updating': trending_cache['is_updating']
        }
    })

@app.route('/', methods=['GET'])
def home():
    """ Home endpoint with API information """
    return jsonify({
        "message": "Combined API Server - Memory Optimized",
        "endpoints": {
            "hate_speech_detection": {
                "predict": "POST /api/hate-speech/predict",
                "health": "GET /api/hate-speech/health"
            },
            "trending_topics": {
                "get": "GET /api/trending",
                "refresh": "POST /api/trending/refresh"
            },
            "recommendations": "GET /api/recommendations/<user_id>",
            "health": "GET /api/health"
        }
    })

if __name__ == '__main__':
    initialize_background_services()
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)