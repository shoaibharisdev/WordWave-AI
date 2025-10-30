from flask import Flask, jsonify
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
from tqdm import tqdm
from flask_cors import CORS
import nltk

# nltk.download('punkt')
# nltk.download('stopwords')
nltk.download('wordnet')

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)


# Define a set of meaningless words to remove
meaningless_words = {
    'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 
    'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 
    'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 
    'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 
    'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 
    'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 
    'do', 'done', 'down', 'due', 'during', 'each', 'eg','come','came','coming','live' 'living', 'eight', 'either', 'eleven', 'else', 'elsewhere', 
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
    'yours', 'yourself', 'yourselves',
    'theyre', 'their', 'shouldnt', 'wont', 'wouldnt', 'couldnt', 'cant', 'werent', 'arent', 'isnt', 
    'wasnt', 'havent', 'hasnt', 'hadnt', 'shouldve', 'couldve', 'wouldve', 'mightve', 'mustve', 'theres', 
    'heres', 'didnt', 'doesnt', 'dont', 'shant', 'shan', 'neither', 'nor', 'its', 'it', 'ill', 'im', 'ive', 
    'id', 'lets', 'thats', 'whats', 'whose', 'whos', 'shes', 'hes', 'whos', 'youd', 'youve', 'youd', 'youre', 
    'youll', 'yall', 'yalls', 'couldve', 'wouldve', 'shouldve', 'mightve', 'mustve', 'arent', 'isnt', 'wasnt', 
    'werent', 'dont', 'doesnt', 'didnt', 'wont', 'cant', 'shouldnt', 'couldnt', 'wouldnt', 'neednt', 'shant', 
    'shan', 'mustnt', 'theres', 'heres', 'wheres', 'whens', 'whys', 'hows', 'lets', 'used', 'using', 'use', 
    'many', 'much', 'a', 'an', 'some', 'any', 'more', 'most', 'all', 'both', 'every', 'either', 'neither', 
    'each', 'few', 'several', 'no', 'other', 'another', 'such', 'that', 'these', 'those', 'this', 'one', 'once', 
    'twice', 'and', 'but', 'or', 'nor', 'for', 'so', 'yet', 'not', 'only', 'with', 'without', 'also', 'too', 
    'very', 'just', 'now', 'then', 'than', 'even', 'ever', 'already', 'still', 'almost', 'often', 'sometimes', 
    'usually', 'always', 'never', 'perhaps', 'maybe', 'possibly', 'probably', 'really', 'quite', 'rather', 'so', 
    'such', 'too', 'enough', 'very', 'indeed', 'less', 'more', 'a lot', 'lots', 'kind of', 'sort of', 'seem', 
    'seems', 'appear', 'appears', 'might', 'may', 'must', 'should', 'could', 'would', 'ought', 'shall', 'will', 
    'can', 'cannot', 'couldnt', 'wouldnt', 'shouldnt', 'heres', 'theres', 'wheres', 'whens', 'whys', 'hows', 
    'they', 'he', 'she', 'we', 'you', 'theyre', 'im', 'youre', 'hes', 'shes', 'its', 'its', 'theyll', 'weve', 
    'theyve', 'couldve', 'wouldve', 'shouldve', 'mightve', 'mustve', 'youll', 'ive', 'youd', 'weve', 'theyve', 
    'youve', 'youd', 'id', 'would', 'could', 'should', 'might', 'must', 'did', 'does', 'do', 'dont', 'didnt', 
    'doesnt', 'doest', 'arent', 'isnt', 'wasnt', 'werent', 'havent', 'hasnt', 'hadnt', 'havent', 'hasnt', 
    'hadnt', 'wont', 'wont', 'dont', 'doesnt', 'didnt', 'dont', 'doesnt', 'didnt', 'shant', 'shan', 'neednt', 
    'mustnt', 'oughtnt', 'thats', 'whos', 'thats', 'whats', 'whos', 'whats', 'whos', 'thats', 'whats', 'whos', 
    'couldnt', 'wouldnt', 'shouldnt', 'mightnt', 'mustnt', 'shant', 'shouldnt', 'wouldnt', 'couldnt', 'wont', 
    'shall', 'should', 'shouldnt', 'ought', 'would', 'might', 'may', 'maynt', 'mightnt', 'must', 'mustnt', 
    'need', 'neednt', 'ought', 'oughtnt', 'shall', 'shant', 'should', 'shouldnt', 'will', 'wont', 'would', 
    'wouldnt', 'must', 'mustnt', 'may', 'might', 'mightnt', 'may', 'might', 'must', 'ought', 'oughtnt', 'shall', 
    'shant', 'should', 'shouldnt', 'will', 'wont', 'would', 'wouldnt', 'can', 'cant', 'could', 'couldnt', 
    'may', 'might', 'mightnt', 'must', 'mustnt', 'ought', 'oughtnt', 'shall', 'should', 'want','shouldnt', 'will', 
    'wont', 'would', 'wouldnt', 'need', 'neednt', 'dare', 'dares', 'daren', 'darent', 'shall', 'should', 
    'shouldnt', 'will', 'wont', 'would', 'wouldnt', 'can', 'cant', 'could', 'couldnt', 'may', 'might', 
    'mightnt', 'must', 'mustnt', 'ought', 'oughtnt', 'shall', 'shant', 'should', 'shouldnt', 'will', 'wont', 
    'would', 'wouldnt', 'need', 'neednt', 'dare', 'dares', 'daren', 'darent', 'ought', 'oughtnt', 'need', 
    'neednt', 'dare', 'dares', 'daren', 'darent', 'cant', 'cannot', 'cant', 'cannot', 'cant', 'cannot', 'cannot', 
    'cant', 'cannot'}

# MongoDB connection setup
def get_mongo_data():
    client = pymongo.MongoClient(os.getenv("MONGO_URI"), tlsCAFile=certifi.where())
    db = client[os.getenv("DB_NAME")]
    collection = db[os.getenv("COLLECTION_NAME")]
    return list(collection.find().sort("_id", pymongo.DESCENDING).limit(1000))

# Text preprocessing functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    # Remove meaningless words
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

# Flask route to get trending topics
@app.route('/trending', methods=['GET'])
def get_trending_topics():
    recent_data = get_mongo_data()
    df_twitter_recent = pd.DataFrame(recent_data)
    # Apply preprocessing to the 'description' column
    df_twitter_recent['text'] = df_twitter_recent['text'].apply(preprocess_text)

    # Create a dictionary and corpus for LDA analysis
    dictionary = corpora.Dictionary(df_twitter_recent['text'].str.split())
    corpus = [dictionary.doc2bow(text.split()) for text in df_twitter_recent['text']]

    # Train the LDA model
    lda_model = models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)

    topics_info = []
    for i in range(10):
        topic_words = lda_model.show_topic(i, topn=1)
        if topic_words:  # Check if topic_words is not empty
            topic_words = [word for word in topic_words if len(word[0]) > 3]  
            if topic_words:
                pattern = create_pattern_from_top_words_with_synonyms(lda_model, i)
                regex = re.compile(pattern)

                # Identify related posts using the compiled regex pattern
                related_posts = [post for post in df_twitter_recent['text'] if regex.search(post)]
                
                # Only include topics with more than 1 related post
                if len(related_posts) > 1:
                    topics_info.append({
                        "Topic": topic_words[0][0],
                        "Number of Related Posts": len(related_posts),  # Include the count of related posts
                        "Related Posts": related_posts
                    })

    return jsonify(topics_info)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
