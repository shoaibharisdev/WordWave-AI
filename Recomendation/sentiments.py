from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

@app.route('/sentiment', methods=['POST'])
def detect_sentiment():
    data = request.json
    text = data.get('text', '')

    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Perform sentiment analysis
    sentiment_scores = analyzer.polarity_scores(text)

    # Determine emotion based on sentiment scores
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        emotion = 'positive'
    elif compound_score <= -0.05:
        emotion = 'negative'
    else:
        emotion = 'neutral'

    # Prepare response
    response = {
        'text': text,
        'sentiment': sentiment_scores,
        'emotion': emotion
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=6000)
