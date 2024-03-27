import csv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')


sia = SentimentIntensityAnalyzer()

dataset_file = 'C:/Users/Hamza/Desktop/train.csv'

dataset = []
with open(dataset_file, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        text = row[0]
        dataset.append(text)


for text in dataset:
    sentiment = sia.polarity_scores(text)
    compound_score = sentiment['compound']

    if compound_score >= 0.5:
        s = 'Very Positive'
    elif compound_score >= 0.05:
        s = 'Positive'
    elif compound_score <= -0.5:
        s = 'Very Negative'
    elif compound_score <= -0.05:
        s = 'Negative'
    else:
        s = 'Neutral'

    print(f'Text: {text}\nSentiment: {s}\n')
