# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:33:10 2020

@author: Nikhil K
"""

import pickle
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
import re, string
from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.view import view_config
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

def remove_noise(review_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(review_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

@view_config(renderer='json')
def get_sentiment(request):
    tknzr = TweetTokenizer()
    text1 = request.params.get('text', 'No Name Provided')
    custom_tokens = remove_noise(tknzr.tokenize(text1))
    Dict={}
    Dict['Text']=text1
    Dict['AirLine_Sentiment']=classifier.classify(dict([token, True] for token in custom_tokens))
    Dict['Confidence %']=classifier.confidence(dict([token, True] for token in custom_tokens))*100
    return Dict

filename = 'finalized_model.sav'
classifier = pickle.load(open(filename, 'rb'))
config = Configurator()
config.add_route('sentiment', u'/sentiment')
config.add_view(get_sentiment, route_name='sentiment',renderer='json')
app = config.make_wsgi_app()
server = make_server('0.0.0.0', 6543, app)
print('Server has started, hit the request')
server.serve_forever()