# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:58:19 2020

@author: Nikhil K
"""

import pandas as pd
import nltk
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import re, string, random
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI,SklearnClassifier
from statistics import mode
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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

def get_reviews_for_model(cleaned_tokens_list):
    for review_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in review_tokens)

if __name__ == "__main__":
    
    df1 = pd.read_csv('airline_sentiment_analysis.csv', header=None, names=['S.No','AirLine_Sentiment', 'Text'])
    df=df1.iloc[1:]
    df.drop(df.columns[0], axis=1,inplace=True)
    
    X = df.iloc[:, 1].values
    y = df.iloc[:, 0].values

    positive_reviews = []
    negative_reviews = []
    for i in range(0,len(X)):
        if(y[i]=='positive'):
            positive_reviews.append(X[i])
        else:
            negative_reviews.append(X[i])

    stop_words = stopwords.words('english')
    tknzr = TweetTokenizer()

    positive_review_tokens = [tknzr.tokenize(i) for i in positive_reviews]
    negative_review_tokens = [tknzr.tokenize(i) for i in negative_reviews]

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_review_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_review_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    positive_tokens_for_model = get_reviews_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_reviews_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(review_dict, "positive")
                         for review_dict in positive_tokens_for_model]

    negative_dataset = [(review_dict, "negative")
                         for review_dict in negative_tokens_for_model]

    x = positive_dataset + negative_dataset
    random.shuffle(x)
    training_set = x[:10000]
    testing_set = x[10000:]

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
    
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
    
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression(random_state = 0))
    LogisticRegression_classifier.train(training_set)
    print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
    
    SGDC_classifier = SklearnClassifier(SGDClassifier())
    SGDC_classifier.train(training_set)
    print("SGDC_classifier accuracy percent:", (nltk.classify.accuracy(SGDC_classifier, testing_set))*100)
    
    KernalSVC_classifier = SklearnClassifier(SVC(kernel = 'rbf', random_state = 0))
    KernalSVC_classifier.train(training_set)
    print("KernalSVC_classifier accuracy percent:", (nltk.classify.accuracy(KernalSVC_classifier, testing_set))*100)
    
    SVC_classifier = SklearnClassifier(SVC(kernel = 'linear', random_state = 0))
    SVC_classifier.train(training_set)
    print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)
    
    DecisionTree_classifier = SklearnClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 0))
    DecisionTree_classifier.train(training_set)
    print("DecisionTree_classifier accuracy percent:", (nltk.classify.accuracy(DecisionTree_classifier, testing_set))*100)
    
    RandomForest_classifier = SklearnClassifier(RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0))
    RandomForest_classifier.train(training_set)
    print("RandomForest_classifier accuracy percent:", (nltk.classify.accuracy(RandomForest_classifier, testing_set))*100)
    
    KNN_classifier = SklearnClassifier(KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2))
    KNN_classifier.train(training_set)
    print("KNN_classifier accuracy percent:", (nltk.classify.accuracy(KNN_classifier, testing_set))*100)
    
    voted_classifier = VoteClassifier(MNB_classifier,
                                 BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDC_classifier,
                                  KernalSVC_classifier,
                                  SVC_classifier,
                                  DecisionTree_classifier,
                                  RandomForest_classifier,
                                  KNN_classifier)
    print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
    
    filename = 'finalized_model.sav'
    pickle.dump(voted_classifier, open(filename, 'wb'))

    classifier = pickle.load(open(filename, 'rb'))

    custom_review= 'Congrats #SportStar on your 7th best goal from last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies'

    custom_tokens = remove_noise(tknzr.tokenize(custom_review))

    print(custom_review)
    print("Classification: ", voted_classifier.classify(dict([token, True] for token in custom_tokens)),"Confidence %: ",voted_classifier.confidence(dict([token, True] for token in custom_tokens))*100)
