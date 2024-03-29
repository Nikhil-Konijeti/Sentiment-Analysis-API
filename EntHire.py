# -*- coding: utf-8 -*-
"""EntHire.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19SdFFqjCxL9qS-NywGuKlEPuDoN_z2AN
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score

# Read train data
df1 = pd.read_csv('airline_sentiment_analysis.csv', header=None, names=['S.No','AirLine_Sentiment', 'Text'])
df=df1.iloc[1:]
df.drop(df.columns[0], axis=1,inplace=True) 
for i in range(1,len(df['Text'])+1):
  if(df['AirLine_Sentiment'][i]=='positive'):
    df['AirLine_Sentiment'][i]=1
  else:
    df['AirLine_Sentiment'][i]=0

# Cleaning the texts
import re
corpus = []
for i in range(1,len(df['Text'])+1):
    review = re.sub('[^a-zA-Z\ ]', '', df['Text'][i])
    review = review.lower()
    review = review.split()
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 0].values
y=y.astype('int')

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

from sklearn.svm import SVC
classifier1 = SVC(kernel = 'rbf', random_state = 0)
classifier1.fit(X_train, y_train)
y_pred = classifier1.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier2 = SVC(kernel = 'linear', random_state = 0)
classifier2.fit(X_train, y_train)
y_pred = classifier2.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier3 = LogisticRegression(random_state = 0)
classifier3.fit(X_train, y_train)
y_pred = classifier3.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier5 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier5.fit(X_train, y_train)
y_pred = classifier5.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier6 = GaussianNB()
classifier6.fit(X_train, y_train)
y_pred = classifier6.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier7 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier7.fit(X_train, y_train)
y_pred = classifier7.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier8 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier8.fit(X_train, y_train)
y_pred = classifier8.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)

filename = 'finalized_model.sav'
pickle.dump(classifier1, open(filename, 'wb'))
classifier = pickle.load(open(filename, 'rb'))

text = 'Congrats #SportStar on your 7th best goal from last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies'
print(text)
text = re.sub('[^a-zA-Z\ ]', '', text)
text = text.lower()
text = text.split()
text = ' '.join(text)
text = cv.transform([text]).toarray()
text = tfidfconverter.transform(text).toarray()
label = classifier.predict(text)[0]

if(label==0):
        print('Negative')
else:
        print('Positive')
        
pickle.dump(cv, open("vectorizer.pickle", "wb")) 
pickle.dump(tfidfconverter, open("tfidfconverter.pickle", "wb"))