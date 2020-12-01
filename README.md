# Sentiment-Analysis-API
An API endpoint that can accept a text and return associated sentiment with it. 

# Files-Description

Train.py ---> Training the data after applying the required NLP techniques without any Under or Over sampling using OOPS approach

API-1.py ---> Uses the saved classifier from the OverSampling.py and EntHire.py to output the sentiment when request has been hit 

API.py ---> Uses the saved classifier from the Undersampling.py to output the sentiment when request has been hit 

EntHire ML+Backend Project.pdf ---> Given task and the Output requirements

EntHire.py ---> Training the data after applying the required NLP techniques without any Under or Over sampling without OOPS approach

OverSampling.py ---> Since the no.of -ve samples >>> no.of +ve samples, we bring up the no.of +ve samples to that of -ve and then train the data

PostMan-Request.flv ---> Small Demo showcasing the JSON response when the request has been hit from Postman

Swagger Documentation.zip ---> Contains the Swagger documentation in Sentiment.html

Undersampling.py ---> Since the no.of -ve samples >>> no.of +ve samples, we bring down the no.of -ve samples to that of +ve and then train the data

airline_sentiment_analysis.csv ---> Given Dataset contained the Text and AirLine_Sentiment

request-hit.py ---> This part of code can be used to hit the requests from the IDE rather than using Postman or Browser 

tfidfconverter.pickle ---> Pickle file which contained the object of TfidfTransformer

vectorizer.pickle ---> Pickle file which contained the object of CountVectorizer
