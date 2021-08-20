# Sentiment-Analysis-API
Objective:
  
  You need to create an API endpoint that can accept a text and return associated sentiment with it.
  
Dataset:

  You can find the training dataset in arline_sentiment_analysis.csv file.
  
Project:

  1. Build a binary classification model using any library of your choice.
  2. Develop an API server on python using Pyramid web framework (https://docs.pylonsproject.org/projects/pyramid/en/latest/).
  3. Implement APIs that can accept an english text and respond with the predicted sentiment.
  4. Use OOPs concept to train the model, reading data for training, and implement inference class.
  5. Upload the entire code to a newly created Git Repo.
  6. Integrate Swagger documentation (https://pypi.org/project/pyramid-swagger/) for your Rest API endpoint.
  7. A small 1 page report on what all models you experimented with, what the final metrics from model training and testing are. Also report what steps you tried to     tune your model.
  8. A screen recording video where you showcase that you are able to make API calls from your browser or an App like Postman and get a JSON response from your server.

# Files-Description

Train.py ---> Training the data after applying the required NLP techniques without any Under or Over sampling using OOPS approach

API.py ---> Uses the saved classifier from the Train.py and Undersampling.py to output the sentiment when request has been hit and to start the server, used an OOPS approach

API-1.py ---> Uses the saved classifier from the OverSampling.py and EntHire.py to output the sentiment when request has been hit and to start the server

EntHire.py ---> Training the data after applying the required NLP techniques without any Under or Over sampling without OOPS approach

finalized_model.sav ---> Saving the classifier in a file, for not training the data in the future. 

OverSampling.py ---> Since the no.of -ve samples >>> no.of +ve samples, we bring up the no.of +ve samples to that of -ve and then train the data

PostMan-Request.flv ---> Small Demo showcasing the JSON response when the request has been hit from Postman

Swagger Documentation.zip ---> Contains the Swagger documentation in Sentiment.html

Undersampling.py ---> Since the no.of -ve samples >>> no.of +ve samples, we bring down the no.of -ve samples to that of +ve and then train the data

airline_sentiment_analysis.csv ---> Given Dataset contained the Text and AirLine_Sentiment

request-hit.py ---> This part of code can be used to hit the requests from the IDE rather than using Postman or Browser 

tfidfconverter.pickle ---> Pickle file which contained the object of TfidfTransformer

vectorizer.pickle ---> Pickle file which contained the object of CountVectorizer

Training Results.pdf ---> The final metrics for different ML algorithms used
