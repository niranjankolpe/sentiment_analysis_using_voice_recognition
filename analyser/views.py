from django.shortcuts import render

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')
import pickle
import joblib

import datetime
from analyser.models import SentimentData
# import warnings
# warnings.filterwarnings('ignore')
import pyttsx3
import speech_recognition as sr
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('wordnet')
import nltk
nltk.download('stopwords')
#sklearn package
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



# Create your views here.
def home(request):
    return render(request, 'index.html')

def custom_encoder(df):
        df.replace(to_replace="surprise", value=1, inplace=True)
        df.replace(to_replace="love",     value=1, inplace=True)
        df.replace(to_replace="joy",      value=1, inplace=True)
        df.replace(to_replace="sadness",  value=0, inplace=True)
        df.replace(to_replace="anger",    value=0, inplace=True)
        df.replace(to_replace="fear",     value=0, inplace=True)
        return df

def text_transformation(df_col):
        lm = WordNetLemmatizer()
        corpus = []
        for item in df_col:
            new_item = re.sub('[^a-zA-Z]',' ',str(item))
            new_item = new_item.lower()
            new_item = new_item.split()
            new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
            corpus.append(' '.join(str(x) for x in new_item))
        return corpus

def model_refresh(request):
    print("\n\n\nModel refreshing started!\n")

    df_train = pd.read_csv("static/train.txt", delimiter=';', names=['text', 'label'])
    df_val   = pd.read_csv('static/val.txt',   delimiter=';', names=['text', 'label'])
    df = pd.concat([df_train, df_val])
    df.reset_index(inplace=True, drop=True)
    df['label'] = custom_encoder(df['label'])
    corpus = text_transformation(df['text'])
    cv = CountVectorizer(ngram_range = (1, 2))
    traindata = cv.fit_transform(corpus)
    x = traindata
    y = df.label

    rf=RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, min_samples_split=5, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                              max_features='sqrt', # type: ignore
                              max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                              verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

    rf.fit(x, y)
    
    input_statement="sometimes I just want to punch someone in the face."
    input = text_transformation([input_statement])
    transformed_input = cv.transform(input)
    prediction = rf.predict(transformed_input)
    
    with open('static/django_joblib_sentimental_model','wb') as f1:
        joblib.dump(rf, f1)
    
    with open('static/django_joblib_counter_vector_model','wb') as f2:
        joblib.dump(cv, f2)
    
    with open('static/django_pickle_sentimental_model','wb') as f3:
        pickle.dump(rf, f3)
    
    with open('static/django_pickle_counter_vector_model','wb') as f4:
        pickle.dump(cv, f4)

    print("Prediction: {0}".format(prediction))
    print("\nPrediction: {0}".format(prediction[0]))
    print("\n\nModel Refreshed!\n\n\n")
    return render(request, 'index.html')


def predictSentiment(text):
    cv_model = joblib.load("static/django_joblib_counter_vector_model")
    text = cv_model.transform(text)
    del cv_model
    
    rf_model = joblib.load("static/django_joblib_sentimental_model")
    prediction = rf_model.predict(text)
    del rf_model
    
    if prediction==0:
        prediction = "Negative"
    elif prediction==1:
        prediction = "Positive"
    else:
        prediction = "Invalid Input"
    
    return prediction

def sentiment_predictor(text_input):
    transformed_input = text_transformation([text_input])
    prediction = predictSentiment(transformed_input)
    return prediction

def speak(audio):
    engine = pyttsx3.init('sapi5')
    voice = engine.getProperty('voices')
    engine.setProperty('voices',voice[1].id)
    engine.say(audio)
    engine.runAndWait()
    return True

def takecommand(question):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, 1)
        speak(question.lower())
        print('Listening...')
        speak("listening")
        audio = r.listen(source)
        time.sleep(2)
        speak('recognizing')
    try:
        time.sleep(2)
        query = r.recognize_google(audio, language='en')
        print("You said: {0}".format(query))
        return query
    except sr.UnknownValueError:
        speak('could not hear you, try again')
        return False
    except sr.RequestError:
        speak('make sure you have a good internet')
        return False

def result(request):
    # questions  = ['how are you', 'do you like icecream']
    # query_list = list()
    # result_list = list()
    # for question in questions:
    #     query = takecommand(question)
    #     query_list.append(query)
    #     result = sentiment_predictor([query])
    #     result_list.append(result)
    #     # text_input = str(request.POST['text_input'])
    #     # result = sentiment_predictor(text_input)
    #     print("\nAudio Input: {0}".format(query))
    #     print("\nPredicted Sentiment: {0}\n".format(result))
    # audio_inputs = str(query_list)
    # predicted_value = str(result_list)


    question  = 'how are you'
    audio_input = takecommand(question)
    prediction = sentiment_predictor([audio_input])
    
    entry = SentimentData(question=question, audio_input=audio_input, prediction=prediction, date_time=datetime.datetime.now())
    entry.save()

    data = {'input': audio_input, 'prediction': prediction}
    return render(request, 'result.html', data)

def report(request):
    records = SentimentData.objects.all()
    data = {'data':records}
    return render(request, 'report.html', data)