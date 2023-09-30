import pyttsx3
import re,joblib
import speech_recognition as sr
import datetime,nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

engine =pyttsx3.init('sapi5')
voice = engine.getProperty('voices')
engine.setProperty('voices',voice[1].id)

questions  = [
    'how are you?',
    'do you like icecream',
]

lm = WordNetLemmatizer()
cv = joblib.load('countvectorizer')
rf = joblib.load('sentimental_model')

def text_transformation(df_col):
    corpus=[]
    for item in df_col:
        new_item=re.sub('[^a-zA-Z]',' ',str(item))
        new_item=new_item.lower()
        new_item=new_item.split()
        new_item=[lm.lemmatize(word)for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x)for x in new_item))
    return corpus

def expressin_check(prediction_input):
    if prediction_input==0:
        speak("Input statement has negative sentiment")
    elif prediction_input==1:
        speak("Input statement has positive sentiment")
    else:
        speak("invalid statement")

def sentiment_predictor(input):
    input=text_transformation(input)
    transformed_input=cv.transform(input)
    prediction=rf.predict(transformed_input)
    print(prediction)
    expressin_check(prediction)

def wishme():
    hour = int(datetime.datetime.now().hour)
    if hour >=0 and hour < 12:
        speak("good morning master")
    elif hour >=12 and hour <18:
        speak("good afternoon master")
    else:
        speak('good evening master')

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source,1)
        print('listening...')
        speak('listening')
        audio = r.listen(source)
    try:
        print('recognizing...')
        speak('analyzing the statement')
        query = r.recognize_google(audio,language='en')
        print(f'User said {query}')
    except sr.UnknownValueError:
        speak('could not hear you, try again')
    except sr.RequestError:
        speak('make sure you have a goog internet')
    
    return query

if __name__ == '__main__':
    wishme()
    for question in questions:
        speak(question.lower())
        query = takecommand().lower()
        sentiment_predictor([query])



