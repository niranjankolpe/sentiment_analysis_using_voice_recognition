# sentiment_analysis_using_voice_recognition
A Machine Learning Project that predicts the sentiment of a user by analyzing his voice.

Steps to execute the project:
1. Install Python (version 3.11.4) on your computer. While installing, select the "Add to PATH" option to add the python installation to the system's environment variables.
2. On the desktop, create a new folder "Sentiment Analysis using Voice Recognition" and add all the project files to it. Open command prompt and type 'cd "Sentiment Analysis using Voice Recognition"'.
3. Type "python -m venv venv" to create a virtual environment named "venv". Now type "venv\Scripts\activate" to activate the virtual environment.
4. Install following Dependencies using "pip install dependency_name":
    i. django==4.2.3
    ii. nltk==3.8.1
    iii. pyttsx3==2.90
    iv. SpeechRecognition==3.10.0
    v. pandas==2.0.1
    vi. matplotlib==3.7.2
    vii. seaborn==0.12.2
    viii. wordcloud==1.9.2
    ix. scikit-learn==1.3.0
    x. pyaudio==0.2.13
5. Type the following commands to execute the project:
    i. python manage.py makemigrations
    ii. python manage.py migrate
    iii. python manage.py runserver
6. Open a browser (Google Chrome recommended) and type the url 127.0.0.1:8000.
7. The home page of the project will be opened and can be played around by the user.

