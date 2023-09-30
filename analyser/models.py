from django.db import models
import datetime

# Create your models here.
class SentimentData(models.Model):
    id = models.AutoField(primary_key=True)
    question = models.TextField(default="No question")
    audio_input = models.TextField(default="No input")
    prediction = models.TextField(default="No prediction")
    date_time = models.DateTimeField(default=datetime.datetime.now())
    
    def __str__(self):
        return self.prediction