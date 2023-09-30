from django.contrib import admin
from django.urls import path
from analyser import views

urlpatterns = [
    path('', views.home, name='index'),
    path('home', views.home, name='home'),
    path('result', views.result, name='result'),
    path('report', views.report, name='report'),
    path('model_refresh', views.model_refresh, name='model_refresh') # type: ignore

]
